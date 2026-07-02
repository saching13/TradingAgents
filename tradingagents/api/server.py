import threading
from datetime import date as date_cls
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cli.models import AnalystType
from cli.utils import detect_asset_type, filter_analysts_for_asset_type
from tradingagents.api.jobs import JobStore
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

app = FastAPI(title="TradingAgents API")
_jobs = JobStore()

REPORT_KEYS = [
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "quant_report",
]

DEFAULT_ANALYSTS = ("market", "social", "news", "fundamentals", "quant")

# Mirrors cli/main.py's select_research_depth() DEPTH_OPTIONS mapping
# (shallow=1, medium=3, deep=5 rounds), so the API's "depth" field has the
# same meaning as the CLI's research-depth prompt.
DEPTH_ROUND_MAP = {"shallow": 1, "medium": 3, "deep": 5}

# Mirrors cli/main.py's ANALYST_REPORT_MAP.
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
    "quant": "quant_report",
}


class AnalyzeRequest(BaseModel):
    ticker: str
    date: str | None = None
    depth: str | None = None
    analysts: list[str] | None = None


class AnalyzeResponse(BaseModel):
    job_id: str


def _run_job(job_id: str, req: AnalyzeRequest) -> None:
    _jobs.update_job(job_id, status="running")
    try:
        requested = list(req.analysts) if req.analysts else list(DEFAULT_ANALYSTS)
        trade_date = req.date or date_cls.today().strftime("%Y-%m-%d")
        asset_type = detect_asset_type(req.ticker)

        # Match the CLI's crypto filtering (cli/main.py -> select_analysts ->
        # filter_analysts_for_asset_type): crypto tickers have no fundamentals
        # data, so drop that analyst regardless of an explicit request.
        filtered = filter_analysts_for_asset_type(
            [AnalystType(a) for a in requested], asset_type
        )
        selected = tuple(a.value for a in filtered)
        report_keys = [
            ANALYST_REPORT_MAP[a] for a in selected if a in ANALYST_REPORT_MAP
        ]
        _jobs.update_job(job_id, reports_total=len(report_keys))

        # Match the CLI's research-depth -> round-count mapping
        # (cli/utils.py's select_research_depth). Unrecognized/omitted depth
        # falls back to "deep", matching the README/skill's stated default.
        depth = (req.depth or "deep").lower()
        rounds = DEPTH_ROUND_MAP.get(depth, DEPTH_ROUND_MAP["deep"])
        config = dict(DEFAULT_CONFIG)
        config["max_debate_rounds"] = rounds
        config["max_risk_discuss_rounds"] = rounds

        graph = TradingAgentsGraph(selected_analysts=selected, config=config)

        final_state, _signal = graph.propagate(req.ticker, trade_date, asset_type=asset_type)

        reports = {k: final_state[k] for k in report_keys if final_state.get(k)}
        _jobs.update_job(
            job_id,
            status="done",
            reports=reports,
            reports_completed=len(reports),
            reports_total=len(report_keys),
            final_decision=final_state.get("final_trade_decision"),
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the poller
        _jobs.update_job(job_id, status="error", error=str(exc))


@app.post("/analyze", status_code=202, response_model=AnalyzeResponse)
def post_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    job_id = _jobs.create_job()
    thread = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    thread.start()
    return AnalyzeResponse(job_id=job_id)


@app.get("/analyze/{job_id}")
def get_analyze(job_id: str) -> dict[str, Any]:
    job = _jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
