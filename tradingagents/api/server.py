import threading
from datetime import date as date_cls
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cli.utils import detect_asset_type
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
        selected = tuple(req.analysts) if req.analysts else (
            "market", "social", "news", "fundamentals", "quant"
        )
        graph = TradingAgentsGraph(selected_analysts=selected, config=DEFAULT_CONFIG)
        trade_date = req.date or date_cls.today().strftime("%Y-%m-%d")
        asset_type = detect_asset_type(req.ticker)

        final_state, _signal = graph.propagate(req.ticker, trade_date, asset_type=asset_type)

        reports = {k: final_state[k] for k in REPORT_KEYS if final_state.get(k)}
        _jobs.update_job(
            job_id,
            status="done",
            reports=reports,
            reports_completed=len(reports),
            reports_total=len(REPORT_KEYS),
            final_decision=final_state.get("final_trade_decision"),
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure to the poller
        _jobs.update_job(job_id, status="error", error=str(exc))


@app.post("/analyze", status_code=202, response_model=AnalyzeResponse)
def post_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    job_id = _jobs.create_job()
    _jobs.update_job(job_id, reports_total=len(REPORT_KEYS))
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
