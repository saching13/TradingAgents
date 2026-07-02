# Kronos Quant Forecast + HTTP API + Hermes Integration

Date: 2026-07-02
Status: Approved by user, proceeding to implementation

## Context

TradingAgents is a multi-agent LLM trading framework (analysts -> bull/bear
researchers -> research manager -> trader -> risk team -> portfolio manager),
run either via an interactive CLI/TUI or programmatically. The user also runs
a separate always-on assistant, "Hermes" (NousResearch hermes-agent, in
Docker, reachable via Telegram/Discord/etc. and a dashboard), which has its
own markdown-based skills system for extending what the assistant can do in
conversation.

Kronos (github.com/shiyu-coder/Kronos) is an open-source foundation model for
financial K-line (OHLCV) forecasting: a tokenizer + autoregressive transformer
that predicts a future OHLCV path from historical bars. It ships as plain
Python (no PyPI package) with model weights on Hugging Face Hub
(`NeoQuasar/Kronos-small` + `NeoQuasar/Kronos-Tokenizer-base` chosen for this
integration: 24.7M params, 512-bar context, good quality/speed balance for a
daily-cadence pipeline).

## Goal

Add Kronos to TradingAgents as a new "Quant Forecast Analyst" node producing
real predicted price paths (not just indicators), make the whole pipeline
callable over HTTP instead of only through the interactive TUI, run
everything in the existing Docker setup with GPU passthrough (host already
has nvidia-container-toolkit + `nvidia` runtime configured), and let the
user query results conversationally through Hermes.

## Components

### 1. Vendored Kronos model code

`tradingagents/vendor/kronos/{__init__.py,kronos.py,module.py}` — copied
verbatim (MIT license, `NOTICE` file with attribution + link to upstream)
from the Kronos repo's `model/` directory. Not pip-installable upstream, so
vendoring is simpler and more transparent than a git submodule for two files.

### 2. Config

New `default_config.py` block:

```python
"kronos_forecast": {
    "enabled": True,
    "model": "NeoQuasar/Kronos-small",
    "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
    "device": "auto",       # "auto" picks cuda if available else cpu
    "lookback": 400,
    "pred_len": 5,          # trading days ahead
    "temperature": 1.0,
    "top_p": 0.9,
    "sample_count": 1,
}
```

### 3. Dataflow

`tradingagents/dataflows/kronos_forecast.py`:
- `load_kronos(model_name, tokenizer_name, device)` — lazy singleton loader
  (module-level cache) for tokenizer + model + `KronosPredictor`.
- `get_kronos_forecast(symbol, curr_date, pred_len=5, lookback=400) -> dict`
  — reuses `load_ohlcv(symbol, curr_date)` (same cached, look-ahead-safe
  source the indicator tools use) for history, runs the predictor, returns
  structured result: predicted daily OHLCV path + derived stats (% change,
  direction, path high/low).

### 4. Tool + Analyst

- `tradingagents/agents/utils/kronos_tools.py`: `@tool
  get_kronos_forecast(symbol, curr_date)`, formatted-text return, same
  convention as `get_indicators`.
- `tradingagents/agents/analysts/quant_analyst.py`: `create_quant_analyst(llm)`,
  modeled on `market_analyst.py` — single tool, system prompt explains the
  model is a probabilistic forecast (not a guarantee), asks the analyst to
  report the path, note single-sample vs. multi-sample variance, and flag
  disagreement with the Technical Analyst rather than silently reconciling.

### 5. Graph wiring

- `ANALYST_NODE_SPECS["quant"]` in `analyst_execution.py`.
- `agent_states.py`: `quant_report` field.
- `propagation.py`: init `quant_report: ""`.
- `trading_graph.py`: include in final state assembly + `_create_tool_nodes`.
- `conditional_logic.py`: `should_continue_quant`.
- `setup.py`: `analyst_factories["quant"]`.
- `cli/main.py`: `ANALYST_ORDER`, `ANALYST_REPORT_MAP`, analyst-selection menu.
- Default `selected_analysts` tuple gains `"quant"` (user chose on-by-default
  despite the new torch dependency).

### 6. New dependencies

`torch`, `einops`, `safetensors`, `huggingface_hub` (may already be present;
dedupe) added to `pyproject.toml`. This is the project's first local-ML
dependency — previously pure LLM-API calls.

### 7. HTTP API

`tradingagents/api/server.py` (FastAPI + uvicorn), new CLI subcommand
`tradingagents api` (`--host`, `--port`, default `0.0.0.0:8080`):
- `POST /analyze` body `{ticker, date?, depth?, analysts?}` -> `202
  {job_id}`. Runs the graph in a background task; in-memory job registry
  (no Redis/Celery — single-user local deployment). Reuses existing
  report-saving path on completion.
- `GET /analyze/{job_id}` -> `{status: queued|running|done|error,
  reports_completed, reports_total, reports: {...as available...},
  final_decision}` (mirrors the CLI's existing progress model).
- `GET /health` -> `{status: "ok"}`.

### 8. Docker

- `Dockerfile`: base stays `python:3.12-slim` (PyPI torch wheels bundle CUDA
  runtime; no need for an nvidia/cuda base image). Builder stage installs
  the new deps via the existing `pip install .`.
- `docker-compose.yml`: `tradingagents` service gains a GPU reservation
  (`deploy.resources.reservations.devices` with `capabilities: [gpu]`), a
  new `huggingface_cache:/home/appuser/.cache/huggingface` volume (persist
  the ~100-500MB Kronos weights across container recreation), and port
  `8080:8080` exposed. `ollama`/`tradingagents-ollama` services untouched.

### 9. Claude Code `/analyze` skill update

`.claude/skills/analyze/SKILL.md`: replace the tmux `source
.venv/bin/activate && tradingagents` TUI-driving flow with `POST /analyze`
then poll `GET /analyze/{job_id}` every ~10s, reporting
`reports_completed`/`reports_total` the same way. Report-saving step at the
end is unchanged. Bump the doc's example `Reports: N/7` to `N/8` (5th
analyst added).

### 10. New Hermes skill

New category `~/.hermes/skills/trading/` with `DESCRIPTION.md` (matching the
existing `mlops`/`devops`/`productivity` category convention) and
`trading/tradingagents/SKILL.md` (frontmatter matching Hermes's schema:
`name, description, version, author, license, tags, platforms`). Instructs
the agent to `curl POST http://localhost:8080/analyze`, poll `GET
/analyze/{job_id}` until `done`, then **quote the actual recommendation /
rationale / Kronos forecast content in the chat reply** — explicit
instruction not to just reference a file path. This is a separate front
door onto the same API as `/analyze`, not a duplicate of its logic: `/analyze`
is for driving a run from inside a Claude Code session on this repo; the
Hermes skill is for asking conversationally from any channel Hermes bridges
to (phone, etc.), with the emphasis on inline content over paths.

## Testing

- **Kronos dataflow**: unit tests with a stubbed/mocked `KronosPredictor`
  (avoid requiring GPU/network in CI) verifying `get_kronos_forecast` shapes
  its output correctly from a synthetic OHLCV frame; one `@pytest.mark.
  integration` test that actually downloads `Kronos-small` and runs a real
  forecast on cached sample data, skipped unless a
  `KRONOS_INTEGRATION_TEST=1` env var is set (mirrors the existing
  `integration`/`smoke` marker convention in `pyproject.toml`).
- **Graph wiring**: extend existing `tests/test_analyst_execution.py` /
  `tests/test_capabilities.py`-style tests to cover the `quant` key.
- **HTTP API**: `fastapi.testclient.TestClient` tests for `POST /analyze` ->
  job_id, `GET /analyze/{job_id}` state transitions, and error cases
  (unknown job id, invalid ticker), with the LangGraph run itself mocked out
  (no real LLM/GPU calls in these tests).
- **Docker**: `docker compose config` validation (catches YAML/schema
  errors) plus a manual `docker compose build` + smoke run as part of
  verification (not part of automated pytest suite).
- **Hermes skill**: manual verification only (ask Hermes a trading question,
  confirm it calls the API and quotes content) — no automated test harness
  for Hermes skills exists.

## Out of scope

- Backtesting Kronos itself (repo already has this upstream; not part of
  this integration).
- Any change to `alpha_vantage_indicator.py`, `alpha_vantage_stock.py`,
  `reddit.py`, `stockstats_utils.py`, or the existing one-line
  `default_config.py` diff already present as uncommitted WIP in this
  working tree — those are pre-existing changes unrelated to this work and
  must not be touched or reverted.
- Redis/Celery-based job queue (deferred — in-memory is sufficient for
  single-user local deployment; revisit if multi-user/durable-across-restart
  job state is ever needed).
