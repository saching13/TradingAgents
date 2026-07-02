# Kronos Quant Forecast + HTTP API + Hermes Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Kronos as a new "Quant Forecast Analyst" to TradingAgents, expose the whole pipeline over an HTTP API, wire GPU-enabled Docker deployment, and let the user query it conversationally through Hermes.

**Architecture:** Vendor two small Kronos model files (MIT-licensed, no PyPI package upstream) into `tradingagents/vendor/kronos/`; add a dataflow function that reuses the existing cached/look-ahead-safe `load_ohlcv` helper to feed Kronos; wrap it as a LangChain tool consumed by a new analyst node wired into the existing LangGraph pipeline the same way `market_analyst` is; add a FastAPI server as a new CLI subcommand for programmatic access; update Docker Compose for GPU passthrough + weight caching; update the Claude Code `/analyze` skill and add a new Hermes skill, both as thin clients over the same HTTP API.

**Tech Stack:** Python 3.12, PyTorch (CPU/CUDA), FastAPI + uvicorn, LangGraph, existing TradingAgents conventions (LangChain `@tool`, questionary CLI, Docker Compose).

## Global Constraints

- Do not modify `tradingagents/dataflows/alpha_vantage_indicator.py`, `alpha_vantage_stock.py`, `reddit.py`, `stockstats_utils.py`, or the existing uncommitted one-line diff in `default_config.py` — pre-existing unrelated WIP in this working tree.
- `git add` specific files only when committing (never `-A`), so the pre-existing WIP stays uncommitted and untouched.
- New deps go in `pyproject.toml` `dependencies` (or a new optional extra if you decide GPU inference should be opt-in at install time — this plan adds them as core deps per the approved design, since "quant" is on by default).
- Kronos model choice: `NeoQuasar/Kronos-small` + `NeoQuasar/Kronos-Tokenizer-base`, `pred_len=5`, `lookback=400`.
- Follow existing code style: `ruff` config in `pyproject.toml` (line-length 100, target py310); run `ruff check` on changed files before committing.
- Baseline test state before this work: `436 passed, 7 failed (pre-existing, unrelated), 2 skipped`. Do not introduce new failures; the 7 pre-existing failures are out of scope.

---

### Task 1: Vendor the Kronos model code

**Files:**
- Create: `tradingagents/vendor/__init__.py` (empty)
- Create: `tradingagents/vendor/kronos/__init__.py`
- Create: `tradingagents/vendor/kronos/kronos.py`
- Create: `tradingagents/vendor/kronos/module.py`
- Create: `tradingagents/vendor/kronos/NOTICE`

**Interfaces:**
- Produces: `tradingagents.vendor.kronos.KronosTokenizer`, `tradingagents.vendor.kronos.Kronos`, `tradingagents.vendor.kronos.KronosPredictor` — used by Task 3.

- [ ] **Step 1: Fetch the two upstream files verbatim**

```bash
mkdir -p tradingagents/vendor/kronos
touch tradingagents/vendor/__init__.py
gh api repos/shiyu-coder/Kronos/contents/model/kronos.py --jq '.content' | base64 -d > tradingagents/vendor/kronos/kronos.py
gh api repos/shiyu-coder/Kronos/contents/model/module.py --jq '.content' | base64 -d > tradingagents/vendor/kronos/module.py
```

- [ ] **Step 2: Write the package `__init__.py`**

```python
from .kronos import Kronos, KronosPredictor, KronosTokenizer

__all__ = ["Kronos", "KronosPredictor", "KronosTokenizer"]
```

- [ ] **Step 3: Write the NOTICE file**

```
This directory vendors kronos.py and module.py verbatim from:
https://github.com/shiyu-coder/Kronos (MIT License, Copyright (c) 2025 ShiYu)

Kronos is not distributed as an installable package (no pyproject.toml/
setup.py upstream), so the two self-contained model files are vendored
here directly rather than depended on via git submodule or pip.

No modifications have been made to kronos.py or module.py.
```

- [ ] **Step 4: Verify the vendored module imports cleanly**

Run: `cd /home/sags/Projects/TradingAgents && source .venv/bin/activate && python -c "from tradingagents.vendor.kronos import Kronos, KronosTokenizer, KronosPredictor; print('ok')"`
Expected: fails with `ModuleNotFoundError: No module named 'torch'` (torch isn't installed yet — that's expected here; Task 7 adds it). If it fails with a different ImportError (e.g. `einops`), note it — Task 7 must add whichever vendored-file imports are missing.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/vendor/__init__.py tradingagents/vendor/kronos/
git commit -m "Vendor Kronos model code (MIT, from shiyu-coder/Kronos)"
```

---

### Task 2: Add Kronos config block

**Files:**
- Modify: `tradingagents/default_config.py`

**Interfaces:**
- Produces: `DEFAULT_CONFIG["kronos_forecast"]` dict with keys `enabled, model, tokenizer, device, lookback, pred_len, temperature, top_p, sample_count` — read by Task 3's `get_kronos_forecast`.

**Note:** This file already has one uncommitted line of unrelated WIP (per Global Constraints). Add your block without touching that existing diff line — read the file first to see its current state before editing.

- [ ] **Step 1: Read the file to confirm current state**

Run: `git diff tradingagents/default_config.py` — confirm the existing one-line change, then locate the closing `})` of `DEFAULT_CONFIG` to insert before it.

- [ ] **Step 2: Add the config block immediately before the final `})`**

```python
    # Kronos quant forecast (Quant Forecast Analyst). Kronos is an
    # autoregressive K-line foundation model (github.com/shiyu-coder/Kronos);
    # "small" balances quality/speed for a daily-cadence pipeline.
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
    },
})
```

- [ ] **Step 3: Verify the config loads**

Run: `python -c "from tradingagents.default_config import DEFAULT_CONFIG; print(DEFAULT_CONFIG['kronos_forecast'])"`
Expected: prints the dict with no errors.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/default_config.py
git commit -m "Add kronos_forecast config block"
```

Note: this stages only the hunk you added if using `git add -p`; since the file has a pre-existing unstaged line, prefer `git add -p tradingagents/default_config.py` and select only your new hunk, OR confirm with the user before staging the whole file. Default to `git add -p` and select only the new block.

---

### Task 3: Kronos dataflow function

**Files:**
- Create: `tradingagents/dataflows/kronos_forecast.py`
- Test: `tests/test_kronos_forecast.py`

**Interfaces:**
- Consumes: `tradingagents.dataflows.stockstats_utils.load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame` (columns: `Date, Open, High, Low, Close, Volume`), `tradingagents.dataflows.config.get_config() -> dict`.
- Produces: `get_kronos_forecast(symbol: str, curr_date: str, pred_len: int | None = None, lookback: int | None = None) -> dict` with keys `symbol, curr_date, pred_len, predicted: list[dict]` (each dict has `date, open, high, low, close, volume`), `pct_change_close: float`, `direction: "up"|"down"|"flat"`, `path_high: float`, `path_low: float`. Used by Task 4's tool wrapper.
- Produces: `load_kronos(model_name: str, tokenizer_name: str, device: str) -> KronosPredictor` — lazy singleton, mockable in tests via monkeypatching `tradingagents.dataflows.kronos_forecast.load_kronos`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kronos_forecast.py
from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest


def _fake_ohlcv(rows: int = 450) -> pd.DataFrame:
    dates = pd.date_range(end=datetime(2026, 6, 30), periods=rows, freq="D")
    return pd.DataFrame({
        "Date": dates,
        "Open": [100.0 + i * 0.1 for i in range(rows)],
        "High": [101.0 + i * 0.1 for i in range(rows)],
        "Low": [99.0 + i * 0.1 for i in range(rows)],
        "Close": [100.5 + i * 0.1 for i in range(rows)],
        "Volume": [1_000_000 for _ in range(rows)],
    })


class _FakePredictor:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, df, x_timestamp, y_timestamp, pred_len, **kwargs):
        last_close = df["close"].iloc[-1]
        return pd.DataFrame({
            "open": [last_close + i for i in range(pred_len)],
            "high": [last_close + i + 1 for i in range(pred_len)],
            "low": [last_close + i - 1 for i in range(pred_len)],
            "close": [last_close + i + 0.5 for i in range(pred_len)],
            "volume": [1_000_000 for _ in range(pred_len)],
        })


@patch("tradingagents.dataflows.kronos_forecast.load_kronos")
@patch("tradingagents.dataflows.kronos_forecast.load_ohlcv")
def test_get_kronos_forecast_shape(mock_load_ohlcv, mock_load_kronos):
    mock_load_ohlcv.return_value = _fake_ohlcv()
    mock_load_kronos.return_value = _FakePredictor()

    from tradingagents.dataflows.kronos_forecast import get_kronos_forecast

    result = get_kronos_forecast("NVDA", "2026-06-30", pred_len=5, lookback=400)

    assert result["symbol"] == "NVDA"
    assert result["pred_len"] == 5
    assert len(result["predicted"]) == 5
    first = result["predicted"][0]
    assert set(first.keys()) == {"date", "open", "high", "low", "close", "volume"}
    assert result["direction"] in {"up", "down", "flat"}
    assert isinstance(result["pct_change_close"], float)
    assert result["path_high"] >= result["path_low"]


@patch("tradingagents.dataflows.kronos_forecast.load_kronos")
@patch("tradingagents.dataflows.kronos_forecast.load_ohlcv")
def test_get_kronos_forecast_direction_up(mock_load_ohlcv, mock_load_kronos):
    mock_load_ohlcv.return_value = _fake_ohlcv()
    mock_load_kronos.return_value = _FakePredictor()

    from tradingagents.dataflows.kronos_forecast import get_kronos_forecast

    result = get_kronos_forecast("NVDA", "2026-06-30", pred_len=3, lookback=400)
    # _FakePredictor always predicts a rising close path from last_close
    assert result["direction"] == "up"
    assert result["pct_change_close"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_kronos_forecast.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tradingagents.dataflows.kronos_forecast'`

- [ ] **Step 3: Write the implementation**

```python
# tradingagents/dataflows/kronos_forecast.py
"""Kronos quant forecast: wraps the vendored Kronos model as a TradingAgents
data source. Reuses load_ohlcv (same cached, look-ahead-safe source the
indicator tools use) for history, then runs KronosPredictor for a future
OHLCV path.
"""
from typing import Any

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.stockstats_utils import load_ohlcv

_PREDICTOR_CACHE: dict[tuple[str, str, str], Any] = {}


def load_kronos(model_name: str, tokenizer_name: str, device: str):
    """Lazily load and cache tokenizer + model + KronosPredictor.

    Cached per (model_name, tokenizer_name, device) so repeated calls in one
    process reuse the loaded weights instead of re-downloading/re-loading.
    """
    cache_key = (model_name, tokenizer_name, device)
    if cache_key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[cache_key]

    import torch

    from tradingagents.vendor.kronos import Kronos, KronosPredictor, KronosTokenizer

    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, device=resolved_device, max_context=512)

    _PREDICTOR_CACHE[cache_key] = predictor
    return predictor


def get_kronos_forecast(
    symbol: str,
    curr_date: str,
    pred_len: int | None = None,
    lookback: int | None = None,
) -> dict[str, Any]:
    """Predict a future OHLCV path for ``symbol`` as of ``curr_date``.

    Returns a dict with the predicted daily bars plus derived summary stats
    (direction, % change, path high/low) that the Quant Forecast Analyst's
    tool wrapper formats into text for the LLM.
    """
    config = get_config()
    kconf = config.get("kronos_forecast", {})
    pred_len = pred_len or kconf.get("pred_len", 5)
    lookback = lookback or kconf.get("lookback", 400)

    history = load_ohlcv(symbol, curr_date)
    history = history.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    context = history.tail(lookback).reset_index(drop=True)

    x_df = context[["open", "high", "low", "close", "volume"]]
    x_timestamp = context["Date"]
    last_date = pd.to_datetime(context["Date"].iloc[-1])
    y_timestamp = pd.Series(pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=pred_len
    ))

    predictor = load_kronos(
        kconf.get("model", "NeoQuasar/Kronos-small"),
        kconf.get("tokenizer", "NeoQuasar/Kronos-Tokenizer-base"),
        kconf.get("device", "auto"),
    )
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=kconf.get("temperature", 1.0),
        top_p=kconf.get("top_p", 0.9),
        sample_count=kconf.get("sample_count", 1),
        verbose=False,
    )

    predicted = [
        {
            "date": y_timestamp.iloc[i].strftime("%Y-%m-%d"),
            "open": round(float(pred_df["open"].iloc[i]), 4),
            "high": round(float(pred_df["high"].iloc[i]), 4),
            "low": round(float(pred_df["low"].iloc[i]), 4),
            "close": round(float(pred_df["close"].iloc[i]), 4),
            "volume": round(float(pred_df["volume"].iloc[i]), 2),
        }
        for i in range(len(pred_df))
    ]

    last_actual_close = float(context["close"].iloc[-1])
    final_pred_close = predicted[-1]["close"]
    pct_change = (final_pred_close - last_actual_close) / last_actual_close * 100

    if pct_change > 0.1:
        direction = "up"
    elif pct_change < -0.1:
        direction = "down"
    else:
        direction = "flat"

    return {
        "symbol": symbol,
        "curr_date": curr_date,
        "pred_len": pred_len,
        "predicted": predicted,
        "pct_change_close": round(pct_change, 4),
        "direction": direction,
        "path_high": max(p["high"] for p in predicted),
        "path_low": min(p["low"] for p in predicted),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_kronos_forecast.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add tradingagents/dataflows/kronos_forecast.py tests/test_kronos_forecast.py
git commit -m "Add Kronos dataflow function with mocked-predictor unit tests"
```

---

### Task 4: Kronos tool wrapper + agent_utils re-export

**Files:**
- Create: `tradingagents/agents/utils/kronos_tools.py`
- Modify: `tradingagents/agents/utils/agent_utils.py`
- Test: `tests/test_kronos_forecast.py` (append)

**Interfaces:**
- Consumes: `tradingagents.dataflows.kronos_forecast.get_kronos_forecast(symbol, curr_date, pred_len=None, lookback=None) -> dict` (Task 3).
- Produces: `get_kronos_forecast` LangChain `@tool` (name `get_kronos_forecast`) importable from `tradingagents.agents.utils.agent_utils` — consumed by Task 5's analyst and Task 6's `trading_graph.py`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_kronos_forecast.py

@patch("tradingagents.agents.utils.kronos_tools.get_kronos_forecast")
def test_kronos_forecast_tool_formats_text(mock_forecast):
    mock_forecast.return_value = {
        "symbol": "NVDA",
        "curr_date": "2026-06-30",
        "pred_len": 2,
        "predicted": [
            {"date": "2026-07-01", "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1000000},
            {"date": "2026-07-02", "open": 102.0, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 1000000},
        ],
        "pct_change_close": 1.5,
        "direction": "up",
        "path_high": 103.0,
        "path_low": 100.0,
    }

    from tradingagents.agents.utils.kronos_tools import get_kronos_forecast as forecast_tool

    text = forecast_tool.invoke({"symbol": "NVDA", "curr_date": "2026-06-30"})

    assert "NVDA" in text
    assert "2026-07-01" in text
    assert "up" in text
    assert "1.5" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_kronos_forecast.py::test_kronos_forecast_tool_formats_text -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tradingagents.agents.utils.kronos_tools'`

- [ ] **Step 3: Write the tool wrapper**

```python
# tradingagents/agents/utils/kronos_tools.py
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.kronos_forecast import get_kronos_forecast as _get_kronos_forecast


@tool
def get_kronos_forecast(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
) -> str:
    """
    Get a Kronos foundation-model price-path forecast for a ticker.

    Kronos is an autoregressive transformer trained on historical K-line
    (OHLCV) data; this is a probabilistic forecast, not a guarantee. Use it
    as one input among several, and note when it disagrees with
    indicator-based technical analysis rather than silently reconciling.

    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
    Returns:
        str: A formatted table of the predicted daily OHLCV path plus summary stats.
    """
    result = _get_kronos_forecast(symbol, curr_date)

    lines = [
        f"# Kronos forecast for {result['symbol']} as of {result['curr_date']}",
        f"# Predicted direction: {result['direction']} ({result['pct_change_close']:+.2f}% close change over {result['pred_len']} trading days)",
        f"# Path high: {result['path_high']} | Path low: {result['path_low']}",
        "",
        "date,open,high,low,close,volume",
    ]
    for bar in result["predicted"]:
        lines.append(
            f"{bar['date']},{bar['open']},{bar['high']},{bar['low']},{bar['close']},{bar['volume']}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Re-export from agent_utils.py**

In `tradingagents/agents/utils/agent_utils.py`, add the import alongside the other tool imports (near `from tradingagents.agents.utils.technical_indicators_tools import get_indicators`):

```python
from tradingagents.agents.utils.kronos_tools import get_kronos_forecast
```

Add `"get_kronos_forecast"` to the module's `__all__` list, next to `"get_indicators"`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_kronos_forecast.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add tradingagents/agents/utils/kronos_tools.py tradingagents/agents/utils/agent_utils.py tests/test_kronos_forecast.py
git commit -m "Add get_kronos_forecast LangChain tool"
```

---

### Task 5: Quant Forecast Analyst

**Files:**
- Create: `tradingagents/agents/analysts/quant_analyst.py`
- Modify: `tradingagents/agents/__init__.py`

**Interfaces:**
- Consumes: `get_kronos_forecast` tool, `get_instrument_context_from_state`, `get_language_instruction` from `tradingagents.agents.utils.agent_utils` (Task 4; existing helpers).
- Produces: `create_quant_analyst(llm) -> Callable[[state], dict]` node factory, matching the signature of `create_market_analyst` — consumed by Task 6's `setup.py`.

- [ ] **Step 1: Write the analyst factory**

```python
# tradingagents/agents/analysts/quant_analyst.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_kronos_forecast,
    get_language_instruction,
)


def create_quant_analyst(llm):

    def quant_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [get_kronos_forecast]

        system_message = (
            """You are a quantitative trading assistant. Your one job is to call get_kronos_forecast for this ticker and the current date, then report what it says.

Kronos is an autoregressive transformer foundation model trained on historical K-line (OHLCV) data across 45+ global exchanges. It produces a probabilistic forecast of the future price path, not a guarantee or a fundamental analysis. Treat a single sample_count=1 run as one draw from a distribution, not a certainty.

In your report:
- State the predicted direction, the % change in close over the forecast horizon, and the predicted path's high/low.
- Note the forecast horizon explicitly (how many trading days ahead).
- If this forecast's direction disagrees with what a reasonable technical read of recent price action would suggest, say so explicitly rather than silently blending the two views — flag the disagreement for the research team to weigh.
- Do not invent confidence levels or probabilities the tool did not report."""
            + """ Make sure to append a Markdown table at the end of the report showing the predicted daily OHLCV path."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "quant_report": report,
        }

    return quant_analyst_node
```

- [ ] **Step 2: Export from the agents package**

In `tradingagents/agents/__init__.py`, add:

```python
from .analysts.quant_analyst import create_quant_analyst
```

next to the `market_analyst` import, and add `"create_quant_analyst"` to `__all__`.

- [ ] **Step 3: Verify it imports**

Run: `source .venv/bin/activate && python -c "from tradingagents.agents import create_quant_analyst; print('ok')"`
Expected: prints `ok`

- [ ] **Step 4: Commit**

```bash
git add tradingagents/agents/analysts/quant_analyst.py tradingagents/agents/__init__.py
git commit -m "Add Quant Forecast Analyst node"
```

---

### Task 6: Wire the Quant analyst into the graph, CLI, and state

**Files:**
- Modify: `tradingagents/agents/utils/agent_states.py`
- Modify: `tradingagents/graph/propagation.py`
- Modify: `tradingagents/graph/conditional_logic.py`
- Modify: `tradingagents/graph/analyst_execution.py`
- Modify: `tradingagents/graph/setup.py`
- Modify: `tradingagents/graph/trading_graph.py`
- Modify: `cli/models.py`
- Modify: `cli/utils.py`
- Modify: `cli/main.py`
- Test: `tests/test_quant_analyst_wiring.py`

**Interfaces:**
- Consumes: `create_quant_analyst` (Task 5), `get_kronos_forecast` (Task 4).
- Produces: `AgentState.quant_report`, `ANALYST_NODE_SPECS["quant"]`, `AnalystType.QUANT`, default `selected_analysts` including `"quant"` — consumed by Task 8 (HTTP API uses the same `TradingAgentsGraph`).

- [ ] **Step 1: Write the failing wiring test**

```python
# tests/test_quant_analyst_wiring.py
def test_quant_in_analyst_node_specs():
    from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS

    assert "quant" in ANALYST_NODE_SPECS
    spec = ANALYST_NODE_SPECS["quant"]
    assert spec.report_key == "quant_report"
    assert spec.agent_node == "Quant Forecast Analyst"


def test_quant_report_field_in_agent_state():
    from tradingagents.agents.utils.agent_states import AgentState

    assert "quant_report" in AgentState.__annotations__


def test_quant_report_initialized_by_propagator():
    from tradingagents.graph.propagation import Propagator

    state = Propagator().create_initial_state("NVDA", "2026-06-30")
    assert state["quant_report"] == ""


def test_should_continue_quant_exists():
    from tradingagents.graph.conditional_logic import ConditionalLogic

    assert hasattr(ConditionalLogic(), "should_continue_quant")


def test_analyst_type_quant_exists():
    from cli.models import AnalystType

    assert AnalystType.QUANT == "quant"


def test_default_selected_analysts_includes_quant():
    import inspect

    from tradingagents.graph.trading_graph import TradingAgentsGraph

    sig = inspect.signature(TradingAgentsGraph.__init__)
    assert "quant" in sig.parameters["selected_analysts"].default
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && python -m pytest tests/test_quant_analyst_wiring.py -v`
Expected: FAIL (6 tests, `KeyError`/`AttributeError`/`AssertionError`)

- [ ] **Step 3: `agent_states.py`** — add the report field

In `tradingagents/agents/utils/agent_states.py`, in `AgentState`, add next to `fundamentals_report`:

```python
    quant_report: Annotated[str, "Report from the Quant Forecast Analyst (Kronos)"]
```

- [ ] **Step 4: `propagation.py`** — initialize the field

In `create_initial_state`'s returned dict, add next to `"fundamentals_report": ""`:

```python
            "quant_report": "",
```

- [ ] **Step 5: `conditional_logic.py`** — add the continue-check

Add this method next to `should_continue_fundamentals`:

```python
    def should_continue_quant(self, state: AgentState):
        """Determine if quant forecast analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_quant"
        return "Msg Clear Quant"
```

- [ ] **Step 6: `analyst_execution.py`** — add the node spec

In `ANALYST_NODE_SPECS`, add next to `"fundamentals"`:

```python
    "quant": AnalystNodeSpec(
        key="quant",
        agent_node="Quant Forecast Analyst",
        clear_node="Msg Clear Quant",
        tool_node="tools_quant",
        report_key="quant_report",
    ),
```

- [ ] **Step 7: `setup.py`** — register the factory and default selection

Add the import next to `create_market_analyst`:

```python
    create_quant_analyst,
```

Add to `analyst_factories` next to `"fundamentals"`:

```python
            "quant": lambda: create_quant_analyst(self.quick_thinking_llm),
```

Change the `setup_graph` default parameter:

```python
    def setup_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals", "quant")
    ):
```

- [ ] **Step 8: `trading_graph.py`** — tool node, default param, `_log_state`

Add the import next to `get_indicators`:

```python
    get_kronos_forecast,
```

Change the `__init__` default parameter (matching `setup.py`):

```python
        selected_analysts=("market", "social", "news", "fundamentals", "quant"),
```

Add to `_create_tool_nodes`'s returned dict, next to `"fundamentals"`:

```python
            "quant": ToolNode(
                [
                    get_kronos_forecast,
                ]
            ),
```

Add to `_log_state`, next to `"fundamentals_report"`:

```python
            "quant_report": final_state["quant_report"],
```

- [ ] **Step 9: `cli/models.py`** — add the enum member

```python
class AnalystType(str, Enum):
    MARKET = "market"
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    QUANT = "quant"
```

- [ ] **Step 10: `cli/utils.py`** — add to the selection menu

In `ANALYST_ORDER`, add:

```python
    ("Quant Forecast Analyst", AnalystType.QUANT),
```

- [ ] **Step 11: `cli/main.py`** — add to status tracking

In `ANALYST_ORDER` (line ~885):

```python
ANALYST_ORDER = ["market", "social", "news", "fundamentals", "quant"]
```

In `ANALYST_AGENT_NAMES`:

```python
    "quant": "Quant Forecast Analyst",
```

In `ANALYST_REPORT_MAP`:

```python
    "quant": "quant_report",
```

- [ ] **Step 12: Run tests to verify they pass**

Run: `source .venv/bin/activate && python -m pytest tests/test_quant_analyst_wiring.py -v`
Expected: PASS (6 tests)

- [ ] **Step 13: Run the full suite to check for regressions**

Run: `source .venv/bin/activate && python -m pytest -q`
Expected: same 7 pre-existing failures as baseline, no new failures; new test files pass.

- [ ] **Step 14: Commit**

```bash
git add tradingagents/agents/utils/agent_states.py tradingagents/graph/propagation.py \
  tradingagents/graph/conditional_logic.py tradingagents/graph/analyst_execution.py \
  tradingagents/graph/setup.py tradingagents/graph/trading_graph.py \
  cli/models.py cli/utils.py cli/main.py tests/test_quant_analyst_wiring.py
git commit -m "Wire Quant Forecast Analyst into graph, CLI, and default analyst set"
```

---

### Task 7: Add new dependencies

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: installable `torch`, `einops`, `safetensors`, `huggingface_hub` (Kronos), `fastapi`, `uvicorn[standard]` (Task 8's HTTP API).

- [ ] **Step 1: Add dependencies**

In `pyproject.toml`'s `dependencies` list, add (alphabetical, matching existing style):

```toml
    "einops>=0.8.1",
    "fastapi>=0.115.0",
    "huggingface-hub>=0.33.1",
    "safetensors>=0.6.2",
    "torch>=2.2.0",
    "uvicorn[standard]>=0.34.0",
```

- [ ] **Step 2: Install into the existing venv and verify**

Run: `source .venv/bin/activate && pip install -e . && python -c "import torch, fastapi, uvicorn, einops, safetensors, huggingface_hub; print(torch.cuda.is_available())"`
Expected: prints `True` (this machine has an RTX 3080 Ti with the driver/toolkit already configured); no import errors.

- [ ] **Step 3: Re-run Task 1's vendor import check now that torch is present**

Run: `python -c "from tradingagents.vendor.kronos import Kronos, KronosTokenizer, KronosPredictor; print('ok')"`
Expected: prints `ok`

- [ ] **Step 4: Run the full test suite again**

Run: `python -m pytest -q`
Expected: same as Task 6 Step 13 (no regressions from adding deps).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "Add torch/einops/safetensors/huggingface_hub/fastapi/uvicorn dependencies"
```

Note: `uv.lock` will now be stale (dependencies were added via `pip install -e .`, not `uv add`). If `uv` becomes available in this environment, run `uv lock` to refresh it; otherwise note in the commit body that `uv.lock` needs regeneration on a machine with `uv` installed.

---

### Task 8: HTTP API server

**Files:**
- Create: `tradingagents/api/__init__.py` (empty)
- Create: `tradingagents/api/server.py`
- Create: `tradingagents/api/jobs.py`
- Modify: `cli/main.py`
- Test: `tests/test_api_server.py`

**Interfaces:**
- Consumes: `tradingagents.graph.trading_graph.TradingAgentsGraph` (existing class; constructor `TradingAgentsGraph(selected_analysts=..., config=...)`, `.propagate(company_name, trade_date) -> (final_state, signal)` — verify exact method name in Task 8 Step 0 below).
- Produces: `tradingagents.api.jobs.JobStore` (in-memory registry: `create_job() -> str`, `get_job(job_id) -> dict | None`, `update_job(job_id, **fields)`), FastAPI `app` object in `tradingagents.api.server`, CLI subcommand `tradingagents api`.

- [ ] **Step 0: Graph run method (already confirmed)**

`tradingagents/graph/trading_graph.py:321` — `TradingAgentsGraph.propagate(self, company_name, trade_date, asset_type: str = "stock") -> tuple[dict, str]` (returns `(final_state, signal)`; `signal` comes from `self.process_signal(final_state["final_trade_decision"])`). `asset_type` defaults to `"stock"` but should be detected from the ticker the same way the CLI does, via `cli.utils.detect_asset_type(ticker) -> AssetType` (returns `AssetType.STOCK` or `AssetType.CRYPTO`, both `str` enums). Step 7 below uses this.

- [ ] **Step 1: Write the failing job-store test**

```python
# tests/test_api_server.py
def test_job_store_lifecycle():
    from tradingagents.api.jobs import JobStore

    store = JobStore()
    job_id = store.create_job()
    job = store.get_job(job_id)
    assert job["status"] == "queued"
    assert job["reports_completed"] == 0

    store.update_job(job_id, status="running", reports_completed=2)
    job = store.get_job(job_id)
    assert job["status"] == "running"
    assert job["reports_completed"] == 2

    assert store.get_job("does-not-exist") is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `source .venv/bin/activate && python -m pytest tests/test_api_server.py::test_job_store_lifecycle -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'tradingagents.api'`

- [ ] **Step 3: Write `jobs.py`**

```python
# tradingagents/api/jobs.py
"""In-memory job registry for the HTTP API. Single-user/local deployment —
no Redis/Celery needed; job state doesn't need to survive a process restart.
"""
import uuid
from typing import Any


class JobStore:
    def __init__(self):
        self._jobs: dict[str, dict[str, Any]] = {}

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {
            "status": "queued",
            "reports_completed": 0,
            "reports_total": 0,
            "reports": {},
            "final_decision": None,
            "error": None,
        }
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **fields: Any) -> None:
        if job_id in self._jobs:
            self._jobs[job_id].update(fields)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_api_server.py::test_job_store_lifecycle -v`
Expected: PASS

- [ ] **Step 5: Write the failing FastAPI endpoint tests (graph run mocked)**

```python
# append to tests/test_api_server.py
from unittest.mock import patch

from fastapi.testclient import TestClient


def _fake_final_state(ticker, date, asset_type="stock"):
    return {
        "market_report": "market ok",
        "quant_report": "quant ok",
        "final_trade_decision": "BUY",
    }, "BUY"


@patch("tradingagents.api.server.TradingAgentsGraph")
def test_post_analyze_returns_job_id(mock_graph_cls):
    mock_graph_cls.return_value.propagate.side_effect = (
        lambda ticker, date, asset_type="stock": _fake_final_state(ticker, date, asset_type)
    )

    from tradingagents.api.server import app

    client = TestClient(app)
    resp = client.post("/analyze", json={"ticker": "NVDA", "depth": "shallow"})

    assert resp.status_code == 202
    assert "job_id" in resp.json()


@patch("tradingagents.api.server.TradingAgentsGraph")
def test_get_analyze_job_completes(mock_graph_cls):
    mock_graph_cls.return_value.propagate.side_effect = (
        lambda ticker, date, asset_type="stock": _fake_final_state(ticker, date, asset_type)
    )

    from tradingagents.api.server import app

    client = TestClient(app)
    job_id = client.post("/analyze", json={"ticker": "NVDA"}).json()["job_id"]

    resp = client.get(f"/analyze/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["final_decision"] == "BUY"
    assert "quant ok" in body["reports"]["quant_report"]


def test_get_analyze_unknown_job_404():
    from tradingagents.api.server import app

    client = TestClient(app)
    resp = client.get("/analyze/does-not-exist")
    assert resp.status_code == 404


def test_health_endpoint():
    from tradingagents.api.server import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 6: Run to verify they fail**

Run: `python -m pytest tests/test_api_server.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'tradingagents.api.server'`

- [ ] **Step 7: Write `server.py`**

```python
# tradingagents/api/server.py
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
```

- [ ] **Step 8: Run to verify they pass**

Run: `python -m pytest tests/test_api_server.py -v`
Expected: PASS (5 tests). If the mocked `propagate` signature mismatch causes failures, adjust the mock's `side_effect` lambda signature to match what Step 0 found, not the other way around.

- [ ] **Step 9: Add the `tradingagents api` CLI subcommand**

In `cli/main.py`, add near the existing `analyze` command (after it, before `if __name__ == "__main__":`):

```python
@app.command()
def api(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host for the HTTP API."),
    port: int = typer.Option(8080, "--port", help="Bind port for the HTTP API."),
):
    """Run the TradingAgents HTTP API server (uvicorn)."""
    import uvicorn

    uvicorn.run("tradingagents.api.server:app", host=host, port=port)
```

- [ ] **Step 10: Verify the CLI subcommand is registered**

Run: `source .venv/bin/activate && tradingagents --help`
Expected: output lists both `analyze` and `api` commands (confirms the earlier single-command auto-invoke shortcut is now gone — Task 10 fixes the README/Docker references that relied on it).

- [ ] **Step 11: Run the full test suite**

Run: `python -m pytest -q`
Expected: same 7 pre-existing failures, no new ones; 8 new tests pass (3 job store + 5 API, adjust count to match Step 0's findings).

- [ ] **Step 12: Commit**

```bash
git add tradingagents/api/ tests/test_api_server.py cli/main.py
git commit -m "Add HTTP API server with async job queue for /analyze"
```

---

### Task 9: Docker + docker-compose GPU/API wiring

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Dockerfile` (only if Step 1 finds missing system deps for torch)

**Interfaces:** None (infrastructure only; consumes the `tradingagents api` subcommand from Task 8).

- [ ] **Step 1: Check whether the slim Python image needs extra system packages for torch**

Run: `docker run --rm python:3.12-slim bash -c "pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -5"`
If this fails on missing shared libraries (e.g. `libgomp`), add to the `Dockerfile`'s final stage, before the `USER appuser` line:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*
```
If it succeeds cleanly, skip this — no Dockerfile change needed.

- [ ] **Step 2: Rewrite `docker-compose.yml`**

```yaml
services:
  tradingagents:
    build: .
    env_file:
      - .env
    command: ["analyze"]
    volumes:
      - tradingagents_data:/home/appuser/.tradingagents
      - huggingface_cache:/home/appuser/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    tty: true
    stdin_open: true

  tradingagents-api:
    build: .
    env_file:
      - .env
    command: ["api", "--host", "0.0.0.0", "--port", "8080"]
    ports:
      - "8080:8080"
    volumes:
      - tradingagents_data:/home/appuser/.tradingagents
      - huggingface_cache:/home/appuser/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    profiles:
      - ollama

  tradingagents-ollama:
    build: .
    env_file:
      - .env
    environment:
      - LLM_PROVIDER=ollama
    command: ["analyze"]
    volumes:
      - tradingagents_data:/home/appuser/.tradingagents
    depends_on:
      - ollama
    tty: true
    stdin_open: true
    profiles:
      - ollama

volumes:
  tradingagents_data:
  ollama_data:
  huggingface_cache:
```

- [ ] **Step 3: Validate the compose file**

Run: `docker compose config --quiet && echo "valid"`
Expected: prints `valid` with no YAML/schema errors.

- [ ] **Step 4: Build the image**

Run: `docker compose build tradingagents-api`
Expected: build succeeds (this will take a while the first time — torch is a large wheel).

- [ ] **Step 5: Smoke-test the API container with GPU passthrough**

Run: `docker compose up -d tradingagents-api && sleep 5 && curl -sf http://localhost:8080/health && docker compose logs tradingagents-api --tail 20`
Expected: `{"status":"ok"}`; no GPU-related errors in logs.

Run: `docker compose exec tradingagents-api python -c "import torch; print(torch.cuda.is_available())"`
Expected: `True` (confirms GPU passthrough works inside the container).

Then: `docker compose down`

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml Dockerfile
git commit -m "Add GPU passthrough, HF cache volume, and API service to Docker Compose"
```

---

### Task 10: Fix README CLI usage docs

**Files:**
- Modify: `README.md`

**Interfaces:** None (docs only).

- [ ] **Step 1: Update the bare-invocation lines**

Change (around line 173-174):
```
tradingagents          # installed command
python -m cli.main     # alternative: run directly from source
```
to:
```
tradingagents analyze          # installed command
python -m cli.main analyze     # alternative: run directly from source
```

- [ ] **Step 2: Add an HTTP API usage section**

Insert a new subsection after the existing "### CLI Usage" section:

```markdown
### HTTP API

Run the API server instead of the interactive CLI:
```bash
tradingagents api --host 0.0.0.0 --port 8080
# or via Docker:
docker compose up tradingagents-api
```

Submit an analysis and poll for the result:
```bash
curl -X POST http://localhost:8080/analyze -H "Content-Type: application/json" \
  -d '{"ticker": "NVDA", "depth": "deep"}'
# => {"job_id": "..."}

curl http://localhost:8080/analyze/<job_id>
# => {"status": "running", "reports_completed": 2, "reports_total": 5, ...}
```
```

- [ ] **Step 3: Verify no other bare `tradingagents` invocation remains**

Run: `grep -n '^tradingagents$\|[^a-z-]tradingagents *$' README.md`
Expected: no matches (or only matches inside the Docker `run --rm tradingagents` line, which is unaffected since the compose service now has an explicit `command:`).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Update README for the analyze subcommand and new HTTP API"
```

---

### Task 11: Update the Claude Code `/analyze` skill

**Files:**
- Modify: `.claude/skills/analyze/SKILL.md`

**Interfaces:** Consumes the HTTP API from Task 8 (`POST /analyze`, `GET /analyze/{job_id}`).

- [ ] **Step 1: Replace the tmux-driving steps with HTTP calls**

Replace the "2. Kill any stale tmux session..." through "5. Monitor progress" sections with:

```markdown
**2. Ensure the API is running**

```bash
curl -sf http://localhost:8080/health || (cd /home/sags/Projects/TradingAgents && docker compose up -d tradingagents-api)
```

Wait until `curl -sf http://localhost:8080/health` returns `{"status":"ok"}` before proceeding.

**3. Submit the analysis**

```bash
curl -s -X POST http://localhost:8080/analyze -H "Content-Type: application/json" \
  -d '{"ticker": "{TICKER}", "depth": "{DEPTH}"}'
```

Extract `job_id` from the JSON response.

**4. Poll for progress**

Every 10 seconds:
```bash
curl -s http://localhost:8080/analyze/{JOB_ID}
```

Report brief status updates to the user using `reports_completed`/`reports_total` from the response.

Stop polling when `status` is `"done"` (success) or `"error"` (failure — report the `error` field to the user).
```

- [ ] **Step 2: Update the report-count expectation**

Find any mention of `Reports: N/7` in the skill file and change to `Reports: N/8` (5th analyst — Quant Forecast — added).

- [ ] **Step 3: Manually verify the updated skill**

Run `/analyze NVDA shallow` in a Claude Code session once Task 9's `tradingagents-api` container is running, and confirm it submits the job, polls, and reports completion without touching tmux.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/analyze/SKILL.md
git commit -m "Update /analyze skill to use the HTTP API instead of tmux TUI automation"
```

---

### Task 12: Hermes skill for conversational queries

**Files:**
- Create: `/home/sags/.hermes/skills/trading/DESCRIPTION.md`
- Create: `/home/sags/.hermes/skills/trading/tradingagents/SKILL.md`

**Interfaces:** Consumes the same HTTP API as Task 11 (`POST /analyze`, `GET /analyze/{job_id}`).

- [ ] **Step 1: Write the category description**

```markdown
# Trading

Skills for querying the user's local TradingAgents multi-agent trading
analysis pipeline (stock/crypto research + a Kronos-based quant forecast),
running at http://localhost:8080.
```

- [ ] **Step 2: Write the skill file**

```markdown
---
name: tradingagents
description: "Run or check a TradingAgents multi-agent stock/crypto analysis (fundamentals, sentiment, news, technicals, Kronos quant forecast) and report the actual recommendation in chat."
version: 1.0.0
author: local
license: MIT
tags: [trading, stocks, crypto, finance, forecasting]
platforms: [linux]
---

# TradingAgents Query

The user has a local TradingAgents HTTP API running at
`http://localhost:8080` (via Docker Compose in
`/home/sags/Projects/TradingAgents`). Use it when the user asks for a stock
or crypto read, e.g. "what's your take on NVDA", "run an analysis on AVGO",
"what did the trading pipeline say about BTC".

## Steps

1. Check the API is up: `curl -sf http://localhost:8080/health`. If it
   fails, tell the user the service isn't running rather than guessing.
2. Submit the request:
   ```bash
   curl -s -X POST http://localhost:8080/analyze -H "Content-Type: application/json" \
     -d '{"ticker": "<TICKER>", "depth": "shallow"}'
   ```
   Extract `job_id`.
3. Poll `curl -s http://localhost:8080/analyze/<job_id>` every 10-15 seconds
   until `status` is `"done"` or `"error"`. This can take several minutes —
   let the user know it's running rather than going silent.
4. **On completion, quote the actual content in your reply** — the
   `final_decision` field, and the key points from `reports.quant_report`
   (the Kronos forecast) and `reports.market_report` (technical read).
   **Do not** just tell the user "the report is saved at ..." or reference a
   file path. The whole point of this skill is that the recommendation and
   its reasoning show up directly in the conversation, wherever the user is
   chatting from.
5. If `status` is `"error"`, report the `error` field directly — don't
   retry silently or fabricate a result.
```

- [ ] **Step 3: Manually verify**

Ask Hermes (through whichever channel is convenient — dashboard, Telegram, etc.) something like "what's your read on NVDA", and confirm it calls the API, polls, and quotes the recommendation content directly rather than pointing at a file.

- [ ] **Step 4: No git commit** — `~/.hermes/skills/` is Hermes's own config directory, not part of the TradingAgents git repo. No commit step for this task.

---

## Post-implementation check

- [ ] Run the full TradingAgents test suite once more: `source .venv/bin/activate && python -m pytest -q` — confirm the same 7 pre-existing failures (or fewer, never more) and all new tests passing.
- [ ] Run `ruff check tradingagents/ cli/ tests/` on the changed files and fix any new lint findings.

