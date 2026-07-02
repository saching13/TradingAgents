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
