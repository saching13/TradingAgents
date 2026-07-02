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
