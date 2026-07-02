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
