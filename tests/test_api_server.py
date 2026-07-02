import time
from unittest.mock import patch

from fastapi.testclient import TestClient


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


def _fake_final_state(ticker, date, asset_type="stock"):
    return {
        "market_report": "market ok",
        "quant_report": "quant ok",
        "final_trade_decision": "BUY",
    }, "BUY"


def _wait_for_done(client, job_id, timeout=2.0):
    """Poll until the background job thread finishes (status done/error)."""
    deadline = time.time() + timeout
    body = None
    while time.time() < deadline:
        body = client.get(f"/analyze/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.01)
    raise AssertionError(f"job did not complete in time, last state: {body}")


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


# --- depth -> round-count mapping (Finding 1) --------------------------------


@patch("tradingagents.api.server.TradingAgentsGraph")
def test_depth_shallow_maps_to_one_debate_round(mock_graph_cls):
    mock_graph_cls.return_value.propagate.side_effect = (
        lambda ticker, date, asset_type="stock": _fake_final_state(ticker, date, asset_type)
    )

    from tradingagents.api.server import app

    client = TestClient(app)
    job_id = client.post("/analyze", json={"ticker": "NVDA", "depth": "shallow"}).json()["job_id"]
    _wait_for_done(client, job_id)

    _, kwargs = mock_graph_cls.call_args
    config = kwargs["config"]
    assert config["max_debate_rounds"] == 1
    assert config["max_risk_discuss_rounds"] == 1


@patch("tradingagents.api.server.TradingAgentsGraph")
def test_depth_deep_maps_to_five_debate_rounds(mock_graph_cls):
    mock_graph_cls.return_value.propagate.side_effect = (
        lambda ticker, date, asset_type="stock": _fake_final_state(ticker, date, asset_type)
    )

    from tradingagents.api.server import app

    client = TestClient(app)
    job_id = client.post("/analyze", json={"ticker": "NVDA", "depth": "deep"}).json()["job_id"]
    _wait_for_done(client, job_id)

    _, kwargs = mock_graph_cls.call_args
    config = kwargs["config"]
    assert config["max_debate_rounds"] == 5
    assert config["max_risk_discuss_rounds"] == 5


@patch("tradingagents.api.server.TradingAgentsGraph")
def test_depth_omitted_defaults_to_deep(mock_graph_cls):
    mock_graph_cls.return_value.propagate.side_effect = (
        lambda ticker, date, asset_type="stock": _fake_final_state(ticker, date, asset_type)
    )

    from tradingagents.api.server import app

    client = TestClient(app)
    job_id = client.post("/analyze", json={"ticker": "NVDA"}).json()["job_id"]
    _wait_for_done(client, job_id)

    _, kwargs = mock_graph_cls.call_args
    config = kwargs["config"]
    assert config["max_debate_rounds"] == 5
    assert config["max_risk_discuss_rounds"] == 5


@patch("tradingagents.api.server.TradingAgentsGraph")
def test_depth_unrecognized_falls_back_to_deep(mock_graph_cls):
    mock_graph_cls.return_value.propagate.side_effect = (
        lambda ticker, date, asset_type="stock": _fake_final_state(ticker, date, asset_type)
    )

    from tradingagents.api.server import app

    client = TestClient(app)
    job_id = client.post(
        "/analyze", json={"ticker": "NVDA", "depth": "not-a-real-depth"}
    ).json()["job_id"]
    body = _wait_for_done(client, job_id)

    assert body["status"] == "done"
    _, kwargs = mock_graph_cls.call_args
    config = kwargs["config"]
    assert config["max_debate_rounds"] == 5
    assert config["max_risk_discuss_rounds"] == 5


# --- crypto analyst filtering (Finding 4) ------------------------------------


@patch("tradingagents.api.server.TradingAgentsGraph")
def test_crypto_ticker_excludes_fundamentals_analyst(mock_graph_cls):
    mock_graph_cls.return_value.propagate.side_effect = (
        lambda ticker, date, asset_type="stock": _fake_final_state(ticker, date, asset_type)
    )

    from tradingagents.api.server import app

    client = TestClient(app)
    job_id = client.post("/analyze", json={"ticker": "BTC-USD"}).json()["job_id"]
    body = _wait_for_done(client, job_id)

    assert body["status"] == "done"
    _, kwargs = mock_graph_cls.call_args
    assert "fundamentals" not in kwargs["selected_analysts"]
    # reports_total (bonus fix) should reflect the filtered analyst count, not
    # the hardcoded 5.
    assert body["reports_total"] == len(kwargs["selected_analysts"])


@patch("tradingagents.api.server.TradingAgentsGraph")
def test_stock_ticker_includes_fundamentals_analyst(mock_graph_cls):
    mock_graph_cls.return_value.propagate.side_effect = (
        lambda ticker, date, asset_type="stock": _fake_final_state(ticker, date, asset_type)
    )

    from tradingagents.api.server import app

    client = TestClient(app)
    job_id = client.post("/analyze", json={"ticker": "NVDA"}).json()["job_id"]
    body = _wait_for_done(client, job_id)

    assert body["status"] == "done"
    _, kwargs = mock_graph_cls.call_args
    assert "fundamentals" in kwargs["selected_analysts"]
    assert body["reports_total"] == 5
