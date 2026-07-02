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
