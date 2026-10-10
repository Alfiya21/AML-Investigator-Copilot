from fastapi.testclient import TestClient

from aml_copilot import llm
from aml_copilot.api.main import app
from aml_copilot.retrieval import search
from aml_copilot.transactions import db

from .test_agent import make_fake_chat, patch_world


def test_api_flow(trained, monkeypatch):
    account = db.sample_account_ids(1, suspicious_only=True)[0]
    fake, _ = make_fake_chat(account, verify_results=[0])
    patch_world(monkeypatch, fake)

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/").status_code == 200

        sample = client.get("/accounts/sample", params={"n": 3}).json()
        assert sample and {"account_id", "risk_score", "flagged"} <= set(sample[0])

        res = client.post("/investigate", json={"account_id": account, "question": "why?"})
        assert res.status_code == 200
        body = res.json()
        assert body["verification"]["n_unsupported"] == 0

        assert client.post("/investigate", json={"account_id": "A99999"}).status_code == 404

        m = client.get("/metrics").json()
        assert m["runs"] >= 1
