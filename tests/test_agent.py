import asyncio
import json
from types import SimpleNamespace

from aml_copilot import llm
from aml_copilot.agent.graph import build_graph
from aml_copilot.agent.mcp_client import ToolBox
from aml_copilot.agent.runner import run_investigation
from aml_copilot.ingestion.chunking import Chunk
from aml_copilot.retrieval import search
from aml_copilot.transactions import db


def _tool_call(name, account_id):
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps({"account_id": account_id})),
    )


def make_fake_chat(account_id, verify_results):
    """Scripted stand-in for the LLM. verify_results is a list of unsupported-claim counts, one per verify call."""
    calls = {"investigate": 0, "verify": 0, "draft": 0}

    def fake(messages, step, model=None, tools=None, json_mode=False, temperature=0.0):
        if step == "investigate":
            calls["investigate"] += 1
            if calls["investigate"] == 1:
                return SimpleNamespace(content=None, tool_calls=[
                    _tool_call("score_account", account_id), _tool_call("explain_flag", account_id)])
            return SimpleNamespace(content="DONE", tool_calls=None)
        if step == "queries":
            return SimpleNamespace(content=json.dumps({"queries": ["structuring cash deposits"]}), tool_calls=None)
        if step == "draft":
            calls["draft"] += 1
            return SimpleNamespace(content=f"draft {calls['draft']} [T1] [D1]", tool_calls=None)
        if step == "verify":
            bad = verify_results[min(calls["verify"], len(verify_results) - 1)]
            calls["verify"] += 1
            claims = [{"text": "ok claim", "supported": True, "source": "T1"}]
            claims += [{"text": f"bad claim {i}", "supported": False, "source": "D1"} for i in range(bad)]
            return SimpleNamespace(content=json.dumps({"claims": claims}), tool_calls=None)
        raise AssertionError(f"unexpected step {step}")

    return fake, calls


def run_agent(account_id):
    async def go():
        async with ToolBox() as toolbox:
            assert {"score_account", "explain_flag"} <= {t["function"]["name"] for t in toolbox.openai_tools}
            graph = build_graph(toolbox)
            return await run_investigation(graph, account_id, "Why was this flagged?")
    return asyncio.run(go())


def patch_world(monkeypatch, fake):
    monkeypatch.setattr(llm, "chat", fake)
    monkeypatch.setattr(search, "retrieve", lambda q, candidates=20, top_n=5: [
        Chunk("doc-p1-t0", "doc", 1, "text", "Structuring means splitting deposits to avoid reporting.")])


def test_agent_runs_end_to_end(trained, monkeypatch):
    account = db.sample_account_ids(1, suspicious_only=True)[0]
    fake, calls = make_fake_chat(account, verify_results=[0])
    patch_world(monkeypatch, fake)

    out = run_agent(account)

    assert [t["tool"] for t in out["tools_used"]] == ["score_account", "explain_flag"]
    assert out["attempts"] == 1
    assert out["verification"]["n_unsupported"] == 0
    assert out["sources"][0]["doc"] == "doc"
    assert calls["draft"] == 1


def test_agent_redrafts_when_claims_are_unsupported(trained, monkeypatch):
    account = db.sample_account_ids(1, suspicious_only=True)[0]
    fake, calls = make_fake_chat(account, verify_results=[2, 0])
    patch_world(monkeypatch, fake)

    out = run_agent(account)

    assert out["attempts"] == 2
    assert calls["draft"] == 2
    assert out["verification"]["n_unsupported"] == 0


def test_agent_stops_after_max_drafts(trained, monkeypatch):
    account = db.sample_account_ids(1)[0]
    fake, calls = make_fake_chat(account, verify_results=[1])
    patch_world(monkeypatch, fake)

    out = run_agent(account)

    assert out["attempts"] == 2
    assert out["verification"]["n_unsupported"] == 1  # surfaced to the analyst, not hidden
