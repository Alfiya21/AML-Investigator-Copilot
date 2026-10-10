import time
import uuid

from .. import observability


async def run_investigation(graph, account_id, question):
    rid = uuid.uuid4().hex[:12]
    observability.run_id.set(rid)
    start = time.perf_counter()
    try:
        state = await graph.ainvoke({"account_id": account_id, "question": question, "attempts": 0})
    except Exception:
        observability.log_run(rid, account_id, (time.perf_counter() - start) * 1000, None, 0, "error")
        raise

    latency_ms = (time.perf_counter() - start) * 1000
    observability.log_run(rid, account_id, latency_ms, state["verification"], state["attempts"], "ok")
    return {
        "run_id": rid,
        "account_id": account_id,
        "answer": state["draft"],
        "verification": state["verification"],
        "attempts": state["attempts"],
        "queries": state["queries"],
        "tools_used": [{"tool": e["tool"], "args": e["args"]} for e in state["evidence"]],
        "sources": [{"doc": p["doc"], "page": p["page"], "kind": p["kind"]} for p in state["passages"]],
        "latency_ms": round(latency_ms, 1),
    }
