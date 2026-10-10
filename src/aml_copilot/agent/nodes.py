import asyncio
import json
from dataclasses import asdict

from .. import llm, prompts
from ..retrieval import search

MAX_TOOL_ROUNDS = 5
MAX_DRAFTS = 2


async def _chat(*args, **kwargs):
    # the OpenAI client is synchronous, so keep it off the event loop
    return await asyncio.to_thread(llm.chat, *args, **kwargs)


def format_evidence(evidence):
    lines = []
    for i, item in enumerate(evidence, 1):
        args = ", ".join(f"{k}={v}" for k, v in item["args"].items())
        body = json.dumps(item["result"], default=str)[:2500]
        lines.append(f"[T{i}] {item['tool']}({args}): {body}")
    return "\n".join(lines)


def format_passages(passages):
    return "\n".join(
        f"[D{i}] ({p['doc']}, page {p['page']}, {p['kind']}) {p['text']}"
        for i, p in enumerate(passages, 1)
    )


def _json(text, default):
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return default


async def investigate(state, toolbox):
    messages = [
        {"role": "system", "content": prompts.load("agent_system")},
        {"role": "user", "content": f"Account: {state['account_id']}\nQuestion: {state['question']}"},
    ]
    evidence = []

    for _ in range(MAX_TOOL_ROUNDS):
        msg = await _chat(messages, step="investigate", tools=toolbox.openai_tools)
        if not msg.tool_calls:
            break

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in msg.tool_calls
            ],
        })
        for call in msg.tool_calls:
            args = _json(call.function.arguments, {})
            args["account_id"] = state["account_id"]  # the model doesn't get to wander to other accounts
            result = await toolbox.call(call.function.name, args)
            evidence.append({"tool": call.function.name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, default=str)[:6000]})

    return {"evidence": evidence}


async def retrieve(state):
    msg = await _chat(
        [{"role": "system", "content": prompts.load("queries")},
         {"role": "user", "content": f"Question: {state['question']}\n\nEvidence:\n{format_evidence(state['evidence'])}"}],
        step="queries", json_mode=True,
    )
    queries = _json(msg.content, {}).get("queries") or [state["question"]]
    queries = queries[:3]

    seen, passages = set(), []
    for q in queries:
        for chunk in await asyncio.to_thread(search.retrieve, q, 20, 4):
            if chunk.id not in seen:
                seen.add(chunk.id)
                passages.append(asdict(chunk))
    return {"queries": queries, "passages": passages[:8]}


async def draft(state):
    context = f"{format_evidence(state['evidence'])}\n\n{format_passages(state['passages'])}"
    user = f"Analyst question: {state['question']}\nAccount: {state['account_id']}\n\nContext:\n{context}"
    if state.get("feedback"):
        user += ("\n\nA reviewer found these claims were not supported by the context. "
                 f"Fix or remove them:\n{state['feedback']}")
    msg = await _chat(
        [{"role": "system", "content": prompts.load("draft")}, {"role": "user", "content": user}],
        step="draft",
    )
    return {"draft": msg.content, "attempts": state.get("attempts", 0) + 1}


async def verify(state):
    context = f"{format_evidence(state['evidence'])}\n\n{format_passages(state['passages'])}"
    msg = await _chat(
        [{"role": "system", "content": prompts.load("verify")},
         {"role": "user", "content": f"Context:\n{context}\n\nDraft:\n{state['draft']}"}],
        step="verify", json_mode=True,
    )
    claims = _json(msg.content, {}).get("claims", [])
    unsupported = [c.get("text", "") for c in claims if not c.get("supported")]
    return {
        "verification": {"n_claims": len(claims), "n_unsupported": len(unsupported),
                         "unsupported": unsupported},
        "feedback": "\n".join(f"- {t}" for t in unsupported),
    }


def after_verify(state):
    if state["verification"]["n_unsupported"] and state["attempts"] < MAX_DRAFTS:
        return "draft"
    return "end"
