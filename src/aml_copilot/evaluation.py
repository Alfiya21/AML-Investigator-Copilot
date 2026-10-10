import json

from . import llm, prompts
from .retrieval import search


def hit_rank(chunks, phrases):
    """1-based rank of the first chunk containing any expected phrase, or None."""
    phrases = [p.lower() for p in phrases]
    for rank, chunk in enumerate(chunks, 1):
        text = chunk.text.lower()
        if any(p in text for p in phrases):
            return rank
    return None


def retrieval_scores(questions, k=5, retrieve=None):
    retrieve = retrieve or (lambda q: search.retrieve(q, 20, k))
    ranks = [hit_rank(retrieve(q["question"]), q["expected_phrases"]) for q in questions]
    hits = [r for r in ranks if r is not None]
    return {
        "n": len(questions),
        f"hit_rate@{k}": round(len(hits) / len(questions), 4),
        "mrr": round(sum(1 / r for r in hits) / len(questions), 4),
    }


def answer_from_docs(question, k=5):
    chunks = search.retrieve(question, 20, k)
    context = "\n".join(f"[D{i}] {c.text}" for i, c in enumerate(chunks, 1))
    msg = llm.chat(
        [{"role": "system", "content": prompts.load("qa")},
         {"role": "user", "content": f"Question: {question}\n\nPassages:\n{context}"}],
        step="eval_answer",
    )
    return msg.content, context


def judge(answer, context):
    msg = llm.chat(
        [{"role": "system", "content": prompts.load("verify")},
         {"role": "user", "content": f"Context:\n{context}\n\nDraft:\n{answer}"}],
        step="eval_judge", json_mode=True,
    )
    try:
        claims = json.loads(msg.content).get("claims", [])
    except json.JSONDecodeError:
        claims = []
    unsupported = [c for c in claims if not c.get("supported")]
    return len(claims), len(unsupported)


def generation_scores(questions, k=5):
    total_claims = total_bad = answers_with_bad = 0
    for q in questions:
        answer, context = answer_from_docs(q["question"], k)
        n, bad = judge(answer, context)
        total_claims += n
        total_bad += bad
        answers_with_bad += 1 if bad else 0
    return {
        "faithfulness": round(1 - total_bad / total_claims, 4) if total_claims else None,
        "hallucination_rate": round(answers_with_bad / len(questions), 4),
        "claims_checked": total_claims,
    }
