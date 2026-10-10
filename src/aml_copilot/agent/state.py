from typing import TypedDict


class InvState(TypedDict, total=False):
    account_id: str
    question: str
    evidence: list      # tool calls and what they returned
    queries: list       # search queries used against the documents
    passages: list      # retrieved chunks, as dicts
    draft: str
    verification: dict
    feedback: str       # unsupported claims, fed back into the next draft
    attempts: int
