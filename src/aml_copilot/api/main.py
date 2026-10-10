from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import observability
from ..agent.graph import build_graph
from ..agent.mcp_client import ToolBox
from ..agent.runner import run_investigation
from ..transactions import db, scorer

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app):
    async with ToolBox() as toolbox:
        app.state.graph = build_graph(toolbox)
        yield


app = FastAPI(title="AML Investigator Copilot", lifespan=lifespan)


class InvestigateRequest(BaseModel):
    account_id: str
    question: str = "Why was this account flagged, and what do the rules say?"


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/accounts/sample")
def sample_accounts(n: int = 8):
    # mix of flagged and ordinary accounts so the demo has something to show
    ids = db.sample_account_ids(n, suspicious_only=True) + db.sample_account_ids(max(n // 2, 1))
    out = []
    for account_id in dict.fromkeys(ids):
        s = scorer.score_account(account_id)
        out.append({"account_id": account_id, "risk_score": s["risk_score"], "flagged": s["flagged"]})
    return out


@app.post("/investigate")
async def investigate(req: InvestigateRequest):
    if db.get_profile(req.account_id) is None:
        raise HTTPException(404, f"unknown account {req.account_id}")
    return await run_investigation(app.state.graph, req.account_id, req.question)


@app.get("/metrics")
def metrics():
    return observability.summary()
