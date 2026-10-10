from mcp.server.fastmcp import FastMCP

from ..transactions import db, explain, scorer

mcp = FastMCP("aml-tools")


def _missing(account_id):
    return {"error": f"no account found with id {account_id}"}


@mcp.tool()
def get_account_profile(account_id: str) -> dict:
    """Customer segment and region for an account."""
    return db.get_profile(account_id) or _missing(account_id)


@mcp.tool()
def get_recent_transactions(account_id: str, limit: int = 30) -> dict:
    """Most recent transactions for an account, newest first (max 200)."""
    rows = db.get_transactions(account_id, limit)
    return {"account_id": account_id, "transactions": rows} if rows else _missing(account_id)


@mcp.tool()
def score_account(account_id: str) -> dict:
    """Risk score from the XGBoost model and whether it crosses the alert threshold."""
    return scorer.score_account(account_id) or _missing(account_id)


@mcp.tool()
def explain_flag(account_id: str, top_k: int = 5) -> dict:
    """Top SHAP reasons behind the model's score for this account."""
    reasons = explain.top_reasons(account_id, top_k)
    return {"account_id": account_id, "reasons": reasons} if reasons else _missing(account_id)


if __name__ == "__main__":
    mcp.run()  # stdio transport
