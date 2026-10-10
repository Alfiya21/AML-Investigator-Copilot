import pandas as pd

FEATURES = [
    "n_txn", "total_in", "total_out", "avg_amount", "std_amount", "max_amount",
    "cash_ratio", "near_threshold_count", "near_threshold_ratio", "n_counterparties",
    "high_risk_ratio", "in_out_ratio", "max_txn_per_day", "pass_through_count",
]

# plain-language meaning of each feature, passed to the LLM next to the SHAP values
DESCRIPTIONS = {
    "n_txn": "number of transactions",
    "total_in": "total money received",
    "total_out": "total money sent",
    "avg_amount": "average transaction amount",
    "std_amount": "spread of transaction amounts",
    "max_amount": "largest single transaction",
    "cash_ratio": "share of transactions made in cash",
    "near_threshold_count": "transactions between 8,000 and 10,000 (just under a reporting line)",
    "near_threshold_ratio": "share of transactions just under the reporting line",
    "n_counterparties": "number of distinct counterparties",
    "high_risk_ratio": "share of transactions involving high-risk jurisdictions",
    "in_out_ratio": "money received relative to money sent",
    "max_txn_per_day": "most transactions seen in a single day",
    "pass_through_count": "incoming payments sent back out within 24h at a similar amount",
}


def _pass_through_count(g):
    last_in = None
    hits = 0
    for ts, amount, direction in zip(g["ts"], g["amount"], g["direction"]):
        if direction == "in":
            last_in = (ts, amount)
        elif last_in is not None:
            hours = (ts - last_in[0]).total_seconds() / 3600
            if hours <= 24 and 0.8 <= amount / last_in[1] <= 1.05:
                hits += 1
                last_in = None
    return hits


def account_features(g):
    g = g.sort_values("ts")
    amount = g["amount"]
    n = len(g)
    total_in = amount[g["direction"] == "in"].sum()
    total_out = amount[g["direction"] == "out"].sum()
    near = int(((amount >= 8000) & (amount < 10000)).sum())
    return {
        "n_txn": n,
        "total_in": total_in,
        "total_out": total_out,
        "avg_amount": amount.mean(),
        "std_amount": amount.std(ddof=0),
        "max_amount": amount.max(),
        "cash_ratio": (g["channel"] == "cash").mean(),
        "near_threshold_count": near,
        "near_threshold_ratio": near / n,
        "n_counterparties": g["counterparty_id"].nunique(),
        "high_risk_ratio": (g["jurisdiction_risk"] == "high").mean(),
        "in_out_ratio": total_in / (total_out + 1),
        "max_txn_per_day": g["ts"].dt.date.value_counts().max(),
        "pass_through_count": _pass_through_count(g),
    }


def build_features(txns):
    txns = txns.assign(ts=pd.to_datetime(txns["ts"]))
    rows = {acct: account_features(g) for acct, g in txns.groupby("account_id")}
    return pd.DataFrame.from_dict(rows, orient="index")[FEATURES]
