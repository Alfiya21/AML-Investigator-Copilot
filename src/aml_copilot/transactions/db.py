import sqlite3
from contextlib import closing

import pandas as pd

from .. import config


def connect():
    return sqlite3.connect(config.DB_PATH)


def load_frames(accounts, txns):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with closing(connect()) as conn:
        accounts.to_sql("accounts", conn, if_exists="replace", index=False)
        txns.to_sql("transactions", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_txn_acct ON transactions(account_id, ts)")
        conn.commit()


def get_profile(account_id):
    # true_pattern and is_suspicious are ground truth for evaluation, so they stay out of here
    with closing(connect()) as conn:
        row = conn.execute(
            "SELECT account_id, segment, region FROM accounts WHERE account_id = ?", (account_id,)
        ).fetchone()
    if row is None:
        return None
    return {"account_id": row[0], "segment": row[1], "region": row[2]}


def get_transactions(account_id, limit=30):
    limit = max(1, min(int(limit), 200))
    with closing(connect()) as conn:
        cur = conn.execute(
            "SELECT txn_id, ts, counterparty_id, direction, channel, amount, jurisdiction_risk"
            " FROM transactions WHERE account_id = ? ORDER BY ts DESC LIMIT ?",
            (account_id, limit),
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def all_transactions(account_id=None):
    query = "SELECT * FROM transactions"
    params = ()
    if account_id is not None:
        query += " WHERE account_id = ?"
        params = (account_id,)
    with closing(connect()) as conn:
        return pd.read_sql_query(query, conn, params=params)


def all_accounts():
    with closing(connect()) as conn:
        return pd.read_sql_query("SELECT * FROM accounts", conn)


def sample_account_ids(n=8, suspicious_only=False):
    # demo helper for the UI and tests, not something the agent can call
    query = "SELECT account_id FROM accounts"
    if suspicious_only:
        query += " WHERE is_suspicious = 1"
    query += " ORDER BY RANDOM() LIMIT ?"
    with closing(connect()) as conn:
        return [r[0] for r in conn.execute(query, (n,)).fetchall()]
