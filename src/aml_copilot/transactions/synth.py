import numpy as np
import pandas as pd

# share of accounts, avg txns per account, median amount, share of cash txns
SEGMENTS = {
    "retail": (0.70, 30, 120, 0.10),
    "small_business": (0.25, 70, 600, 0.30),
    "corporate": (0.05, 140, 4000, 0.03),
}
REGIONS = ["north", "south", "east", "west"]
CHANNELS = ["cash", "wire", "card", "ach"]
START = pd.Timestamp("2025-01-01")
DAYS = 180


def make_accounts(n, rng):
    names = list(SEGMENTS)
    segment = rng.choice(names, size=n, p=[SEGMENTS[s][0] for s in names])
    return pd.DataFrame({
        "account_id": [f"A{i:05d}" for i in range(n)],
        "segment": segment,
        "region": rng.choice(REGIONS, size=n),
        "true_pattern": "none",
    })


def _rows(account_id, ts, amount, direction, channel, counterparty, risk):
    return pd.DataFrame({
        "ts": ts, "account_id": account_id, "counterparty_id": counterparty,
        "direction": direction, "channel": channel, "amount": np.round(amount, 2),
        "jurisdiction_risk": risk,
    })


def _random_times(rng, n, first_day, last_day):
    days = rng.integers(first_day, last_day, n)
    secs = rng.integers(0, 86400, n)
    return START + pd.to_timedelta(days, unit="D") + pd.to_timedelta(secs, unit="s")


def normal_activity(account_id, segment, rng):
    _, rate, median, cash_share = SEGMENTS[segment]
    n = max(3, rng.poisson(rate))
    amount = np.clip(rng.lognormal(np.log(median), 0.9, n), 1, 60000)
    channel = rng.choice(CHANNELS, size=n, p=[cash_share, 0.15, 0.60 - cash_share, 0.25])
    counterparty = [f"C{i:05d}" for i in rng.integers(0, 6000, n)]
    risk = rng.choice(["low", "medium", "high"], size=n, p=[0.93, 0.06, 0.01])
    return _rows(account_id, _random_times(rng, n, 0, DAYS), amount,
                 rng.choice(["in", "out"], size=n), channel, counterparty, risk)


def structuring(account_id, rng, n=None):
    # cash deposits just under the 10k reporting line, over a couple of weeks
    n = n or rng.integers(4, 31)
    first = rng.integers(0, DAYS - 25)
    counterparty = [f"C9{i:04d}" for i in rng.integers(0, 4, n)]
    return _rows(account_id, _random_times(rng, n, first, first + 20),
                 rng.uniform(8000, 9900, n), "in", "cash", counterparty,
                 rng.choice(["low", "medium"], size=n))


def pass_through(account_id, rng, pairs=None):
    # money comes in and leaves again within hours, nearly the same amount
    out = []
    for _ in range(pairs or rng.integers(3, 41)):
        t = _random_times(rng, 1, 0, DAYS)[0]
        amount = rng.uniform(2000, 30000)
        out.append(_rows(account_id, [t], [amount], "in", "wire",
                         [f"C{rng.integers(0, 6000):05d}"], ["medium"]))
        leave = t + pd.Timedelta(hours=float(rng.uniform(1, 18)))
        risk = "high" if rng.random() < 0.6 else "medium"
        out.append(_rows(account_id, [leave], [amount * rng.uniform(0.9, 0.99)], "out", "wire",
                         [f"C{rng.integers(0, 6000):05d}"], [risk]))
    return pd.concat(out, ignore_index=True)


def fan_in(account_id, rng, n_in=None):
    # many small wires from different senders in a week, then a few big ones out
    n_in = n_in or rng.integers(8, 51)
    first = rng.integers(0, DAYS - 20)
    amounts = rng.uniform(500, 3000, n_in)
    incoming = _rows(account_id, _random_times(rng, n_in, first, first + 7), amounts, "in", "wire",
                     [f"C8{i:04d}" for i in range(n_in)], "medium")
    n_out = int(rng.integers(1, 4))
    outgoing = _rows(account_id, _random_times(rng, n_out, first + 7, first + 12),
                     np.full(n_out, amounts.sum() * 0.9 / n_out), "out", "wire",
                     [f"C7{i:04d}" for i in range(n_out)], "high")
    return pd.concat([incoming, outgoing], ignore_index=True)


PATTERNS = {"structuring": structuring, "pass_through": pass_through, "fan_in": fan_in}


def lookalike(account_id, rng):
    # legitimate accounts that resemble a pattern: a shop banking its cash, a payment
    # intermediary, a fundraiser. Without these the task is unrealistically easy.
    kind = rng.integers(0, 3)
    if kind == 0:
        return structuring(account_id, rng, n=int(rng.integers(2, 7)))
    if kind == 1:
        return pass_through(account_id, rng, pairs=int(rng.integers(2, 6)))
    first = int(rng.integers(0, DAYS - 20))
    n = int(rng.integers(8, 21))
    return _rows(account_id, _random_times(rng, n, first, first + 10), rng.uniform(20, 200, n),
                 "in", "ach", [f"C6{i:04d}" for i in range(n)], "low")


def generate(n_accounts=4000, suspicious_rate=0.04, lookalike_rate=0.10, seed=7):
    rng = np.random.default_rng(seed)
    accounts = make_accounts(n_accounts, rng)

    names = list(PATTERNS)
    bad = rng.choice(n_accounts, size=int(n_accounts * suspicious_rate), replace=False)
    for i in bad:
        accounts.loc[i, "true_pattern"] = names[rng.integers(0, len(names))]

    frames = []
    for acct in accounts.itertuples():
        frames.append(normal_activity(acct.account_id, acct.segment, rng))
        if acct.true_pattern != "none":
            frames.append(PATTERNS[acct.true_pattern](acct.account_id, rng))
        elif rng.random() < lookalike_rate:
            frames.append(lookalike(acct.account_id, rng))

    txns = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
    txns.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(txns))])
    txns["ts"] = txns["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")

    accounts["is_suspicious"] = (accounts["true_pattern"] != "none").astype(int)
    return accounts, txns
