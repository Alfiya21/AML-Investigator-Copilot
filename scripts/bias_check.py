import pandas as pd

from aml_copilot import config
from aml_copilot.transactions import db, scorer
from aml_copilot.transactions.features import FEATURES, build_features


def main():
    model, meta = scorer.load_model()
    test_ids = meta["test_ids"]  # only accounts the model never trained on

    accounts = db.all_accounts().set_index("account_id").loc[test_ids]
    txns = db.all_transactions()
    feats = build_features(txns[txns["account_id"].isin(test_ids)]).loc[test_ids, FEATURES]
    accounts["flagged"] = (model.predict_proba(feats)[:, 1] >= meta["threshold"]).astype(int)

    rows = []
    for column in ("segment", "region"):
        for group, g in accounts.groupby(column):
            bad, ok = g[g.is_suspicious == 1], g[g.is_suspicious == 0]
            rows.append({
                "attribute": column, "group": group, "accounts": len(g), "suspicious": len(bad),
                "flag_rate": round(g.flagged.mean(), 4),
                "recall": round(bad.flagged.mean(), 4) if len(bad) else None,
                "false_positive_rate": round(ok.flagged.mean(), 4) if len(ok) else None,
            })

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))
    out = config.ROOT / "eval" / "bias_report.csv"
    report.to_csv(out, index=False)
    print(f"\nsaved {out}")
    print("Small groups give noisy rates. Read the account counts before drawing conclusions.")


if __name__ == "__main__":
    main()
