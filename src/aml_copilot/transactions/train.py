import json

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .. import config
from . import db
from .features import FEATURES, build_features


def model_paths(version=None):
    version = version or config.MODEL_VERSION
    return config.MODEL_DIR / f"xgb_{version}.json", config.MODEL_DIR / f"meta_{version}.json"


def train(seed=42):
    accounts = db.all_accounts().set_index("account_id")
    feats = build_features(db.all_transactions())
    data = accounts[["is_suspicious"]].join(feats, how="inner")
    X, y = data[FEATURES], data["is_suspicious"]

    # 60 / 20 / 20: fit, pick the threshold, then report on data the model never saw
    X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed)

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        eval_metric="aucpr", random_state=seed,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    prec, rec, thr = precision_recall_curve(y_val, model.predict_proba(X_val)[:, 1])
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    threshold = float(thr[np.argmax(f1[:-1])])

    p_te = model.predict_proba(X_te)[:, 1]
    pred = (p_te >= threshold).astype(int)
    tp = int(((pred == 1) & (y_te == 1)).sum())
    fn = int(((pred == 0) & (y_te == 1)).sum())
    fp = int(((pred == 1) & (y_te == 0)).sum())
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
        "pr_auc": round(float(average_precision_score(y_te, p_te)), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "false_negatives": fn,
        "false_positives": fp,
        "n_test": int(len(y_te)),
        "n_test_positive": int(y_te.sum()),
    }

    model_path, meta_path = model_paths()
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    meta_path.write_text(json.dumps(
        {"features": FEATURES, "threshold": threshold, "metrics": metrics,
         "test_ids": [str(i) for i in X_te.index]}, indent=2))
    return metrics
