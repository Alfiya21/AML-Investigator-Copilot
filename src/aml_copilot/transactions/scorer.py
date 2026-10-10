import json

from xgboost import XGBClassifier

from . import db
from .features import FEATURES, build_features
from .train import model_paths

_model = None
_meta = None


def load_model():
    global _model, _meta
    if _model is None:
        model_path, meta_path = model_paths()
        _model = XGBClassifier()
        _model.load_model(model_path)
        _meta = json.loads(meta_path.read_text())
    return _model, _meta


def feature_row(account_id):
    txns = db.all_transactions(account_id)
    if txns.empty:
        return None
    return build_features(txns)[FEATURES]


def score_account(account_id):
    row = feature_row(account_id)
    if row is None:
        return None
    model, meta = load_model()
    prob = float(model.predict_proba(row)[:, 1][0])
    return {
        "account_id": account_id,
        "risk_score": round(prob, 4),
        "threshold": round(meta["threshold"], 4),
        "flagged": prob >= meta["threshold"],
    }
