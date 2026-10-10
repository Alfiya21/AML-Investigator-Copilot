import numpy as np
import shap

from .features import DESCRIPTIONS
from .scorer import feature_row, load_model

_explainer = None


def top_reasons(account_id, top_k=5):
    global _explainer
    row = feature_row(account_id)
    if row is None:
        return None
    model, _ = load_model()
    if _explainer is None:
        _explainer = shap.TreeExplainer(model)

    values = _explainer.shap_values(row)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)[0]

    order = np.argsort(np.abs(values))[::-1][:top_k]
    return [
        {
            "feature": row.columns[i],
            "meaning": DESCRIPTIONS[row.columns[i]],
            "value": round(float(row.iloc[0, i]), 3),
            "shap": round(float(values[i]), 3),
            "effect": "raises risk" if values[i] > 0 else "lowers risk",
        }
        for i in order
    ]
