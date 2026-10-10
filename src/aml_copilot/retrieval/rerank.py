import numpy as np

from .. import config

_model = None


def rerank(query, chunks, top_n=5):
    global _model
    if not chunks:
        return []
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(config.RERANK_MODEL)
    scores = _model.predict([(query, c.text) for c in chunks])
    order = np.argsort(scores)[::-1][:top_n]
    return [chunks[i] for i in order]
