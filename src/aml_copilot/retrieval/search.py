from .. import config
from .index import HybridIndex
from .rerank import rerank

_index = None


def get_index():
    global _index
    if _index is None:
        _index = HybridIndex.load(config.INDEX_DIR)
    return _index


def retrieve(query, candidates=20, top_n=5):
    found = get_index().search(query, k=candidates)
    return rerank(query, found, top_n=top_n)
