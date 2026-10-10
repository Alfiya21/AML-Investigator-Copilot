def rrf(rankings, k=60):
    """Reciprocal rank fusion. Each ranking is a list of ids, best first."""
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)
