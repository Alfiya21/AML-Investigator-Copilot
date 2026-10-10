from aml_copilot.evaluation import hit_rank
from aml_copilot.ingestion.chunking import Chunk
from aml_copilot.retrieval.fusion import rrf


def test_rrf_prefers_items_ranked_well_in_both_lists():
    assert rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] == ["a", "b"]
    assert rrf([["a", "b"], ["c", "b"]])[0] == "b"


def test_rrf_keeps_items_found_by_only_one_retriever():
    assert set(rrf([["a"], ["b"]])) == {"a", "b"}


def test_hit_rank():
    chunks = [Chunk("1", "d", 1, "text", "nothing here"),
              Chunk("2", "d", 2, "text", "This describes Structuring in detail")]
    assert hit_rank(chunks, ["structuring"]) == 2
    assert hit_rank(chunks, ["layering"]) is None
