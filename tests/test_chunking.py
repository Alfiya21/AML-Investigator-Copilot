from aml_copilot.ingestion.chunking import clean_text, split_text


def test_clean_text_joins_wrapped_lines():
    assert clean_text("the quick\nbrown fox") == "the quick brown fox"
    assert clean_text("trans-\naction") == "transaction"


def test_split_respects_max_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(12))
    chunks = split_text(text, max_chars=500, overlap=80)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 500 * 1.5 for c in chunks)


def test_single_giant_paragraph_still_terminates():
    chunks = split_text("x" * 5000, max_chars=900, overlap=100)
    assert len(chunks) >= 5


def test_overlap_carries_context():
    text = "\n\n".join(["alpha " * 100, "beta " * 100])
    first, second = split_text(text, max_chars=700, overlap=60)[:2]
    assert first[-30:].strip() in second
