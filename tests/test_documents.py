import hashlib

import fitz
import numpy as np

from aml_copilot.ingestion import chunking, pdf_parser
from aml_copilot.retrieval import index as index_mod
from aml_copilot.retrieval.index import HybridIndex


def make_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Structuring means breaking up cash deposits to avoid reporting duties. " * 3)
    # a ruled 3x3 table
    x0, y0, w, h = 72, 200, 100, 24
    for r in range(4):
        page.draw_line((x0, y0 + r * h), (x0 + 3 * w, y0 + r * h))
    for c in range(4):
        page.draw_line((x0 + c * w, y0), (x0 + c * w, y0 + 3 * h))
    for r in range(3):
        for c in range(3):
            page.insert_text((x0 + c * w + 4, y0 + r * h + 16), f"r{r}c{c}")
    # enough vector drawing to look like a chart
    for i in range(30):
        page.draw_rect(fitz.Rect(72 + i * 10, 400, 76 + i * 10, 400 + 5 * (i % 7 + 1)))
    doc.save(path)
    doc.close()


def test_parser_finds_text_table_and_visual_page(tmp_path):
    pdf = tmp_path / "guide.pdf"
    make_pdf(str(pdf))
    pages = pdf_parser.parse_pdf(pdf, tmp_path / "imgs")

    assert len(pages) == 1
    assert "Structuring" in pages[0].text
    assert len(pages[0].tables) == 1
    assert pages[0].image_path is not None and pages[0].image_path.exists()


def test_build_chunks_covers_all_kinds(tmp_path):
    pdf = tmp_path / "guide.pdf"
    make_pdf(str(pdf))
    pages = pdf_parser.parse_pdf(pdf, tmp_path / "imgs")
    captions = {pages[0].image_path.name: "A bar chart of deposit sizes."}

    kinds = {c.kind for c in chunking.build_chunks(pages, captions)}
    assert kinds == {"text", "table", "figure"}


def test_none_caption_is_dropped(tmp_path):
    pdf = tmp_path / "guide.pdf"
    make_pdf(str(pdf))
    pages = pdf_parser.parse_pdf(pdf, tmp_path / "imgs")
    chunks = chunking.build_chunks(pages, {pages[0].image_path.name: "NONE"})
    assert "figure" not in {c.kind for c in chunks}


def fake_embed(texts, batch_size=32):
    # deterministic bag-of-words hashing, good enough to test plumbing without a model
    out = np.zeros((len(texts), 64), dtype="float32")
    for i, t in enumerate(texts):
        for word in t.lower().split():
            out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(norms, 1e-9, None)


def test_index_roundtrip_and_search(tmp_path, monkeypatch):
    monkeypatch.setattr(index_mod, "embed", fake_embed)
    chunks = [
        chunking.Chunk("a", "d", 1, "text", "structuring cash deposits below the reporting threshold"),
        chunking.Chunk("b", "d", 2, "text", "customer due diligence for politically exposed persons"),
        chunking.Chunk("c", "d", 3, "text", "wire transfer travel rule information requirements"),
    ]
    idx = HybridIndex.build(chunks)
    idx.save(tmp_path)

    loaded = HybridIndex.load(tmp_path)
    top = loaded.search("structuring deposits", k=3)
    assert top[0].id == "a"
    assert len(loaded.chunks) == 3
