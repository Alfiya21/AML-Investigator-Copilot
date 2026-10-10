import re
from dataclasses import dataclass
from pathlib import Path

from .captioning import table_to_markdown

MIN_CHARS = 40


@dataclass
class Chunk:
    id: str
    doc: str
    page: int
    kind: str  # text, table or figure
    text: str


def clean_text(text):
    text = re.sub(r"-\n(\w)", r"\1", text)            # re-join hyphenated line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # single newlines are just wrapping
    return re.sub(r"[ \t]+", " ", text)


def split_text(text, max_chars=900, overlap=150):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean_text(text)) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= max_chars:
            current = f"{current} {para}".strip()
            continue
        if current:
            chunks.append(current)
        # carry the end of the last chunk forward so a sentence isn't cut off from its context
        tail = current[-overlap:] if current else ""
        current = f"{tail} {para}".strip()
        while len(current) > max_chars * 1.5:  # one huge paragraph
            chunks.append(current[:max_chars])
            current = current[max_chars - overlap:]
    if current:
        chunks.append(current)
    return chunks


def build_chunks(pages, captions):
    chunks = []
    for page in pages:
        base = f"{page.doc}-p{page.number}"
        for i, piece in enumerate(split_text(page.text)):
            if len(piece) >= MIN_CHARS:
                chunks.append(Chunk(f"{base}-t{i}", page.doc, page.number, "text", piece))
        for j, table in enumerate(page.tables):
            chunks.append(Chunk(f"{base}-tb{j}", page.doc, page.number, "table", table_to_markdown(table)))
        if page.image_path is not None:
            caption = captions.get(Path(page.image_path).name, "")
            if caption and caption.upper() != "NONE":
                chunks.append(Chunk(f"{base}-fig", page.doc, page.number, "figure", caption))
    return chunks
