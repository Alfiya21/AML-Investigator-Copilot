import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .. import config
from ..ingestion.chunking import Chunk
from .fusion import rrf

_embedder = None


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def embed(texts, batch_size=32):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    vecs = _embedder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=False)
    return np.asarray(vecs, dtype="float32")


class HybridIndex:
    def __init__(self, chunks, faiss_index):
        self.chunks = chunks
        self.faiss = faiss_index
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    @classmethod
    def build(cls, chunks):
        import faiss
        vectors = embed([c.text for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])  # vectors are normalised, so inner product = cosine
        index.add(vectors)
        return cls(chunks, index)

    def search(self, query, k=20):
        _, ids = self.faiss.search(embed([query]), k)
        dense = [int(i) for i in ids[0] if i != -1]

        scores = self.bm25.get_scores(tokenize(query))
        sparse = [int(i) for i in np.argsort(scores)[::-1][:k] if scores[i] > 0]

        return [self.chunks[i] for i in rrf([dense, sparse])[:k]]

    def save(self, root):
        import faiss
        digest = hashlib.sha256("".join(c.text for c in self.chunks).encode()).hexdigest()[:8]
        folder = Path(root) / f"idx_{digest}"
        folder.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.faiss, str(folder / "faiss.index"))
        with open(folder / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(asdict(c)) + "\n")
        (folder / "manifest.json").write_text(json.dumps({
            "embed_model": config.EMBED_MODEL,
            "n_chunks": len(self.chunks),
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))
        (Path(root) / "LATEST").write_text(folder.name)
        return folder

    @classmethod
    def load(cls, root):
        import faiss
        root = Path(root)
        folder = root / (root / "LATEST").read_text().strip()
        with open(folder / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        return cls(chunks, faiss.read_index(str(folder / "faiss.index")))
