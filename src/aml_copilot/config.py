import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
PROMPT_DIR = ROOT / "prompts"

PDF_DIR = DATA_DIR / "raw_pdfs"
PAGE_IMG_DIR = DATA_DIR / "page_images"
INDEX_DIR = DATA_DIR / "index"
DB_PATH = DATA_DIR / "transactions.db"
METRICS_DB = DATA_DIR / "metrics.db"
CAPTION_CACHE = DATA_DIR / "captions.json"

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# USD per 1M tokens as (input, output). Prices change, so check the
# provider's pricing page and edit this table when you switch models.
PRICES = {"gpt-4o-mini": (0.15, 0.60)}
