import base64
import json
from pathlib import Path

from .. import config, llm, prompts


def _load_cache():
    if config.CAPTION_CACHE.exists():
        return json.loads(config.CAPTION_CACHE.read_text(encoding="utf-8"))
    return {}


def caption_pages(pages):
    """Caption every page that has a rendered image. Results are cached on disk so a
    crashed or repeated run doesn't pay for the same page twice."""
    cache = _load_cache()
    prompt = prompts.load("caption")

    for page in pages:
        if page.image_path is None:
            continue
        key = Path(page.image_path).name
        if key in cache:
            continue
        b64 = base64.b64encode(Path(page.image_path).read_bytes()).decode()
        msg = llm.chat(
            [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            step="caption", model=config.VISION_MODEL,
        )
        cache[key] = (msg.content or "").strip()
        config.CAPTION_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache


def table_to_markdown(table):
    rows = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in table]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)
