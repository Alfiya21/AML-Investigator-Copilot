from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf
import pdfplumber


@dataclass
class Page:
    doc: str
    number: int
    text: str
    tables: list = field(default_factory=list)
    image_path: Path | None = None


def looks_visual(page):
    # Charts are usually drawn with vector ops, not embedded, so count drawings too.
    # Tiny images (logos, bullets) are ignored or every page would get captioned.
    big_images = [i for i in page.get_image_info() if i["width"] > 200 and i["height"] > 200]
    return bool(big_images) or len(page.get_drawings()) > 25


def parse_pdf(pdf_path, img_dir, dpi=110):
    pdf_path = Path(pdf_path)
    img_dir = Path(img_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    pages = []
    doc = fitz.open(pdf_path)
    with pdfplumber.open(pdf_path) as plumber:
        for i, page in enumerate(doc):
            number = i + 1
            tables = [t for t in plumber.pages[i].extract_tables() if t and len(t) > 1]

            image_path = None
            if looks_visual(page):
                image_path = img_dir / f"{pdf_path.stem}_p{number}.png"
                if not image_path.exists():
                    page.get_pixmap(dpi=dpi).save(str(image_path))

            pages.append(Page(pdf_path.stem, number, page.get_text("text"), tables, image_path))
    doc.close()
    return pages
