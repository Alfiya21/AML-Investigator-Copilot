import argparse

from aml_copilot import config
from aml_copilot.ingestion import captioning, chunking, pdf_parser
from aml_copilot.retrieval.index import HybridIndex


def main():
    ap = argparse.ArgumentParser(description="Parse PDFs in data/raw_pdfs and build the search index")
    ap.add_argument("--no-vision", action="store_true", help="skip figure captioning (no API calls)")
    args = ap.parse_args()

    pdfs = sorted(config.PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs found in {config.PDF_DIR}")

    pages = []
    for pdf in pdfs:
        parsed = pdf_parser.parse_pdf(pdf, config.PAGE_IMG_DIR)
        print(f"{pdf.name}: {len(parsed)} pages, "
              f"{sum(p.image_path is not None for p in parsed)} visual, "
              f"{sum(len(p.tables) for p in parsed)} tables")
        pages += parsed

    captions = {} if args.no_vision else captioning.caption_pages(pages)
    chunks = chunking.build_chunks(pages, captions)
    kinds = {k: sum(c.kind == k for c in chunks) for k in ("text", "table", "figure")}
    print(f"{len(chunks)} chunks: {kinds}")

    folder = HybridIndex.build(chunks).save(config.INDEX_DIR)
    print(f"index saved to {folder}")


if __name__ == "__main__":
    main()
