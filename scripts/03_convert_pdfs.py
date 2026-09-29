"""
Step 1B — Convert manufacturer PDFs to Markdown (tables preserved).

- PDF list:  config/pdf_sources.yaml
- Input:     data/raw/pdf/*.pdf
- Output:    data/markdown/pdf_XXX.md + rows in kb_metadata.csv (source_type=manufacturer_pdf)

Uses pymupdf4llm: converts layout to Markdown, including tables as Markdown tables
and larger fonts as headings. Flags PDFs with no text layer (scanned -> would need OCR).
Idempotent: rerun after adding PDFs or editing the YAML.
"""
import sys
from pathlib import Path

import pymupdf
import pymupdf4llm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag.config import ROOT, path  # noqa: E402
from shopmind_rag.logging_utils import get_logger  # noqa: E402
from shopmind_rag.metadata import replace_source_rows  # noqa: E402

log = get_logger("convert_pdfs")
REQUIRED = {"file", "title", "topic", "device", "lang"}
MIN_CHARS_PER_PAGE = 200  # below this, the PDF is probably scanned (image only)


def parse_pages(spec: str | None, total: int) -> list[int] | None:
    """'21-35' or '3,5,10-12' (1-based, as printed in a PDF viewer) -> 0-based page list."""
    if not spec:
        return None
    pages = []
    for part in str(spec).split(","):
        a, _, b = part.strip().partition("-")
        start, end = int(a), int(b or a)
        pages.extend(range(start - 1, min(end, total)))
    return pages


def convert(pdf_path: Path, page_spec: str | None) -> tuple[str, int]:
    with pymupdf.open(pdf_path) as doc:
        pages = parse_pages(page_spec, doc.page_count)
    md = pymupdf4llm.to_markdown(str(pdf_path), pages=pages, show_progress=False)
    return md, len(pages) if pages else pymupdf.open(pdf_path).page_count


def main():
    cfg = yaml.safe_load(open(ROOT / "config" / "pdf_sources.yaml", encoding="utf-8")) or {}
    entries = cfg.get("pdfs") or []
    raw_dir, out_dir = path("raw_pdf_dir"), path("markdown_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    listed = {e.get("file") for e in entries}
    for f in raw_dir.glob("*.pdf"):
        if f.name not in listed:
            log.warning("UNLISTED %s is in data/raw/pdf but not in pdf_sources.yaml -> ignored", f.name)

    rows = []
    for i, e in enumerate(entries, start=1):
        doc_id = f"pdf_{i:03d}"
        missing = REQUIRED - e.keys()
        if missing:
            log.error("SKIP entry %d: missing fields %s", i, missing); continue
        pdf_path = raw_dir / e["file"]
        if not pdf_path.exists():
            log.error("SKIP %s: file not found in %s", e["file"], raw_dir); continue
        try:
            md, pages = convert(pdf_path, e.get("pages"))
        except Exception as ex:  # corrupted / encrypted PDFs must not stop the run
            log.error("FAIL %s: %s", e["file"], str(ex)[:100]); continue

        chars_per_page = len(md.strip()) / max(pages, 1)
        if chars_per_page < MIN_CHARS_PER_PAGE:
            log.warning("LOW TEXT %s: %.0f chars/page -> probably scanned, needs OCR", e["file"], chars_per_page)
        n_tables = md.count("\n|---") + md.count("\n|:--")

        # Keep the PDF's own title if it starts with a heading, else add ours
        body = md.strip() + "\n" if md.lstrip().startswith("#") else f"# {e['title']}\n\n{md.strip()}\n"
        (out_dir / f"{doc_id}.md").write_text(body, encoding="utf-8")
        words = len(body.split())
        rows.append([doc_id, e["title"], e["topic"], e["device"], e["lang"], "manufacturer_pdf",
                     e.get("source_url", ""), "manufacturer public document (academic use)", words])
        log.info("OK   %s %s | %d pages, %d words, ~%d tables", doc_id, e["file"], pages, words, n_tables)

    total = replace_source_rows("manufacturer_pdf", rows)
    log.info("%d/%d PDFs converted | metadata now has %d docs", len(rows), len(entries), total)


if __name__ == "__main__":
    main()
