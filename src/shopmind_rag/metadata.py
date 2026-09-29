"""Shared KB metadata handling.

Each collector owns the rows of its own source_type: it replaces them and keeps
all other rows, so running one script never erases another source's metadata.
"""
import csv

from .config import path

FIELDS = ["doc_id", "title", "topic", "device_category", "language",
          "source_type", "source_url", "license", "word_count"]


def replace_source_rows(source_type: str, new_rows: list[list]) -> int:
    meta = path("metadata_csv")
    kept = []
    if meta.exists():
        with open(meta, encoding="utf-8", newline="") as f:
            kept = [r for r in csv.DictReader(f) if r["source_type"] != source_type]
    rows = kept + [dict(zip(FIELDS, r)) for r in new_rows]
    rows.sort(key=lambda r: r["doc_id"])
    with open(meta, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
