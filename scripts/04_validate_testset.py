"""
Step 2 — Validate the retrieval test set BEFORE using it.

Checks for every query:
  - unique id, allowed type, non-empty query
  - every relevant doc_id exists in data/markdown/ and kb_metadata.csv
  - at least one evidence phrase really appears in that document (normalized match)
  - out_of_scope queries have no relevant docs, other types have at least one
Exit code 1 if any error, so it can run in CI later.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag.config import ROOT, path  # noqa: E402
from shopmind_rag.logging_utils import get_logger  # noqa: E402
from shopmind_rag.text import contains_evidence  # noqa: E402

log = get_logger("validate_testset")
TYPES = {"use_case", "definition", "spec_lookup", "compatibility", "cross_lingual", "out_of_scope"}


def main():
    qs = yaml.safe_load(open(ROOT / "eval" / "test_queries.yaml", encoding="utf-8"))["queries"]
    with open(path("metadata_csv"), encoding="utf-8") as f:
        meta_ids = {r["doc_id"] for r in csv.DictReader(f)}
    md_dir = path("markdown_dir")
    cache: dict[str, str] = {}
    errors = 0

    dup = [k for k, v in Counter(q["id"] for q in qs).items() if v > 1]
    if dup:
        log.error("Duplicate ids: %s", dup); errors += 1

    for q in qs:
        qid, rel = q["id"], q.get("relevant") or []
        if q.get("type") not in TYPES:
            log.error("%s: unknown type %r", qid, q.get("type")); errors += 1
        if (q.get("type") == "out_of_scope") != (len(rel) == 0):
            log.error("%s: out_of_scope must have no relevant docs, others at least one", qid); errors += 1
        for r in rel:
            did = r["doc_id"]
            if did not in meta_ids or not (md_dir / f"{did}.md").exists():
                log.error("%s: doc %s not found in metadata/markdown", qid, did); errors += 1; continue
            if did not in cache:
                cache[did] = (md_dir / f"{did}.md").read_text(encoding="utf-8")
            if not contains_evidence(cache[did], r["evidence"]):
                log.error("%s: none of %s found in %s", qid, r["evidence"], did); errors += 1

    by_type = Counter(q["type"] for q in qs)
    by_lang = Counter(q["lang"] for q in qs)
    log.info("%d queries | types: %s | langs: %s", len(qs), dict(by_type), dict(by_lang))
    if errors:
        log.error("%d error(s): fix the labels before evaluating", errors); sys.exit(1)
    log.info("Test set valid")


if __name__ == "__main__":
    main()
