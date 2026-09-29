"""
Step 1C — Add the team-written French guides to the knowledge base.

Source (versioned in git): content/guides/*.md, each with a YAML front matter:
    ---
    title: ...
    topic: use_case_guide | compatibility_guide | glossary
    device: laptop | smartphone | accessory | general
    lang: fr
    ---
Output: data/markdown/guide_XXX.md (front matter removed) + rows in kb_metadata.csv
Idempotent: rerun after editing a guide to refresh it.
"""
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag.config import path  # noqa: E402
from shopmind_rag.logging_utils import get_logger  # noqa: E402
from shopmind_rag.metadata import replace_source_rows  # noqa: E402

log = get_logger("add_guides")
REQUIRED = {"title", "topic", "device", "lang"}


def split_front_matter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        raise ValueError("missing front matter")
    _, fm, body = text.split("---", 2)
    meta = yaml.safe_load(fm)
    missing = REQUIRED - meta.keys()
    if missing:
        raise ValueError(f"missing fields: {missing}")
    return meta, body.strip() + "\n"


def main():
    src_dir, out_dir = path("guides_dir"), path("markdown_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, f in enumerate(sorted(src_dir.glob("*.md")), start=1):
        doc_id = f"guide_{i:03d}"
        try:
            meta, body = split_front_matter(f.read_text(encoding="utf-8"))
        except ValueError as e:
            log.error("SKIP %s: %s", f.name, e)
            continue
        (out_dir / f"{doc_id}.md").write_text(body, encoding="utf-8")
        words = len(body.split())
        rows.append([doc_id, meta["title"], meta["topic"], meta["device"], meta["lang"],
                     "team_written", f"content/guides/{f.name}", "own (AI-assisted, team-reviewed)", words])
        log.info("OK   %s %s (%d words)", doc_id, f.name, words)
    total = replace_source_rows("team_written", rows)
    log.info("%d guides added, %d words | metadata now has %d docs",
             len(rows), sum(r[-1] for r in rows), total)


if __name__ == "__main__":
    main()
