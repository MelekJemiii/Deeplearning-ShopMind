"""
Step 3 — Clean the knowledge base: data/markdown/*.md -> data/processed/*.md

Rules (parameters in config/settings.yaml -> cleaning):
  1. Structural noise (all docs): conversion markers, empty bullets, footnote refs, <br>, empty sections
  2. Repeated headers/footers (PDFs only): short lines repeated >= N times in the same document,
     detected statistically (digits masked), so no per-manufacturer hardcoding is needed
  3. Legal boilerplate (all docs): configurable regex list
Never modifies data/markdown/. Writes a per-document report to data/processed/cleaning_report.csv.
"""
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag.config import SETTINGS, path  # noqa: E402
from shopmind_rag.logging_utils import get_logger  # noqa: E402

log = get_logger("clean")
CFG = SETTINGS["cleaning"]
BOILERPLATE = [re.compile(p, re.I) for p in CFG["boilerplate_patterns"]]
HEADING = re.compile(r"^(#{1,6})\s+\S")


def structural(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"<!--.*?-->", "", text)                                  # conversion markers
    text = re.sub(r"<sup>\s*(\[\d+\]|\d+(,\d+)*|\*+)+\s*</sup>", "", text)  # footnote refs [1], 44,45, **
    text = re.sub(r"</?sup>", "", text)                                     # keep other superscript content
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"</?u>", "", text)
    lines = [re.sub(r"\s\*{2,3}$", "", l.rstrip()) for l in text.split("\n")]  # "** / ***" option flags
    lines = [l for l in lines if not re.fullmatch(r"\s*([-•*]|\*\*\s*\*\*)\s*", l)]  # empty bullets
    return "\n".join(lines)


def drop_repeated(text: str) -> tuple[str, int]:
    lines = text.split("\n")
    mask = lambda l: re.sub(r"\d+", "#", l.strip().strip("*").strip())  # noqa: E731
    counts = Counter(mask(l) for l in lines[1:] if l.strip() and not l.lstrip().startswith("|"))
    noisy = {k for k, c in counts.items()
             if c >= CFG["repeated_line_min_count"] and len(k) <= CFG["repeated_line_max_chars"]}
    kept = [lines[0]] + [l for l in lines[1:]
                         if l.lstrip().startswith("|") or mask(l) not in noisy]
    return "\n".join(kept), len(lines) - len(kept)


def drop_boilerplate(text: str) -> tuple[str, int]:
    lines = text.split("\n")
    kept = [l for l in lines if not any(p.search(l) for p in BOILERPLATE)]
    return "\n".join(kept), len(lines) - len(kept)


def drop_empty_sections(text: str) -> str:
    """Remove a heading whose section has no content before the next heading of same/higher level."""
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        m = HEADING.match(line)
        if m and i > 0:
            level, j = len(m.group(1)), i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            nxt = HEADING.match(lines[j]) if j < len(lines) else None
            if j >= len(lines) or (nxt and len(nxt.group(1)) <= level):
                continue
        out.append(line)
    return "\n".join(out)


def clean(text: str, source_type: str) -> tuple[str, dict]:
    stats = {}
    text = structural(text)
    if source_type == "manufacturer_pdf":
        text, stats["repeated_lines"] = drop_repeated(text)
    text, stats["boilerplate_lines"] = drop_boilerplate(text)
    for _ in range(3):  # removing a section can make its parent empty
        text = drop_empty_sections(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text, stats


def main():
    with open(path("metadata_csv"), encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    src_dir, out_dir = path("markdown_dir"), path("processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    report, totals = [], defaultdict(lambda: [0, 0])

    for m in meta:
        src = src_dir / f"{m['doc_id']}.md"
        if not src.exists():
            log.error("MISSING %s", src.name); continue
        raw = src.read_text(encoding="utf-8")
        cleaned, stats = clean(raw, m["source_type"])
        (out_dir / src.name).write_text(cleaned, encoding="utf-8")
        # count words on the same basis (<br> glues words in raw text)
        before, after = len(re.sub(r"<br\s*/?>", " ", raw).split()), len(cleaned.split())
        pct = 100 * (before - after) / max(before, 1)
        totals[m["source_type"]][0] += before
        totals[m["source_type"]][1] += after
        report.append({"doc_id": m["doc_id"], "source_type": m["source_type"], "words_before": before,
                       "words_after": after, "removed_pct": round(pct, 1),
                       "repeated_lines": stats.get("repeated_lines", 0),
                       "boilerplate_lines": stats["boilerplate_lines"]})
        if after < CFG["min_doc_words"]:
            log.warning("SHORT %s: only %d words after cleaning", m["doc_id"], after)

    with open(out_dir / "cleaning_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader(); w.writerows(report)

    for st, (b, a) in sorted(totals.items()):
        log.info("%-17s %7d -> %7d words  (-%.1f%%)", st, b, a, 100 * (b - a) / max(b, 1))
    b, a = sum(v[0] for v in totals.values()), sum(v[1] for v in totals.values())
    log.info("TOTAL             %7d -> %7d words  (-%.1f%%) | %d docs -> %s", b, a, 100 * (b - a) / max(b, 1),
             len(report), out_dir)


if __name__ == "__main__":
    main()
