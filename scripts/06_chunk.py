"""
Step 4 — Chunk the cleaned knowledge base with each configured strategy.

Input:  data/processed/*.md + data/kb_metadata.csv
Output: data/chunks/chunks_<S>.jsonl (one JSON chunk per line, with metadata)
        data/chunks/stats_<S>.json  (size distribution, per source type)
Also reports, for each test query, whether its evidence is still contained in ONE chunk
(evidence split across a boundary cannot be retrieved as a whole).

Usage: python scripts/06_chunk.py            # all strategies
       python scripts/06_chunk.py --strategy B
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag.chunking import Tokenizer, chunk_headings, chunk_recursive  # noqa: E402
from shopmind_rag.config import ROOT, SETTINGS, path  # noqa: E402
from shopmind_rag.logging_utils import get_logger  # noqa: E402
from shopmind_rag.text import contains_evidence  # noqa: E402

log = get_logger("chunk")
CFG = SETTINGS["chunking"]
META_FIELDS = ["title", "topic", "device_category", "language", "source_type", "source_url"]


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def run(name: str, scfg: dict, meta: list[dict], tok: Tokenizer, out_dir: Path) -> list[dict]:
    chunks = []
    for m in meta:
        src = path("processed_dir") / f"{m['doc_id']}.md"
        if not src.exists():
            log.error("MISSING %s (run 05_clean.py first)", src.name); continue
        md = src.read_text(encoding="utf-8")
        if scfg["method"] == "recursive":
            parts = chunk_recursive(md, scfg, tok)
        elif scfg["method"] == "headings":
            parts = chunk_headings(md, scfg, tok, m["title"])
        else:
            raise ValueError(f"unknown method {scfg['method']}")
        for i, c in enumerate(parts):
            chunks.append({"chunk_id": f"{m['doc_id']}#{name}{i:03d}", "doc_id": m["doc_id"], "strategy": name,
                           "text": c.text, "n_tokens": tok.count(c.text), "heading_path": c.heading_path,
                           **{k: m[k] for k in META_FIELDS}})

    with open(out_dir / f"chunks_{name}.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sizes = [c["n_tokens"] for c in chunks]
    by_src = defaultdict(list)
    for c in chunks:
        by_src[c["source_type"]].append(c["n_tokens"])
    stats = {"strategy": name, "config": scfg, "n_chunks": len(chunks), "total_tokens": sum(sizes),
             "mean": round(statistics.mean(sizes), 1), "median": statistics.median(sizes),
             "p5": pct(sizes, 5), "p95": pct(sizes, 95), "min": min(sizes), "max": max(sizes),
             "tiny_chunks_lt_30": sum(s < 30 for s in sizes),
             "by_source": {k: {"n": len(v), "mean": round(statistics.mean(v), 1)} for k, v in by_src.items()}}
    (out_dir / f"stats_{name}.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("[%s] %d chunks | tokens mean %.0f, median %d, p5 %d, p95 %d, max %d | <30 tokens: %d",
             name, len(chunks), stats["mean"], stats["median"], stats["p5"], stats["p95"], stats["max"],
             stats["tiny_chunks_lt_30"])
    over = [c["chunk_id"] for c in chunks if c["n_tokens"] > scfg["max_tokens"]]
    if over:
        log.warning("[%s] %d chunks exceed max_tokens, e.g. %s", name, len(over), over[:3])
    return chunks


def evidence_coverage(name: str, chunks: list[dict]):
    """For each (query, relevant doc), is the evidence fully inside at least one chunk?"""
    qs = yaml.safe_load(open(ROOT / "eval" / "test_queries.yaml", encoding="utf-8"))["queries"]
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c["text"])
    missing = []
    for q in qs:
        for r in q.get("relevant") or []:
            if r["doc_id"] not in by_doc:
                missing.append(f"{q['id']}/{r['doc_id']}(no chunks)"); continue
            if not any(contains_evidence(t, r["evidence"]) for t in by_doc[r["doc_id"]]):
                missing.append(f"{q['id']}/{r['doc_id']}")
    total = sum(len(q.get("relevant") or []) for q in qs)
    log.info("[%s] evidence contained in a single chunk: %d/%d%s", name, total - len(missing), total,
             f" | split: {missing}" if missing else "")


def main():
    with open(path("metadata_csv"), encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    out_dir = ROOT / CFG["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(CFG["tokenizer"])
    only = sys.argv[sys.argv.index("--strategy") + 1] if "--strategy" in sys.argv else None
    for name, scfg in CFG["strategies"].items():
        if only and name != only:
            continue
        evidence_coverage(name, run(name, scfg, meta, tok, out_dir))


if __name__ == "__main__":
    main()
