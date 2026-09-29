"""
Step 1 — Collect Wikipedia articles into the knowledge base.

- Article list:   config/wikipedia_articles.yaml
- Parameters:     config/settings.yaml  (collection.wikipedia)
- Contact email:  .env  (WIKIPEDIA_CONTACT_EMAIL)
Idempotent: already-downloaded articles are skipped, metadata is always rebuilt complete.
"""
import re
import sys
import time
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag.config import ROOT, SETTINGS, env, path  # noqa: E402
from shopmind_rag.logging_utils import get_logger  # noqa: E402
from shopmind_rag.metadata import replace_source_rows  # noqa: E402

log = get_logger("collect_wikipedia")
CFG = SETTINGS["collection"]["wikipedia"]
DROP = {s.lower() for s in CFG["drop_sections"]}
HEADERS = {"User-Agent": f"ShopMindAI/1.0 (ESPRIT academic project; contact: {env('WIKIPEDIA_CONTACT_EMAIL')})"}


def fetch_plaintext(title: str, lang: str) -> str | None:
    params = {"action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki",
              "redirects": 1, "titles": title, "format": "json", "formatversion": 2}
    for attempt in range(1, CFG["max_retries"] + 1):
        r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 5 * 2 ** (attempt - 1)))
            log.warning("429 on '%s' -> waiting %ss (attempt %d/%d)", title, wait, attempt, CFG["max_retries"])
            time.sleep(wait)
            continue
        r.raise_for_status()
        page = r.json()["query"]["pages"][0]
        return None if page.get("missing") else page.get("extract", "")
    raise RuntimeError("still rate-limited after retries")


def to_markdown(title: str, text: str) -> str:
    out, skipping, skip_level = [f"# {title}", ""], False, 0
    for line in text.split("\n"):
        m = re.match(r"^(=+)\s*(.*?)\s*=+$", line.strip())
        if m:
            level, name = len(m.group(1)), m.group(2)
            if name.lower() in DROP:
                skipping, skip_level = True, level
                continue
            if skipping and level <= skip_level:
                skipping = False
            if not skipping:
                out.append(f"{'#' * level} {name}")
        elif not skipping:
            out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"


def main():
    articles = yaml.safe_load(open(ROOT / "config" / "wikipedia_articles.yaml", encoding="utf-8"))["articles"]
    out_dir = path("markdown_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []

    for i, a in enumerate(articles, start=1):
        doc_id = f"kb_{i:03d}"
        md_path = out_dir / f"{doc_id}.md"
        url = f"https://{a['lang']}.wikipedia.org/wiki/{a['title'].replace(' ', '_')}"
        if md_path.exists():
            log.info("SKIP %s %s (already downloaded)", doc_id, a["title"])
        else:
            try:
                text = fetch_plaintext(a["title"], a["lang"])
            except Exception as e:  # network errors must not kill the whole run
                failed.append(a["title"]); log.error("FAIL %s: %s", a["title"], str(e)[:100]); continue
            if not text:
                failed.append(a["title"]); log.error("NOT FOUND %s (check title)", a["title"]); continue
            md_path.write_text(to_markdown(a["title"], text), encoding="utf-8")
            log.info("OK   %s %s", doc_id, a["title"])
            time.sleep(CFG["pause_between_s"])
        words = len(md_path.read_text(encoding="utf-8").split())
        rows.append([doc_id, a["title"], a["topic"], a["device"], a["lang"], "wikipedia", url, "CC BY-SA 4.0", words])

    total = replace_source_rows("wikipedia", rows)
    log.info("%d/%d Wikipedia articles, %d words | metadata now has %d docs",
             len(rows), len(articles), sum(r[-1] for r in rows), total)
    if failed:
        log.warning("Failed: %s  (rerun later; done ones are skipped)", ", ".join(failed))


if __name__ == "__main__":
    main()
