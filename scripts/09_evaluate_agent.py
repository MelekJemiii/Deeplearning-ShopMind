"""
Step 8 — End-to-end evaluation of the n8n RAG Agent over the frozen test set.

Sends every test query to the agent's webhook, stores the validated output, then computes
agent-level metrics: abstention on out-of-scope questions, citation of a relevant document,
guardrail activations (invalid citations / wrong numbers removed), search behavior, latency.
Results are also grouped by the model that answered (`model_used`), because the agent has a
fallback chat model: one evaluation run may mix models, and metrics must not hide it.

The Router is bypassed on purpose: `needs` only carries the language and an empty profile,
so the measure isolates the RAG Agent from the quality of other agents.

Resumable: answers are appended to eval/agent_results/raw.jsonl; rerunning skips done queries.
Circuit breaker: stops after N queries fail in a row (quota exhausted or workflow broken).

Prerequisites: n8n workflow "RAG Agent" ACTIVE with its Webhook trigger; N8N_AGENT_URL in .env
Usage: python scripts/09_evaluate_agent.py [--limit 5] [--only q01 q29] [--report-only]
"""
import json
import sys
import time
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag.agent_metrics import aggregate, score_row  # noqa: E402
from shopmind_rag.config import ROOT, SETTINGS, env  # noqa: E402
from shopmind_rag.logging_utils import get_logger  # noqa: E402

log = get_logger("evaluate_agent")
CFG = SETTINGS["agent_evaluation"]
MAX_CONSECUTIVE_FAILURES = CFG.get("max_consecutive_failures", 3)


def arg_list(flag):
    if flag not in sys.argv:
        return None
    i, out = sys.argv.index(flag) + 1, []
    while i < len(sys.argv) and not sys.argv[i].startswith("--"):
        out.append(sys.argv[i]); i += 1
    return out


def call_agent(url: str, q: dict) -> dict:
    payload = {"question": q["query"], "language": q["lang"],
               "needs": {"intent": None, "category": None, "budget_tnd": {"min": None, "max": None},
                         "use_cases": [], "must_have": [], "brand_prefs": [], "brand_excluded": []}}
    for attempt in range(1, CFG["max_retries"] + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=CFG["timeout_s"])
            if r.status_code == 200:
                data = r.json()
                data = data[0] if isinstance(data, list) else data
                data["_latency_s"] = round(time.perf_counter() - t0, 1)
                return data
            log.warning("%s: HTTP %s %s", q["id"], r.status_code, r.text[:150])
        except requests.RequestException as e:
            log.warning("%s: %s", q["id"], str(e)[:150])
        if attempt < CFG["max_retries"]:
            time.sleep(30 * attempt)  # rate limits: back off
    return {"_error": "failed after retries"}


def write_report(out_dir: Path, m: dict, rows: list[dict]):
    i, o, g, b = m["in_scope"], m["out_of_scope"], m["guardrails"], m["behavior"]
    pct = lambda x: "n/a" if x is None else f"{x:.0%}"  # noqa: E731
    models = ", ".join(f"{k} ({v})" for k, v in m["models_used"].items()) or "n/a"
    md = [
        "# RAG Agent — end-to-end evaluation", "",
        f"_{m['n_ok']}/{m['n_queries']} queries answered by the workflow. Router bypassed (empty needs): "
        "measures the RAG Agent alone._", "",
        f"**Models that answered:** {models}", "",
        "## Answer quality (all answered queries)", "",
        "| Metric | Value |", "|---|---|",
        f"| In-scope: at least one fact returned | {pct(i['answered_rate'])} (n={i['n']}) |",
        f"| In-scope: agent retrieved a relevant document | {pct(i['retrieved_relevant_doc_rate'])} |",
        f"| In-scope: agent **cited** a relevant document | {pct(i['cited_relevant_doc_rate'])} |",
        f"| In-scope: wrongly abstained (not_found) | {pct(i['false_abstention_rate'])} |",
        f"| Out-of-scope: correctly abstained | {pct(o['correct_abstention_rate'])} (n={o['n']}) |", ""]
    if m.get("per_model") and len(m["per_model"]) > 1:
        md += ["## Per model (fallback mixed models in this run)", "",
               "| Model | Queries | Cited relevant doc | Wrongly abstained | Correct abstention (OOS) "
               "| Tool calls (mean) | Guardrail removals | Latency (s) |",
               "|---|---|---|---|---|---|---|---|"]
        for name, c in m["per_model"].items():
            md.append(f"| {name} | {c['n']} | {pct(c['cited_relevant_doc_rate'])} | {pct(c['false_abstention_rate'])} "
                      f"| {pct(c['correct_abstention_rate'])} | {c['tool_calls_mean']} | {c['guardrail_removals']} "
                      f"| {c['latency_mean_s']} |")
        md.append("")
    md += ["## By query type (in-scope)", "", "| Type | n | Answered | Cited relevant doc |", "|---|---|---|---|"]
    for t, v in m["by_type"].items():
        md.append(f"| {t} | {v['n']} | {pct(v['answered_rate'])} | {pct(v['cited_relevant_doc_rate'])} |")
    md += ["", "## Guardrails (deterministic validation node)", "",
           f"- Facts removed for citing a chunk never retrieved: **{g['invalid_citations_removed']}**",
           f"- Facts removed for a number absent from the cited chunk: **{g['number_mismatch_facts_removed']}**",
           f"- Queries where a removal happened: {', '.join(g['queries_with_a_removal']) or 'none'}", "",
           "## Search behavior", "",
           f"- Mean tool calls per query: **{b['tool_calls_mean']}**",
           f"- One search per answered question: {pct(b['one_search_per_question_rate'])}", "",
           "| Query type | source_type chosen by the agent (counts) |", "|---|---|"]
    for t, c in sorted(b["source_type_by_query_type"].items()):
        md.append(f"| {t} | {', '.join(f'{k}: {v}' for k, v in sorted(c.items()))} |")
    if m["latency_s"]:
        md += ["", f"**Latency (end-to-end, per query):** mean {m['latency_s']['mean']} s, max {m['latency_s']['max']} s"]
    md += ["", "## Per query", "",
           "| id | type | answered | cited relevant doc | tool calls | sources | model | latency (s) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["ok"]:
            md.append(f"| {r['id']} | {r['type']} | {'yes' if r['answered'] else 'no'} | "
                      f"{'yes' if r['cited_relevant_doc'] else 'no'} | {r['tool_calls']} | "
                      f"{', '.join(r['source_types']) or '-'} | {r['model_used']} | {r['latency_s']} |")
        else:
            md.append(f"| {r['id']} | {r['type']} | ERROR | | | | | |")
    if m["errors"]:
        md += ["", f"Not evaluated yet or failed (rerun to retry): {', '.join(m['errors'])}"]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main():
    queries = yaml.safe_load(open(ROOT / "eval" / "test_queries.yaml", encoding="utf-8"))["queries"]
    out_dir = ROOT / CFG["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.jsonl"
    done = {}
    if raw_path.exists():
        for line in open(raw_path, encoding="utf-8"):
            rec = json.loads(line)
            if "_error" not in rec["response"]:
                done[rec["id"]] = rec["response"]

    if "--report-only" not in sys.argv:
        url = env("N8N_AGENT_URL")
        only, limit = arg_list("--only"), arg_list("--limit")
        todo = [q for q in queries if q["id"] not in done and (not only or q["id"] in only)]
        todo = todo[:int(limit[0])] if limit else todo
        log.info("%d already done, %d to run -> %s", len(done), len(todo), url)
        consecutive_failures = 0
        for n, q in enumerate(todo, 1):
            resp = call_agent(url, q)
            with open(raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": q["id"], "response": resp}, ensure_ascii=False) + "\n")
            if "_error" not in resp:
                consecutive_failures = 0
                done[q["id"]] = resp
                v = resp.get("validation", {})
                log.info("[%d/%d] %s %-13s answered=%s calls=%s %.1fs  (%s)", n, len(todo), q["id"], q["type"],
                         any(x.get("status") == "answered" for x in resp.get("questions", [])),
                         v.get("tool_calls"), resp["_latency_s"], resp.get("model_used", "unknown model"))
            else:
                consecutive_failures += 1
                log.error("[%d/%d] %s failed", n, len(todo), q["id"])
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    log.error("%d queries failed in a row (quota exhausted or workflow broken): stopping. "
                              "Check n8n Executions, then rerun to resume.", consecutive_failures)
                    break
            if n < len(todo):
                time.sleep(CFG["pause_between_queries_s"])

    rows = [score_row(q, done.get(q["id"])) for q in queries]
    m = aggregate(rows)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out_dir, m, rows)
    i, o = m["in_scope"], m["out_of_scope"]
    log.info("%d/%d done | in-scope cited relevant doc %s | out-of-scope correct abstention %s | models %s | "
             "report: %s", m["n_ok"], m["n_queries"], i["cited_relevant_doc_rate"],
             o["correct_abstention_rate"], m["models_used"], out_dir / "summary.md")


if __name__ == "__main__":
    main()
