"""Agent-level metrics computed from the RAG Agent's validated outputs over the test set."""
import statistics
from collections import Counter, defaultdict


def doc_of(chunk_id: str) -> str:
    return chunk_id.split("#")[0]


def score_row(q: dict, resp: dict | None) -> dict:
    """One test query + the agent's validated JSON output -> per-query measures."""
    relevant = {r["doc_id"] for r in q.get("relevant") or []}
    base = {"id": q["id"], "type": q["type"], "lang": q["lang"], "query": q["query"]}
    if not resp or "questions" not in resp:
        return {**base, "ok": False}
    v = resp.get("validation", {})
    questions = resp["questions"]
    cited = [f["chunk_id"] for qq in questions for f in qq.get("facts", [])]
    retrieved = v.get("retrieved_chunk_ids", [])
    queries = v.get("queries_used", [])
    return {
        **base, "ok": True,
        "answered": any(qq.get("status") == "answered" for qq in questions),
        "abstained": all(qq.get("status") == "not_found" for qq in questions),
        "retrieved_relevant_doc": bool(relevant & {doc_of(c) for c in retrieved}),
        "cited_relevant_doc": bool(relevant & {doc_of(c) for c in cited}),
        "n_facts": len(cited),
        "tool_calls": v.get("tool_calls", 0),
        "questions": len(questions),
        "one_search_per_question": v.get("one_search_per_question"),
        "removed_invalid_citations": len(v.get("removed_invalid_citations", [])),
        "removed_number_mismatch": len(v.get("removed_number_mismatch", [])),
        "source_types": [x.get("source_type", "?") for x in queries],
        "latency_s": resp.get("_latency_s"),
        "model_used": resp.get("model_used", "unknown"),
    }


def aggregate(rows: list[dict], by_model: bool = True) -> dict:
    ok = [r for r in rows if r["ok"]]
    ins = [r for r in ok if r["type"] != "out_of_scope"]
    oos = [r for r in ok if r["type"] == "out_of_scope"]
    rate = lambda xs, key: round(sum(bool(x[key]) for x in xs) / len(xs), 3) if xs else None  # noqa: E731

    by_type = defaultdict(list)
    for r in ins:
        by_type[r["type"]].append(r)
    src_by_type = {t: dict(Counter(s for r in rs for s in r["source_types"])) for t, rs in by_type.items()}
    lat = [r["latency_s"] for r in ok if r["latency_s"] is not None]
    return {
        "n_queries": len(rows), "n_ok": len(ok), "errors": [r["id"] for r in rows if not r["ok"]],
        "in_scope": {
            "n": len(ins),
            "answered_rate": rate(ins, "answered"),
            "retrieved_relevant_doc_rate": rate(ins, "retrieved_relevant_doc"),
            "cited_relevant_doc_rate": rate(ins, "cited_relevant_doc"),
            "false_abstention_rate": rate(ins, "abstained"),
        },
        "out_of_scope": {"n": len(oos), "correct_abstention_rate": rate(oos, "abstained"),
                         "answered_ids": [r["id"] for r in oos if r["answered"]]},
        "behavior": {
            "tool_calls_mean": round(statistics.mean(r["tool_calls"] for r in ok), 2) if ok else None,
            "one_search_per_question_rate": rate([r for r in ok if r["answered"]], "one_search_per_question"),
            "source_type_by_query_type": src_by_type,
        },
        "guardrails": {
            "invalid_citations_removed": sum(r["removed_invalid_citations"] for r in ok),
            "number_mismatch_facts_removed": sum(r["removed_number_mismatch"] for r in ok),
            "queries_with_a_removal": [r["id"] for r in ok
                                       if r["removed_invalid_citations"] or r["removed_number_mismatch"]],
        },
        "by_type": {t: {"n": len(rs), "cited_relevant_doc_rate": rate(rs, "cited_relevant_doc"),
                        "answered_rate": rate(rs, "answered")} for t, rs in sorted(by_type.items())},
        "latency_s": {"mean": round(statistics.mean(lat), 1), "max": round(max(lat), 1)} if lat else None,
        "models_used": dict(Counter(r["model_used"] for r in ok)),
        # same metrics per model: fallback may have mixed models within one evaluation run
        "per_model": {m: _core(rs) for m, rs in _group(ok, "model_used").items()} if by_model else None,
    }


def _group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return dict(sorted(g.items()))


def _core(rs: list[dict]) -> dict:
    ins = [r for r in rs if r["type"] != "out_of_scope"]
    oos = [r for r in rs if r["type"] == "out_of_scope"]
    rate = lambda xs, key: round(sum(bool(x[key]) for x in xs) / len(xs), 3) if xs else None  # noqa: E731
    lat = [r["latency_s"] for r in rs if r["latency_s"] is not None]
    return {"n": len(rs), "n_in_scope": len(ins), "n_out_of_scope": len(oos),
            "cited_relevant_doc_rate": rate(ins, "cited_relevant_doc"),
            "false_abstention_rate": rate(ins, "abstained"),
            "correct_abstention_rate": rate(oos, "abstained"),
            "tool_calls_mean": round(statistics.mean(r["tool_calls"] for r in rs), 2),
            "guardrail_removals": sum(r["removed_invalid_citations"] + r["removed_number_mismatch"] for r in rs),
            "latency_mean_s": round(statistics.mean(lat), 1) if lat else None}
