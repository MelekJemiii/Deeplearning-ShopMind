import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag.agent_metrics import aggregate, score_row  # noqa: E402

Q_IN = {"id": "q01", "type": "use_case", "lang": "fr", "query": "RAM ?",
        "relevant": [{"doc_id": "guide_001", "evidence": ["16 Go"]}]}
Q_OOS = {"id": "q29", "type": "out_of_scope", "lang": "fr", "query": "prix ?", "relevant": []}


def resp(status, cited=(), retrieved=(), removed_num=0, sources=("team_written",)):
    return {"questions": [{"question": "x", "status": status,
                           "facts": [{"fact": "f", "chunk_id": c} for c in cited]}],
            "validation": {"tool_calls": len(sources), "one_search_per_question": True,
                           "retrieved_chunk_ids": list(retrieved), "queries_used": [{"source_type": s} for s in sources],
                           "removed_invalid_citations": [], "removed_number_mismatch": [{}] * removed_num},
            "_latency_s": 12.0}


def test_in_scope_good_answer():
    r = score_row(Q_IN, resp("answered", ["guide_001#B001"], ["guide_001#B001", "kb_001#B002"]))
    assert r["answered"] and r["cited_relevant_doc"] and r["retrieved_relevant_doc"] and not r["abstained"]


def test_out_of_scope_abstention_and_aggregate():
    rows = [score_row(Q_IN, resp("answered", ["kb_001#B002"], ["kb_001#B002"], removed_num=1)),
            score_row(Q_OOS, resp("not_found", sources=())),
            score_row({**Q_IN, "id": "q02"}, None)]
    m = aggregate(rows)
    assert m["out_of_scope"]["correct_abstention_rate"] == 1.0
    assert m["in_scope"]["cited_relevant_doc_rate"] == 0.0          # answered, but from a non-relevant doc
    assert m["guardrails"]["number_mismatch_facts_removed"] == 1
    assert m["errors"] == ["q02"] and m["n_ok"] == 2


def test_per_model_grouping():
    a = resp("answered", ["guide_001#B001"], ["guide_001#B001"]); a["model_used"] = "primary: gemini"
    b = resp("answered", ["kb_001#B002"], ["kb_001#B002"]); b["model_used"] = "fallback: groq"
    m = aggregate([score_row(Q_IN, a), score_row({**Q_IN, "id": "q02"}, b)])
    assert m["models_used"] == {"primary: gemini": 1, "fallback: groq": 1}
    assert m["per_model"]["primary: gemini"]["cited_relevant_doc_rate"] == 1.0
    assert m["per_model"]["fallback: groq"]["cited_relevant_doc_rate"] == 0.0
