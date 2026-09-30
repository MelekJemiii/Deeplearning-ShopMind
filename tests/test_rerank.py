import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag import rerank  # noqa: E402


def test_fake_rerank_reorders_and_cuts():
    hits = [{"text": "laptop history", "chunk_id": "a"}, {"text": "RAM for data science laptop", "chunk_id": "b"}]
    out = rerank.FakeReranker().rerank("RAM data science", hits, 1)
    assert [h["chunk_id"] for h in out] == ["b"] and "rerank_score" in out[0]


def test_tei_client_maps_scores_back_to_input_order_across_batches(monkeypatch):
    calls = []

    class Resp:
        def __init__(self, data): self.data = data
        def raise_for_status(self): pass
        def json(self): return self.data

    def fake_post(url, json, timeout):
        calls.append(len(json["texts"]))
        # TEI returns results sorted by score, with the index inside the request batch
        scored = [{"index": i, "score": float(len(t))} for i, t in enumerate(json["texts"])]
        return Resp(sorted(scored, key=lambda x: -x["score"]))

    monkeypatch.setattr(rerank.requests, "post", fake_post)
    monkeypatch.setitem(rerank.CFG, "batch_size", 2)
    texts = ["a", "bbb", "cc", "dddd", "e"]
    assert rerank.TEIReranker().scores("q", texts) == [1.0, 3.0, 2.0, 4.0, 1.0]
    assert calls == [2, 2, 1]
