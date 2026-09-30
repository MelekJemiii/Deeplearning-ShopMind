import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shopmind_rag import sparse  # noqa: E402


def test_tokenize_keeps_spec_tokens_and_strips_accents_stopwords():
    toks = sparse.tokenize("Quel écran pour le Legion 5 15IAX11 ? IP68, 115W")
    assert "ecran" in toks and "15iax11" in toks and "ip68" in toks and "115w" in toks
    assert "quel" not in toks and "pour" not in toks and "le" not in toks


def test_ids_are_stable():
    assert sparse.token_id("ip68") == sparse.token_id("ip68")
    assert sparse.token_id("ip68") != sparse.token_id("ip67")


def test_bm25_saturation_and_length_norm():
    idx1, v1 = sparse.doc_vector("ram ram ram", avgdl=3)
    idx2, v2 = sparse.doc_vector("ram", avgdl=3)
    assert v1[0] > v2[0]            # more occurrences -> higher weight
    assert v1[0] < 3 * v2[0]        # ...but saturating, not linear


def test_query_vector_unique_terms():
    idx, val = sparse.query_vector("ram RAM data science")
    assert len(idx) == 3 and set(val) == {1.0}
