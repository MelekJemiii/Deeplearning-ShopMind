"""BM25 sparse vectors for hybrid search (keyword side).

Dense embeddings capture meaning but are weak on exact tokens: model names ("15IAX11"),
spec codes ("IP68", "115W"). BM25 matches those exactly.
Term-frequency weights (BM25 saturation + length normalization) are computed here;
the IDF part is applied by Qdrant (sparse vector modifier IDF), so it stays correct
as the collection changes. Token ids are stable hashes: no vocabulary file to maintain.
"""
import hashlib
import re
import unicodedata
from collections import Counter

from .config import SETTINGS

CFG = SETTINGS["hybrid"]
STOPWORDS = set("""
a an and are as at be but by for from has have how i in is it its of on or that the this to was what
when where which who why will with you your can does do my me we our
au aux avec ce ces c cela comment d dans de des du elle en est et être il ils je l la le les leur lui
ma mais me mes mon ne nos notre nous on ou où par pas pour qu que quel quelle quels quelles qui sa
se ses son sur ta te tes ton tu un une vos votre vous y est-ce faut peut combien
""".split())


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))  # "écran" -> "ecran"
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in STOPWORDS and len(t) > 1]


def token_id(token: str) -> int:
    return int(hashlib.md5(token.encode()).hexdigest()[:8], 16)  # stable 32-bit id


def doc_vector(text: str, avgdl: float) -> tuple[list[int], list[float]]:
    tf = Counter(tokenize(text))
    dl, k1, b = sum(tf.values()), CFG["k1"], CFG["b"]
    idx, val = [], []
    for tok, f in tf.items():
        idx.append(token_id(tok))
        val.append(f * (k1 + 1) / (f + k1 * (1 - b + b * dl / max(avgdl, 1))))
    return idx, val


def query_vector(text: str) -> tuple[list[int], list[float]]:
    toks = sorted(set(tokenize(text)))
    return [token_id(t) for t in toks], [1.0] * len(toks)


def avg_doc_len(texts: list[str]) -> float:
    return sum(len(tokenize(t)) for t in texts) / max(len(texts), 1)
