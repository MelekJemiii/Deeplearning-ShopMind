"""Text normalization shared by test-set validation and retrieval evaluation.

Converted Markdown contains formatting (**bold**, <br>, table pipes, <sup>) that would
make exact string matching fail even when the content is there. Both the evidence
phrases and the chunk text go through the same normalize(), so matching is fair.
"""
import re
import unicodedata

_MD = re.compile(r"<br\s*/?>|</?sup>|</?u>|[*_`|#]")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u00a0", " ")
    text = _MD.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def contains_evidence(text: str, evidence: list[str]) -> bool:
    """True if any evidence phrase appears in the text (after normalization)."""
    norm = normalize(text)
    return any(normalize(e) in norm for e in evidence)
