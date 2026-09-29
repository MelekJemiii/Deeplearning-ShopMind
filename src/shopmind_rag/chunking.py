"""Two chunking strategies over cleaned Markdown.

A document is first parsed into BLOCKS: headings, paragraphs, and tables (consecutive
'|' lines kept together). Both strategies work on blocks, so a table row is never cut.

Strategy A (recursive, structure-blind): pack blocks into chunks up to max_tokens,
    splitting oversized blocks by sentences then words; overlap carried between chunks.
Strategy B (headings, structure-aware): one chunk per section, text prefixed with the
    heading path (e.g. "Legion 5 > PERFORMANCE > Graphics"); small sections merged,
    oversized sections sub-split with the same packer as A.
"""
import re
from dataclasses import dataclass, field
from functools import lru_cache

import tiktoken

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9\"(])")


@lru_cache(maxsize=1)
def _enc(name: str):
    return tiktoken.get_encoding(name)


class Tokenizer:
    def __init__(self, name: str):
        self.enc = _enc(name)

    def count(self, text: str) -> int:
        return len(self.enc.encode(text, disallowed_special=()))


@dataclass
class Block:
    kind: str          # heading | para | table
    text: str
    level: int = 0     # heading level


@dataclass
class Chunk:
    text: str
    heading_path: list[str] = field(default_factory=list)


def clean_heading(text: str) -> str:
    return re.sub(r"[*_`]", "", text).strip()


def parse_blocks(md: str) -> list[Block]:
    blocks, para, table = [], [], []

    def flush():
        if para:
            blocks.append(Block("para", " ".join(para).strip())); para.clear()
        if table:
            blocks.append(Block("table", "\n".join(table))); table.clear()

    for line in md.split("\n"):
        s = line.strip()
        m = HEADING.match(s)
        if m:
            flush(); blocks.append(Block("heading", s, len(m.group(1))))
        elif s.startswith("|"):
            if para:
                flush()
            table.append(s)
        elif not s:
            flush()
        else:
            if table:
                flush()
            para.append(s)
    flush()
    return [b for b in blocks if b.text]


# ---------- splitting oversized pieces ----------

def _split_words(text: str, max_t: int, tok: Tokenizer) -> list[str]:
    out, cur = [], []
    for w in text.split():
        if cur and tok.count(" ".join(cur + [w])) > max_t:
            out.append(" ".join(cur)); cur = []
        cur.append(w)
    if cur:
        out.append(" ".join(cur))
    return out


def _split_text(text: str, max_t: int, tok: Tokenizer) -> list[str]:
    """Sentences first, words as last resort."""
    if tok.count(text) <= max_t:
        return [text]
    out, cur = [], ""
    for s in SENTENCE.split(text):
        if tok.count(s) > max_t:
            if cur:
                out.append(cur); cur = ""
            out.extend(_split_words(s, max_t, tok)); continue
        cand = f"{cur} {s}".strip()
        if cur and tok.count(cand) > max_t:
            out.append(cur); cur = s
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def _split_table(table: str, max_t: int, tok: Tokenizer) -> list[str]:
    """Split by rows, repeating the header row + separator in each piece."""
    rows = table.split("\n")
    head = rows[:2] if len(rows) > 1 and re.fullmatch(r"\|[\s:|-]+\|?", rows[1]) else rows[:1]
    body = rows[len(head):]
    out, cur = [], list(head)
    for r in body:
        if len(cur) > len(head) and tok.count("\n".join(cur + [r])) > max_t:
            out.append("\n".join(cur)); cur = list(head)
        cur.append(r)
    out.append("\n".join(cur))
    return [p for piece in out for p in (_split_words(piece, max_t, tok) if tok.count(piece) > max_t else [piece])]


def _units(blocks: list[Block], max_t: int, tok: Tokenizer) -> list[str]:
    units = []
    for b in blocks:
        if tok.count(b.text) <= max_t:
            units.append(b.text)
        elif b.kind == "table":
            units.extend(_split_table(b.text, max_t, tok))
        else:
            units.extend(_split_text(b.text, max_t, tok))
    return units


def _tail(text: str, n_tokens: int, tok: Tokenizer) -> str:
    if n_tokens <= 0 or text.lstrip().startswith("|"):
        return ""  # no overlap out of a table: a half table would be misleading
    words, out = text.split(), []
    while words and tok.count(" ".join(out)) < n_tokens:
        out.insert(0, words.pop())
    return " ".join(out)


def pack(blocks: list[Block], max_t: int, overlap: int, tok: Tokenizer) -> list[str]:
    """Greedy packing of units into chunks <= max_t, with word-level overlap."""
    chunks, cur = [], []
    for u in _units(blocks, max_t, tok):
        cand = "\n\n".join(cur + [u])
        if cur and tok.count(cand) > max_t:
            chunks.append("\n\n".join(cur))
            t = _tail(chunks[-1], overlap, tok)
            cur = [t, u] if t and tok.count(f"{t}\n\n{u}") <= max_t else [u]
        else:
            cur.append(u)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


# ---------- strategies ----------

def chunk_recursive(md: str, cfg: dict, tok: Tokenizer) -> list[Chunk]:
    blocks = [Block("para", b.text) if b.kind == "heading" else b for b in parse_blocks(md)]
    return [Chunk(t) for t in pack(blocks, cfg["max_tokens"], cfg["overlap_tokens"], tok)]


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    out = []
    for x, y in zip(a, b):
        if x != y:
            break
        out.append(x)
    return out


def chunk_headings(md: str, cfg: dict, tok: Tokenizer, doc_title: str) -> list[Chunk]:
    max_t, min_t, ovl = cfg["max_tokens"], cfg["min_tokens"], cfg["overlap_tokens"]
    sections, stack, cur = [], [], None  # stack: [(level, title)]
    for b in parse_blocks(md):
        if b.kind == "heading":
            level, title = b.level, clean_heading(HEADING.match(b.text).group(2))
            if level == 1 and not sections and cur is None and not stack:
                continue  # document title line: already in doc_title
            stack = [s for s in stack if s[0] < level] + [(level, title)]
            cur = {"path": [doc_title] + [t for _, t in stack], "blocks": [b]}
            sections.append(cur)
        else:
            if cur is None:
                cur = {"path": [doc_title], "blocks": []}; sections.append(cur)
            cur["blocks"].append(b)

    # merge small sections forward (keeping their heading lines inside the text)
    merged, buf = [], None
    for s in sections:
        if buf is None:
            buf = {"path": s["path"], "blocks": list(s["blocks"])}
        elif tok.count("\n\n".join(x.text for x in buf["blocks"] + s["blocks"])) <= max_t and \
                tok.count("\n\n".join(x.text for x in buf["blocks"])) < min_t:
            buf["blocks"].extend(s["blocks"])
            buf["path"] = _common_prefix(buf["path"], s["path"])  # path must be true for ALL merged sections
        else:
            merged.append(buf); buf = {"path": s["path"], "blocks": list(s["blocks"])}
    if buf:
        merged.append(buf)

    chunks = []
    for s in merged:
        prefix = " > ".join(s["path"])
        budget = max_t - tok.count(prefix) - 2
        content = [b for b in s["blocks"] if b.kind != "heading" or len(s["blocks"]) > 1]
        for piece in pack(content, budget, ovl, tok):
            chunks.append(Chunk(f"{prefix}\n\n{piece}", s["path"]))
    return chunks
