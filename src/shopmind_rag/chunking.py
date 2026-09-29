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
    prefix: str = ""   # heading lines glued to this block (strategy B)


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
        room = max_t - (tok.count(b.prefix) + 2 if b.prefix else 0)
        if tok.count(b.text) <= room:
            parts = [b.text]
        elif b.kind == "table":
            parts = _split_table(b.text, room, tok)
        else:
            parts = _split_text(b.text, room, tok)
        if b.prefix:
            parts[0] = f"{b.prefix}\n\n{parts[0]}"  # headings travel with the first piece of their content
        units.extend(parts)
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


def _glue_headings(blocks: list[Block]) -> list[Block]:
    """Attach heading lines to the block that follows them, so a chunk never ends on a bare heading."""
    out, heads = [], []
    for b in blocks:
        if b.kind == "heading":
            heads.append(b.text)
        else:
            out.append(Block(b.kind, b.text, prefix="\n\n".join(heads)) if heads else b)
            heads = []
    if heads and out:
        out[-1] = Block(out[-1].kind, out[-1].text, prefix=out[-1].prefix)  # trailing headings: no content, dropped
    return out


def chunk_headings(md: str, cfg: dict, tok: Tokenizer, doc_title: str) -> list[Chunk]:
    max_t, min_t, ovl = cfg["max_tokens"], cfg["min_tokens"], cfg["overlap_tokens"]

    # 1) sections: heading path + blocks
    sections, stack, cur = [], [], None  # stack: [(level, title)]
    for b in parse_blocks(md):
        if b.kind == "heading":
            level, title = b.level, clean_heading(HEADING.match(b.text).group(2))
            if level == 1 and not sections and not stack:
                continue  # document title line: already in doc_title
            stack = [s for s in stack if s[0] < level] + [(level, title)]
            cur = {"path": [doc_title] + [t for _, t in stack], "blocks": [b]}
            sections.append(cur)
        else:
            if cur is None:
                cur = {"path": [doc_title], "blocks": []}; sections.append(cur)
            cur["blocks"].append(b)

    # 2) heading-only sections (parent whose content is all in sub-sections) carried into the next one
    folded, pending = [], None
    for s in sections:
        if all(b.kind == "heading" for b in s["blocks"]):
            pending = pending or {"path": s["path"], "blocks": []}
            pending["blocks"].extend(s["blocks"]); continue
        if pending:
            s = {"path": _common_prefix(pending["path"], s["path"]), "blocks": pending["blocks"] + s["blocks"]}
            pending = None
        folded.append(s)

    # 3) pack each section into pieces (path, body) within the budget left by its prefix
    pieces = []
    for s in folded:
        budget = max_t - tok.count(" > ".join(s["path"])) - 2
        for body in pack(_glue_headings(s["blocks"]), budget, ovl, tok):
            pieces.append({"path": s["path"], "body": body})

    # 4) merge small pieces with a neighbour (backward first, then forward) while the result fits
    render = lambda p: f"{' > '.join(p['path'])}\n\n{p['body']}"  # noqa: E731

    def join(a, b):
        return {"path": _common_prefix(a["path"], b["path"]), "body": f"{a['body']}\n\n{b['body']}"}

    merged = []
    for p in pieces:
        if merged and (tok.count(render(merged[-1])) < min_t or tok.count(render(p)) < min_t) \
                and tok.count(render(join(merged[-1], p))) <= max_t:
            merged[-1] = join(merged[-1], p)  # common path: must be true for everything inside
        else:
            merged.append(p)
    return [Chunk(render(p), p["path"]) for p in merged]
