"""Chunk knowledge_base/**/*.md and build a BM25 retriever over the chunks.

Chunking follows DATA_SCHEMA.md's guidance: split each doc on '---' horizontal rules
(major section boundaries). Each chunk keeps the heading hierarchy that applies to it
as metadata — the doc's most recent '#' title plus every heading found inside the
chunk itself — so retrieval results can cite exactly where they came from.

Retrieval is BM25 (rank_bm25) over the chunk text — no vector DB, no external
embeddings service, per PRD §8.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

KB_DIR = Path(__file__).resolve().parent.parent / "knowledge_base"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass
class KBChunk:
    doc_path: str  # relative to knowledge_base/, e.g. "troubleshooting/authentication-sso.md"
    doc_title: str  # most recent '#' heading in effect for this chunk
    headings: list[tuple[int, str]]  # (level, text) for every heading inside the chunk, in order
    text: str  # verbatim chunk text (whitespace-trimmed substring of the source file)


def load_kb_chunks(kb_dir: Path = KB_DIR) -> list[KBChunk]:
    """Read every *.md file under kb_dir and split each on '---' rules into KBChunk
    records. A doc's title tracks the most recently seen '#' heading, so a file that
    contains more than one top-level title (e.g. performance-and-integrations.md,
    which covers two distinct troubleshooting topics) attributes each chunk to the
    title actually in effect at that point in the doc, not just the first one.
    """
    chunks: list[KBChunk] = []
    for md_path in sorted(kb_dir.rglob("*.md")):
        raw = md_path.read_text(encoding="utf-8")
        rel_path = md_path.relative_to(kb_dir).as_posix()
        doc_title = ""
        for section in re.split(r"\n-{3,}\n", raw):
            text = section.strip()
            if not text:
                continue
            headings = [(len(hashes), h.strip()) for hashes, h in _HEADING_RE.findall(section)]
            h1s = [h for level, h in headings if level == 1]
            if h1s:
                doc_title = h1s[-1]
            chunks.append(KBChunk(doc_path=rel_path, doc_title=doc_title, headings=headings, text=text))
    return chunks


class KBRetriever:
    """BM25 retriever over KB chunks."""

    def __init__(self, chunks: list[KBChunk]):
        if not chunks:
            raise ValueError("KBRetriever requires at least one chunk")
        self.chunks = chunks
        self._bm25 = BM25Okapi([self._tokenize(c.text) for c in chunks])

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def search(self, query: str, top_k: int = 3) -> list[tuple[KBChunk, float]]:
        """Up to top_k (chunk, score) pairs, best first. [] if the query has no
        positive-scoring match — callers must treat that as "no good match" and
        fall back to null, never hallucinate a doc.
        """
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [(chunk, score) for chunk, score in ranked[:top_k] if score > 0]


def build_retriever(kb_dir: Path = KB_DIR) -> KBRetriever:
    """Load every KB doc under kb_dir, chunk it, and build a ready-to-query KBRetriever."""
    return KBRetriever(load_kb_chunks(kb_dir))
