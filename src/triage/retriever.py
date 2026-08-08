"""KB matching for Task 1 — wraps kb_index.py's chunker/BM25 retriever and converts
its best hit into the KBMatch shape the triage schema needs.

BM25 alone is not enough here: the 5 product docs reuse a lot of the same generic
support vocabulary ("dashboard loading times unacceptable", "not working after recent
update", ...), so plain top-1 BM25 regularly cites the WRONG product's doc — e.g. an
AnalyticsHub ticket citing cloudsync.md purely on word overlap. That's not the doc-path
hallucination PRD §2.3 warns about (the doc is real), but it's the same trust problem
in practice: an agent should not cite unrelated material. Fix: 437/500 real tickets
literally name their product in the subject/body, and whenever they do it's always the
correct product (verified against tickets.json) — so when a product is named, retrieval
is scoped to that product's doc plus the cross-product docs (troubleshooting/, billing/,
onboarding/), excluding the other four products' docs entirely. Falls back to
unscoped top-1 BM25 for the ~13% of tickets that don't name a product.
"""

from __future__ import annotations

from functools import lru_cache

from src.kb_index import KBChunk, build_retriever
from src.triage.schema import KBMatch

_SNIPPET_MAX_CHARS = 400
_SCOPED_SEARCH_TOP_K = 5

_PRODUCT_DOC_SLUGS = {
    "DataBridge Pro": "databridge-pro",
    "CloudSync": "cloudsync",
    "AnalyticsHub": "analyticshub",
    "SecureVault": "securevault",
    "WorkflowEngine": "workflowengine",
}


@lru_cache(maxsize=1)
def _get_retriever():
    """Build the BM25 retriever once per process — the KB docs don't change at runtime."""
    return build_retriever()


def _snippet(chunk: KBChunk, max_chars: int = _SNIPPET_MAX_CHARS) -> str:
    text = chunk.text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _excluded_product_docs(query_text: str) -> set[str]:
    """doc_paths of OTHER products' docs to exclude, if query_text literally names one
    product. Empty set (no exclusion) if zero or more than one product is named.
    """
    lowered = query_text.lower()
    named = [slug for name, slug in _PRODUCT_DOC_SLUGS.items() if name.lower() in lowered]
    if len(named) != 1:
        return set()
    return {f"products/{slug}.md" for slug in _PRODUCT_DOC_SLUGS.values() if slug != named[0]}


def retrieve_kb_match(query_text: str) -> KBMatch | None:
    """Best KB chunk for query_text, or None if nothing scores above zero. Never
    fabricates a doc when there's no good match (PRD §2.3) — doc_path/heading/snippet
    always trace back to a real chunk from kb_index.load_kb_chunks().
    """
    excluded = _excluded_product_docs(query_text)
    results = _get_retriever().search(query_text, top_k=_SCOPED_SEARCH_TOP_K if excluded else 1)
    results = [(chunk, score) for chunk, score in results if chunk.doc_path not in excluded]
    if not results:
        return None
    chunk, _score = results[0]
    heading = chunk.headings[-1][1] if chunk.headings else chunk.doc_title
    return KBMatch(doc_path=chunk.doc_path, heading=heading, snippet=_snippet(chunk))
