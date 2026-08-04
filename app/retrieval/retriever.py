"""Layer 5 — Ragie retrieval (single-shot, reranked)."""
import os
import time
from typing import Any, List

from app.core.config import RAGIE_API_KEY, RAGIE_PARTITION, RAGIE_TOP_K
from app.core.logging import get_logger

logger = get_logger(__name__)

# Reranked retrieval can be slow on a cold call; give it a generous read timeout + retries.
RETRIEVE_TIMEOUT_MS = int(os.getenv("RAGIE_TIMEOUT_MS", "30000"))
RETRIEVE_RETRIES = int(os.getenv("RAGIE_RETRIES", "2"))
RETRY_BACKOFF_S = float(os.getenv("RAGIE_RETRY_BACKOFF_S", "0.5"))

_client = None


def _ragie():
    global _client
    if _client is None:
        from ragie import Ragie
        _client = Ragie(auth=RAGIE_API_KEY)
    return _client


def _reset_client() -> None:
    # Drop the singleton so the next call builds a fresh connection pool — a reused keep-alive
    # socket dropped by the host (SSL EOF / connection reset, common on Azure) stays poisoned otherwise.
    global _client
    _client = None


def _retrieve(query: str, top_k: int, rerank: bool) -> List[Any]:
    request = {"query": query, "rerank": rerank, "top_k": top_k, "partition": RAGIE_PARTITION}
    for attempt in range(RETRIEVE_RETRIES + 1):
        try:
            result = _ragie().retrievals.retrieve(request=request, timeout_ms=RETRIEVE_TIMEOUT_MS)
            return list(result.scored_chunks)
        except Exception as e:
            last = attempt == RETRIEVE_RETRIES
            level = logger.error if last else logger.warning
            level(f"[RETRIEVAL] Ragie retrieve {'failed' if last else 'retrying'} "
                  f"(attempt {attempt + 1}/{RETRIEVE_RETRIES + 1}): {e}")
            _reset_client()
            if not last and RETRY_BACKOFF_S:
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    return []


def retrieve(query: str, *, top_k: int = RAGIE_TOP_K) -> List[Any]:
    if not RAGIE_API_KEY:
        logger.warning("[RETRIEVAL] RAGIE_API_KEY not set — returning no chunks")
        return []

    chunks = _retrieve(query, top_k, rerank=True)
    if not chunks:
        # Ragie's reranker can reject every candidate for conversationally-phrased queries
        # ("can you show me an image of…"); fall back to the raw top-k so we still ground.
        logger.info("[RETRIEVAL] rerank returned 0 chunks — retrying without rerank")
        chunks = _retrieve(query, top_k, rerank=False)
    return chunks
