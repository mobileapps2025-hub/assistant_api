"""Layer 5 — Ragie retrieval (single-shot, reranked)."""
from typing import Any, List

from app.core.config import RAGIE_API_KEY, RAGIE_PARTITION, RAGIE_TOP_K
from app.core.logging import get_logger

logger = get_logger(__name__)

_client = None


def _ragie():
    global _client
    if _client is None:
        from ragie import Ragie
        _client = Ragie(auth=RAGIE_API_KEY)
    return _client


def retrieve(query: str, *, top_k: int = RAGIE_TOP_K) -> List[Any]:
    if not RAGIE_API_KEY:
        logger.warning("[RETRIEVAL] RAGIE_API_KEY not set — returning no chunks")
        return []
    try:
        result = _ragie().retrievals.retrieve(request={
            "query": query,
            "rerank": True,
            "top_k": top_k,
            "partition": RAGIE_PARTITION,
        })
        return list(result.scored_chunks)
    except Exception as e:
        logger.error(f"[RETRIEVAL] Ragie retrieve failed: {e}")
        return []
