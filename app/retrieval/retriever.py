"""Layer 5 — in-house KB retrieval (replaces Ragie).

Reads the committed index built by ingest_kb.py (app/kb_index/). Units are typed:
  text       a document chunk (may reference approved screenshots)
  procedure  a whole step-by-step procedure; each step carries its approved screenshots
  image      one approved screenshot, retrievable by its semantic description
A query is one OpenAI embedding call + cosine top-k — no external retrieval service.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.config import KB_INDEX_DIR, KB_TOP_K, client
from app.core.logging import get_logger
from app.kb_capabilities import EVERYONE, visible
from app.kb_surfaces import SEARCHABLE_BY_DEFAULT

logger = get_logger(__name__)

EMBED_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class Unit:
    kind: str                       # text | procedure | image
    id: str
    document_name: str
    text: str
    title: str = ""
    surface: str = "both"           # app | web | both — which product the unit describes
    capability: str = EVERYONE      # which permission the unit needs; everyone = no restriction
    images: List[Dict[str, Any]] = field(default_factory=list)   # [{id, path, alt}]
    steps: List[Dict[str, Any]] = field(default_factory=list)    # procedure: [{text, images}]


_index: Optional[tuple[List[Unit], np.ndarray]] = None


_index_mtime: Optional[float] = None


def _load_index() -> Optional[tuple[List[Unit], np.ndarray]]:
    """Load the committed index, reloading automatically when the file changes — so a page
    approved/published on this machine is used on the next question, no restart needed."""
    global _index, _index_mtime
    index_dir = Path(KB_INDEX_DIR)
    index_file, embeddings_file = index_dir / "index.json", index_dir / "embeddings.npz"
    if not index_file.exists() or not embeddings_file.exists():
        logger.error(f"[RETRIEVAL] KB index missing under {index_dir} — run ingest_kb.py")
        return None
    mtime = index_file.stat().st_mtime
    if _index is None or mtime != _index_mtime:
        data = json.loads(index_file.read_text(encoding="utf-8"))
        units = [Unit(**{k: v for k, v in u.items() if k in Unit.__dataclass_fields__}) for u in data["units"]]
        embeddings = np.load(embeddings_file)["embeddings"]
        logger.info(f"[RETRIEVAL] KB index loaded: {len(units)} units")
        _index = (units, embeddings)
        _index_mtime = mtime
    return _index


def _embed_query(query: str) -> Optional[np.ndarray]:
    try:
        response = client.embeddings.create(model=EMBED_MODEL, input=[query], timeout=15)
        vector = np.asarray(response.data[0].embedding, dtype=np.float32)
        return vector / np.linalg.norm(vector)
    except Exception as e:
        logger.error(f"[RETRIEVAL] query embedding failed: {e}")
        return None


def retrieve(query: str, *, top_k: int = KB_TOP_K,
             surfaces: frozenset = SEARCHABLE_BY_DEFAULT, min_score: float = 0.0,
             capabilities: frozenset = frozenset()) -> List[Unit]:
    loaded = _load_index()
    if loaded is None:
        return []
    query_vector = _embed_query(query)
    if query_vector is None:
        return []

    units, embeddings = loaded
    scores = embeddings @ query_vector
    allowed = np.array([u.surface in surfaces and visible(u.capability, capabilities) for u in units])
    scores = np.where(allowed, scores, -np.inf)
    top = [i for i in np.argsort(scores)[::-1][:top_k] if np.isfinite(scores[i])]
    # min_score gates whether a relevant page exists at all. Once the best match clears it, hand
    # over the whole top set (a matching page's own steps can score below the bar) rather than
    # filtering unit by unit and starving the answer.
    if not top or scores[top[0]] < min_score:
        top = []
    logger.info("[RETRIEVAL] surfaces=%s caps=%s min=%.2f top-%d: %s", sorted(surfaces), sorted(capabilities),
                min_score, top_k, ", ".join(f"{units[i].kind}:{units[i].id}={scores[i]:.3f}" for i in top))
    return [units[i] for i in top]
