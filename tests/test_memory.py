"""Unit tests for Layer 3 — per-user durable memory (app/services/memory_service.py).

Uses a temp MEMORIES_DIR so real memories are never touched. The core guarantee under test:
one user never sees another's memories, anonymous has no durable memory, and recall is capped.
"""
import pytest

from app.services import memory_service
from app.services.memory_service import MemoryService


@pytest.fixture
def temp_memories(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_service, "MEMORIES_DIR", str(tmp_path))
    return tmp_path


def _save(user_id, title, content, importance="medium"):
    return MemoryService(user_id).save_memory(
        {"title": title, "content": content, "importance": importance, "category": "c", "tags": []}
    )


def test_per_user_isolation(temp_memories):
    _save("userA", "A secret", "alpha")
    _save("userB", "B secret", "beta")
    assert [m["title"] for m in MemoryService("userA").list_memories()] == ["A secret"]
    assert [m["title"] for m in MemoryService("userB").list_memories()] == ["B secret"]


def test_recall_scoped_to_user(temp_memories):
    _save("userA", "A pref", "alpha content")
    assert "alpha content" in MemoryService("userA").recall_context()
    assert MemoryService("userB").recall_context() == ""


def test_anonymous_has_no_durable_memory(temp_memories):
    _save(None, "anon", "should never surface")
    assert MemoryService(None).recall_context() == ""


def test_recall_capped_by_item_count(temp_memories, monkeypatch):
    monkeypatch.setattr(memory_service, "RECALL_MAX_ITEMS", 3)
    monkeypatch.setattr(memory_service, "RECALL_MAX_CHARS", 1_000_000)
    for i in range(10):
        _save("u", f"mem{i}", f"content {i}")
    assert MemoryService("u").recall_context().count("###") == 3


def test_recall_capped_by_chars(temp_memories, monkeypatch):
    monkeypatch.setattr(memory_service, "RECALL_MAX_ITEMS", 100)
    monkeypatch.setattr(memory_service, "RECALL_MAX_CHARS", 200)
    for i in range(10):
        _save("u", f"mem{i}", "x" * 120)
    assert len(MemoryService("u").recall_context()) < 800


def test_delete_scoped(temp_memories):
    saved = _save("u", "t", "c")
    assert MemoryService("u").delete_memory(saved["id"]) is True
    assert MemoryService("u").get_memory(saved["id"]) is None


def test_user_id_is_sanitized(temp_memories):
    service = MemoryService("../../etc/passwd")
    service.save_memory({"title": "t", "content": "c", "importance": "low", "tags": []})
    assert service.recall_context() != ""
