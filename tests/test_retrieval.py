"""Unit tests for Layer 5 — in-house KB retrieval (app/retrieval).

OpenAI is mocked here; real retrieval/answer quality is covered by the spike eval
(assistant_api/spikes). These verify pipeline plumbing: contextualize fast-path vs. LLM
rewrite, retriever index/error handling + top-k ranking, answer context-building + image
embedding, the no-context fallback, and that the pipeline threads the contextualized
query through.
"""
import types
from unittest.mock import MagicMock, patch

import numpy as np

from app import retrieval
from app.retrieval import answerer, contextualizer, retriever
from app.retrieval.contextualizer import build_vision_query
from app.retrieval.pipeline import _history_text
from app.retrieval.retriever import Unit


def _chunk(text, document_name="doc.pdf", chunk_id="c1", image=False):
    images = [{"id": chunk_id, "path": f"kb/{chunk_id}.png", "alt": f"alt {chunk_id}"}] if image else []
    return Unit(kind="text", id=chunk_id, document_name=document_name, text=text, images=images)


def _procedure(proc_id="create_task", document_name="tasks.pdf"):
    return Unit(
        kind="procedure", id=proc_id, document_name=document_name, title="Create a task",
        text="Procedure: Create a task\n1. Open Tasks.\n2. Tap +.",
        steps=[
            {"text": "Open Tasks.", "images": [{"id": "i1", "path": "kb/i1.png", "alt": "Tasks menu"}]},
            {"text": "Tap +.", "images": []},
        ],
        images=[{"id": "i1", "path": "kb/i1.png", "alt": "Tasks menu"}],
    )


def _image_unit(image_id="dash", document_name="dash.pdf"):
    return Unit(kind="image", id=image_id, document_name=document_name,
                text="The MCL dashboard home screen with status cards.",
                images=[{"id": image_id, "path": f"kb/{image_id}.png", "alt": "Dashboard home"}])


def _response(content):
    response = MagicMock()
    response.choices[0].message.content = content
    return response


# --- contextualizer ---

def test_contextualize_single_message_unchanged_without_llm():
    with patch("app.retrieval.contextualizer.client") as mock_client:
        out = contextualizer.contextualize(
            "How do I create a checklist?",
            [{"role": "user", "content": "How do I create a checklist?"}],
        )
    assert out == "How do I create a checklist?"
    mock_client.chat.completions.create.assert_not_called()


def test_contextualize_followup_uses_llm_rewrite():
    messages = [
        {"role": "user", "content": "How do I create a checklist?"},
        {"role": "assistant", "content": "Use the Checklist Wizard."},
        {"role": "user", "content": "and how do I delete it?"},
    ]
    with patch("app.retrieval.contextualizer.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response("How do I delete a checklist?")
        out = contextualizer.contextualize("and how do I delete it?", messages)
    assert out == "How do I delete a checklist?"


def test_build_vision_query_returns_model_query():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "ok what do I do here?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
        ]},
    ]
    with patch("app.retrieval.contextualizer.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response("How do I use the Checklist Wizard?")
        out = build_vision_query(messages)
    assert out == "How do I use the Checklist Wizard?"


def test_build_vision_query_empty_when_no_messages():
    with patch("app.retrieval.contextualizer.client") as mock_client:
        assert build_vision_query([]) == ""
    mock_client.chat.completions.create.assert_not_called()


def test_build_vision_query_falls_back_to_empty_on_error():
    messages = [{"role": "user", "content": "x"}]
    with patch("app.retrieval.contextualizer.client") as mock_client:
        mock_client.chat.completions.create.side_effect = RuntimeError("boom")
        assert build_vision_query(messages) == ""


def test_contextualize_llm_error_falls_back_to_original():
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "and it?"},
    ]
    with patch("app.retrieval.contextualizer.client") as mock_client:
        mock_client.chat.completions.create.side_effect = RuntimeError("boom")
        out = contextualizer.contextualize("and it?", messages)
    assert out == "and it?"


# --- retriever ---

def _unit(vector):
    v = np.asarray(vector, dtype=np.float32)
    return v / np.linalg.norm(v)


def _fake_index():
    chunks = [_chunk("about tasks", chunk_id="a"), _chunk("about markets", chunk_id="b"),
              _chunk("about sync", chunk_id="c")]
    embeddings = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])])
    return chunks, embeddings


def _embedding_response(vector):
    response = MagicMock()
    response.data = [MagicMock(embedding=list(vector))]
    return response


def test_retrieve_without_index_returns_empty():
    with patch("app.retrieval.retriever._load_index", return_value=None):
        assert retriever.retrieve("anything") == []


def test_retrieve_ranks_by_cosine_similarity():
    with patch("app.retrieval.retriever._load_index", return_value=_fake_index()), \
         patch("app.retrieval.retriever.client") as mock_client:
        mock_client.embeddings.create.return_value = _embedding_response([0.1, 0.9, 0.05])
        chunks = retriever.retrieve("markets question", top_k=2)
    assert [c.id for c in chunks] == ["b", "a"]   # closest first


def test_retrieve_respects_top_k():
    with patch("app.retrieval.retriever._load_index", return_value=_fake_index()), \
         patch("app.retrieval.retriever.client") as mock_client:
        mock_client.embeddings.create.return_value = _embedding_response([1, 1, 1])
        assert len(retriever.retrieve("q", top_k=1)) == 1


def test_retrieve_embedding_error_returns_empty():
    with patch("app.retrieval.retriever._load_index", return_value=_fake_index()), \
         patch("app.retrieval.retriever.client") as mock_client:
        mock_client.embeddings.create.side_effect = RuntimeError("api down")
        assert retriever.retrieve("q") == []


# --- answerer ---

def test_answer_no_chunks_still_calls_llm_for_language_correct_refusal():
    with patch("app.retrieval.answerer.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response("Ich konnte dazu nichts finden.")
        result = answerer.answer("q", [], language="German")
        system_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert result["answer"] == "Ich konnte dazu nichts finden."
    assert result["sources"] == []
    mock_client.chat.completions.create.assert_called_once()
    assert "GERMAN" in system_prompt


def test_answer_builds_context_and_returns_sources():
    chunks = [_chunk("To create a checklist, tap +.", document_name="wizard.pdf")]
    with patch("app.retrieval.answerer.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response("Tap + [Source: wizard.pdf].")
        result = answerer.answer("how to create", chunks, language="en")
        user_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert result["answer"] == "Tap + [Source: wizard.pdf]."
    assert result["sources"] == ["wizard.pdf"]
    assert "# TEXTUAL CONTEXT" in user_prompt
    assert "To create a checklist, tap +." in user_prompt


def test_prompt_exposes_procedure_steps_with_markers_and_no_urls():
    with patch("app.retrieval.answerer.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response("ok")
        answerer.answer("how to create a task", [_procedure(), _image_unit()])
        user_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "# PROCEDURES" in user_prompt
    assert "1. Open Tasks.  {{step:create_task.1}}" in user_prompt   # step with a screenshot gets a marker
    assert "2. Tap +." in user_prompt and "{{step:create_task.2}}" not in user_prompt  # no image → no marker
    assert "{{image:dash}}" in user_prompt                          # standalone screenshot listed by marker
    assert "/images/" not in user_prompt                            # the model never sees a URL


def test_render_markers_places_step_images_deterministically():
    with patch("app.retrieval.answerer.API_PUBLIC_URL", "http://host"):
        out, urls = answerer.render_markers(
            "1. Abre Tareas. {{step:create_task.1}}\n2. Pulsa +. {{step:create_task.2}}", [_procedure()])
    assert "![Tasks menu](http://host/images/kb/i1.png)" in out
    assert urls == ["http://host/images/kb/i1.png"]
    assert "{{" not in out                                          # unknown/empty markers stripped


def test_render_markers_resolves_standalone_and_strips_invented():
    with patch("app.retrieval.answerer.API_PUBLIC_URL", "http://host"):
        out, urls = answerer.render_markers(
            "Here it is: {{image:dash}} and {{image:made_up}} {{step:nope.9}}", [_image_unit()])
    assert "![Dashboard home](http://host/images/kb/dash.png)" in out
    assert "made_up" not in out and "nope" not in out
    assert urls == ["http://host/images/kb/dash.png"]


def test_answer_keeps_rendered_images_through_enforcement():
    with patch("app.retrieval.answerer.client") as mock_client, \
         patch("app.retrieval.answerer.API_PUBLIC_URL", "http://host"):
        mock_client.chat.completions.create.return_value = _response(
            "1. Open Tasks. {{step:create_task.1}} [Source: tasks.pdf]")
        result = answerer.answer("how", [_procedure()])
    assert "![Tasks menu](http://host/images/kb/i1.png)" in result["answer"]   # not stripped as unverified


def test_answer_no_visual_block_when_no_images():
    chunks = [_chunk("text only")]
    with patch("app.retrieval.answerer.client") as mock_client:
        mock_client.chat.completions.create.return_value = _response("ok")
        answerer.answer("q", chunks)
        user_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "# AVAILABLE VISUAL AIDS" not in user_prompt


# --- pipeline ---

def test_history_text_excludes_latest_and_formats_turns():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "latest"},
    ]
    history = _history_text(messages)
    assert "User: first" in history
    assert "Assistant: answer" in history
    assert "latest" not in history


def test_run_threads_contextualized_query(monkeypatch):
    captured = {}

    def fake_retrieve(query):
        captured["retrieved"] = query
        return [_chunk("ctx")]

    def fake_answer(query, chunks, *, language=None, device=None, history_text="", memory=None):
        captured["answered"] = query
        return {"answer": "A", "sources": ["doc.pdf"]}

    monkeypatch.setattr("app.retrieval.pipeline.contextualize", lambda q, m: "standalone query")
    monkeypatch.setattr("app.retrieval.pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("app.retrieval.pipeline.answer", fake_answer)

    result = retrieval.run("raw", [{"role": "user", "content": "raw"}], language="en")
    assert result["answer"] == "A"
    assert captured["retrieved"] == "standalone query"
    assert captured["answered"] == "standalone query"


def test_public_api_surface():
    for name in ("answer", "contextualize", "retrieve", "run"):
        assert hasattr(retrieval, name)
