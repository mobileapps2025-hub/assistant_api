"""Unit tests for Layer 5 — Ragie retrieval (app/retrieval).

Ragie and OpenAI are mocked here; real retrieval/answer quality is covered by the spike eval
(assistant_api/spikes). These verify pipeline plumbing: contextualize fast-path vs. LLM
rewrite, retriever key/error handling, answer context-building + image embedding, the
no-context fallback, and that the pipeline threads the contextualized query through.
"""
import types
from unittest.mock import MagicMock, patch

from app import retrieval
from app.retrieval import answerer, contextualizer, retriever
from app.retrieval.contextualizer import build_vision_query
from app.retrieval.pipeline import _history_text


def _chunk(text, document_name="doc.pdf", document_id="d1", chunk_id="c1", image=False):
    return types.SimpleNamespace(
        text=text,
        document_name=document_name,
        document_id=document_id,
        id=chunk_id,
        links={"self_image": object()} if image else {},
    )


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

def test_retrieve_without_key_returns_empty():
    with patch("app.retrieval.retriever.RAGIE_API_KEY", ""):
        assert retriever.retrieve("anything") == []


def test_retrieve_returns_scored_chunks():
    fake = MagicMock()
    fake.retrievals.retrieve.return_value.scored_chunks = [_chunk("hi")]
    with patch("app.retrieval.retriever.RAGIE_API_KEY", "key"), \
         patch("app.retrieval.retriever._ragie", return_value=fake):
        chunks = retriever.retrieve("q")
    assert len(chunks) == 1 and chunks[0].text == "hi"


def test_retrieve_error_returns_empty():
    fake = MagicMock()
    fake.retrievals.retrieve.side_effect = RuntimeError("boom")
    with patch("app.retrieval.retriever.RAGIE_API_KEY", "key"), \
         patch("app.retrieval.retriever._ragie", return_value=fake):
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


def test_answer_embeds_image_proxy_url_for_image_chunks():
    chunks = [_chunk("photo screen", document_name="photo.pdf", document_id="D", chunk_id="C", image=True)]
    with patch("app.retrieval.answerer.client") as mock_client, \
         patch("app.retrieval.answerer.API_PUBLIC_URL", "http://host"):
        mock_client.chat.completions.create.return_value = _response("see it")
        answerer.answer("photo", chunks)
        user_prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert "# AVAILABLE VISUAL AIDS" in user_prompt
    assert "http://host/api/ragie/image?document_id=D&chunk_id=C" in user_prompt


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
