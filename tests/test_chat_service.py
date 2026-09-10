import json
import re
import types
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.chat_service import ChatService, _format_device, _format_today
from app.enforcement import pending
from app.kb_surfaces import APP_SEARCH
from app.models import Device
from app.retrieval.retriever import Unit


def test_format_device_combines_present_fields():
    assert _format_device(Device(platform="iOS", form_factor="phone", app_version="8.2.1")) == "iOS phone (app v8.2.1)"
    assert _format_device(Device(platform="Android")) == "Android"
    assert _format_device(Device(form_factor="tablet")) == "tablet"
    assert _format_device(None) == ""
    assert _format_device(Device()) == ""


def test_format_today_formats_valid_zone():
    out = _format_today("Europe/Berlin")
    assert re.match(r"^[A-Z][a-z]+day, \d{4}-\d{2}-\d{2}$", out)


def test_format_today_returns_none_for_missing_or_bad_zone():
    assert _format_today(None) is None
    assert _format_today("") is None
    assert _format_today("Not/AZone") is None


def make_service(mock_vision_service, mock_image_validator):
    """Construct a ChatService with all external dependencies mocked."""
    return ChatService(mock_vision_service, mock_image_validator)


def _tool_response(name, args_json):
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = name
    tc.function.arguments = args_json
    resp = MagicMock()
    resp.choices[0].message.tool_calls = [tc]
    resp.choices[0].message.content = None
    return resp


def _text_response(text):
    resp = MagicMock()
    resp.choices[0].message.tool_calls = None
    resp.choices[0].message.content = text
    return resp


# ---------------------------------------------------------------------------
# process_chat_request preflight + dispatch
# ---------------------------------------------------------------------------

class TestProcessChatRequestRouting:
    """Verify preflight labels still preserve image handling and manager dispatch."""

    @pytest.mark.asyncio
    async def test_image_message_goes_through_router_not_forked(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        service._handle_text_request = AsyncMock(
            return_value={"response": "screen help", "success": True, "has_vision": True}
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]
        with patch("app.services.chat_service.classify_route") as mock_route:
            mock_route.return_value.route = "KNOWLEDGE"
            mock_route.return_value.reason = "screen help"
            result = await service.process_chat_request(messages)

        mock_route.assert_called_once()  # the image went THROUGH preflight, not around it
        service._handle_text_request.assert_called_once()
        assert result["has_vision"] is True

    @pytest.mark.asyncio
    async def test_routes_to_text_when_no_image(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        service._handle_vision_request = AsyncMock()
        service._handle_text_request = AsyncMock(
            return_value={"response": "old text answer", "success": True, "has_vision": False}
        )
        service._handle_agent_request = AsyncMock(
            return_value={"response": "manager answer", "success": True, "has_vision": False}
        )

        messages = [{"role": "user", "content": "How do I sync?"}]
        with patch("app.services.chat_service.classify_route") as mock_route:
            mock_route.return_value.route = "KNOWLEDGE"
            mock_route.return_value.reason = "general how-to"
            result = await service.process_chat_request(messages)

        service._handle_agent_request.assert_called_once()
        service._handle_text_request.assert_not_called()
        service._handle_vision_request.assert_not_called()
        assert result["response"] == "manager answer"
        assert result["has_vision"] is False

    @pytest.mark.asyncio
    async def test_returns_error_when_no_user_message(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        result = await service.process_chat_request([{"role": "assistant", "content": "Hi"}])

        assert result["success"] is False
        assert "No user message" in result["response"]


# ---------------------------------------------------------------------------
# _handle_text_request (legacy/image knowledge handler → Ragie retrieval, mocked)
# ---------------------------------------------------------------------------

class TestHandleTextRequest:
    @pytest.mark.asyncio
    async def test_returns_answer_from_retrieval(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        messages = [{"role": "user", "content": "How do I create a task?"}]
        with patch(
            "app.services.chat_service.run_retrieval",
            return_value={"answer": "Here is the answer.", "sources": []},
        ):
            result = await service._handle_text_request(messages, messages[0], None)

        assert result["success"] is True
        assert result["response"] == "Here is the answer."
        assert result["has_vision"] is False

    @pytest.mark.asyncio
    async def test_returns_error_on_retrieval_exception(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        messages = [{"role": "user", "content": "test"}]
        with patch(
            "app.services.chat_service.run_retrieval", side_effect=RuntimeError("retrieval failure")
        ):
            result = await service._handle_text_request(messages, messages[0], None)

        assert result["success"] is False
        assert "error" in result["response"].lower()


# ---------------------------------------------------------------------------
# _handle_agent_request (manager path → docs tool / direct answer / MCL tools)
# ---------------------------------------------------------------------------

class TestHandleAgentRequest:
    @pytest.mark.asyncio
    async def test_direct_recent_action_answer_does_not_search_docs(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        messages = [{"role": "user", "content": "What were the default settings?"}]
        with patch("app.services.chat_service.client") as mock_client, \
             patch("app.services.chat_service.retrieve") as mock_retrieve:
            mock_client.chat.completions.create.return_value = _text_response(
                "I used no due date, no market, the standard task type, and assigned it to you."
            )
            result = await service._handle_agent_request(
                messages, messages[0], "s1", None, "", "English", ""
            )

        mock_retrieve.assert_not_called()
        assert result["response"].startswith("I used no due date")

    @pytest.mark.asyncio
    async def test_knowledge_tool_searches_docs_and_enforces_sources(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        messages = [{"role": "user", "content": "How do I sync?"}]
        chunk = Unit(kind="text", id="u1", document_name="sync_guide.md", text="Tap Sync.")
        with patch("app.services.chat_service.contextualize", return_value="MCL sync"), \
             patch("app.services.chat_service.classify_surface", return_value="app"), \
             patch("app.services.chat_service.retrieve", return_value=[chunk]) as mock_retrieve, \
             patch("app.services.chat_service.client") as mock_client:
            mock_client.chat.completions.create.side_effect = [
                _tool_response("search_mcl_documentation", '{"query":"MCL sync"}'),
                _text_response("Tap Sync [Source: sync_guide.md]. Ignore this [Source: fake.md]."),
            ]
            result = await service._handle_agent_request(
                messages, messages[0], "s1", None, "", "English", ""
            )

        mock_retrieve.assert_called_once_with("MCL sync", surfaces=APP_SEARCH, min_score=0.0)
        assert "[Source: sync_guide.md]" in result["response"]
        assert "fake.md" not in result["response"]

    @pytest.mark.asyncio
    async def test_knowledge_tool_offers_step_markers_and_renders_screenshots(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        messages = [{"role": "user", "content": "How do I create a task?"}]
        procedure = Unit(
            kind="procedure", id="create_task", document_name="tasks.pdf",
            text="Creating a task", title="Creating a task",
            steps=[
                {"text": "Open the Task menu", "images": []},
                {"text": "Tap the + button", "images": [{"id": "img7", "path": "task_plus.png", "alt": "Plus button"}]},
            ],
        )
        with patch("app.services.chat_service.contextualize", return_value="create a task"), \
             patch("app.services.chat_service.retrieve", return_value=[procedure]), \
             patch("app.services.chat_service.client") as mock_client:
            mock_client.chat.completions.create.side_effect = [
                _tool_response("search_mcl_documentation", '{"query":"create a task"}'),
                _text_response(
                    "1. Open the Task menu [Source: tasks.pdf]\n"
                    "2. Tap the **+** button [Source: tasks.pdf] {{step:create_task.2}}"
                ),
            ]
            result = await service._handle_agent_request(
                messages, messages[0], "s1", None, "", "English", ""
            )

        tool_context = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"][-1]["content"]
        assert "{{step:create_task.2}}" in tool_context
        assert "# PROCEDURES" in tool_context

        assert "![Plus button](" in result["response"]
        assert "/images/task_plus.png" in result["response"]
        assert "{{step:" not in result["response"]

    @pytest.mark.asyncio
    async def test_invented_marker_is_stripped_not_rendered(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        messages = [{"role": "user", "content": "Show me the dashboard"}]
        chunk = Unit(kind="text", id="u1", document_name="dash.pdf", text="The dashboard shows cards.")
        with patch("app.services.chat_service.contextualize", return_value="dashboard"), \
             patch("app.services.chat_service.retrieve", return_value=[chunk]), \
             patch("app.services.chat_service.client") as mock_client:
            mock_client.chat.completions.create.side_effect = [
                _tool_response("search_mcl_documentation", '{"query":"dashboard"}'),
                _text_response("Here it is [Source: dash.pdf] {{image:not_a_real_id}}"),
            ]
            result = await service._handle_agent_request(
                messages, messages[0], "s1", None, "", "English", ""
            )

        assert "not_a_real_id" not in result["response"]
        assert "![" not in result["response"]
        assert "[Source: dash.pdf]" in result["response"]

    @pytest.mark.asyncio
    async def test_missing_information_is_recorded_with_the_full_question(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        messages = [{"role": "user", "content": "and can I use it on a smartwatch?"}]
        chunk = Unit(kind="text", id="u1", document_name="guide.pdf", text="MCL runs on phones and tablets.")
        recorded = []

        async def fake_record(question, language=None):
            recorded.append((question, language))
            return True

        with patch("app.services.chat_service.contextualize", return_value="MCL smartwatch support"),              patch("app.services.chat_service.retrieve", return_value=[chunk]),              patch("app.services.chat_service.record_gap", side_effect=fake_record),              patch("app.services.chat_service.client") as mock_client:
            mock_client.chat.completions.create.side_effect = [
                _tool_response("search_mcl_documentation", '{"query":"smartwatch"}'),
                _tool_response("report_missing_information", '{"question":"Does MCL run on a smartwatch?"}'),
                _text_response("I did not find that, and I logged your question."),
            ]
            result = await service._handle_agent_request(
                messages, messages[0], "s1", None, "", "English", ""
            )

        assert recorded == [("Does MCL run on a smartwatch?", "English")]
        assert "logged" in result["response"]

    @pytest.mark.asyncio
    async def test_answered_question_records_no_gap(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        messages = [{"role": "user", "content": "How do I sync?"}]
        chunk = Unit(kind="text", id="u1", document_name="sync.pdf", text="Tap Sync.")
        recorded = []

        async def fake_record(question, language=None):
            recorded.append(question)
            return True

        with patch("app.services.chat_service.contextualize", return_value="MCL sync"),              patch("app.services.chat_service.retrieve", return_value=[chunk]),              patch("app.services.chat_service.record_gap", side_effect=fake_record),              patch("app.services.chat_service.client") as mock_client:
            mock_client.chat.completions.create.side_effect = [
                _tool_response("search_mcl_documentation", '{"query":"sync"}'),
                _text_response("Tap Sync [Source: sync.pdf]."),
            ]
            await service._handle_agent_request(
                messages, messages[0], "s1", None, "", "English", ""
            )

        assert recorded == []

    @pytest.mark.asyncio
    async def test_write_confirmation_stores_conversation_context(
        self, mock_vision_service, mock_image_validator
    ):
        pending._PENDING.clear()
        service = make_service(mock_vision_service, mock_image_validator)
        auth = types.SimpleNamespace(access_token="t", user_id="u1", company_id="c", email="e@x.com")
        messages = [
            {"role": "user", "content": "Show me my tasks"},
            {"role": "assistant", "content": "1. New task\n2. New default task"},
            {"role": "user", "content": "Delete New task"},
        ]
        with patch("app.services.chat_service.client") as mock_client:
            mock_client.chat.completions.create.return_value = _tool_response(
                "delete_task",
                '{"todo_id":"9","confirmation":"Delete the task \\"New task\\"."}',
            )
            result = await service._handle_agent_request(
                messages, messages[-1], "s1", auth, "", "English", ""
            )

        cid = result["confirmation"]["id"]
        stored = pending.peek_pending(cid, "u1")
        assert stored["messages"][-3:] == messages
        assert result["requires_confirmation"] is True

    @pytest.mark.asyncio
    async def test_multi_write_request_stages_full_action_plan(
        self, mock_vision_service, mock_image_validator
    ):
        pending._PENDING.clear()
        service = make_service(mock_vision_service, mock_image_validator)
        auth = types.SimpleNamespace(access_token="t", user_id="u1", company_id="c", email="e@x.com")
        messages = [
            {"role": "user", "content": "Show me my tasks"},
            {"role": "assistant", "content": "1. First\n2. New default task\n3. This is a demo Task"},
            {"role": "user", "content": "Delete the last 2 tasks"},
        ]
        plan_args = {
            "summary": 'Delete 2 tasks: "New default task" and "This is a demo Task".',
            "steps": [
                {
                    "tool": "delete_task",
                    "summary": 'Delete the task "New default task".',
                    "todo_id": "2",
                    "description": None,
                    "due_date": None,
                    "market_id": None,
                    "assigned_user_id": None,
                    "note": None,
                },
                {
                    "tool": "delete_task",
                    "summary": 'Delete the task "This is a demo Task".',
                    "todo_id": "3",
                    "description": None,
                    "due_date": None,
                    "market_id": None,
                    "assigned_user_id": None,
                    "note": None,
                },
            ],
        }
        with patch("app.services.chat_service.client") as mock_client:
            mock_client.chat.completions.create.return_value = _tool_response(
                "confirm_mcl_action_plan",
                json.dumps(plan_args),
            )
            result = await service._handle_agent_request(
                messages, messages[-1], "s1", auth, "", "English", ""
            )

        cid = result["confirmation"]["id"]
        stored = pending.peek_pending(cid, "u1")
        assert stored["tool"] == "confirm_mcl_action_plan"
        assert len(stored["args"]["steps"]) == 2
        assert stored["args"]["steps"][0]["args"]["todo_id"] == "2"
        assert stored["args"]["steps"][1]["args"]["todo_id"] == "3"
        assert "Delete 2 tasks" in result["confirmation"]["action_summary"]


# ---------------------------------------------------------------------------
# KNOWLEDGE-labeled image → _answer_over_image (Decision 12 — Layer 1 + retrieval)
# ---------------------------------------------------------------------------

class TestAnswerOverImage:
    @pytest.mark.asyncio
    async def test_image_knowledge_uses_instruction_and_retrieval(
        self, mock_vision_service, mock_image_validator
    ):
        service = make_service(mock_vision_service, mock_image_validator)
        chunk = types.SimpleNamespace(document_name="guide.pdf", text="Tap the + button.")
        latest = {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this screen?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}},
            ],
        }
        with patch("app.services.chat_service.build_vision_query", return_value="checklist wizard departments") as mock_bq, \
             patch("app.services.chat_service.classify_surface", return_value="app"), \
             patch("app.services.chat_service.retrieve", return_value=[chunk]) as mock_retrieve, \
             patch("app.services.chat_service.client") as mock_client:
            resp = MagicMock()
            resp.choices[0].message.content = "This is the Checklist Wizard [Source: guide.pdf]."
            mock_client.chat.completions.create.return_value = resp
            # entry via the KNOWLEDGE handler, which now branches to the image path
            result = await service._handle_text_request([latest], latest, None, "")
            sent = mock_client.chat.completions.create.call_args.kwargs["messages"]

        assert result["success"] is True and result["has_vision"] is True
        mock_bq.assert_called_once()
        mock_retrieve.assert_called_once_with("checklist wizard departments", surfaces=APP_SEARCH, min_score=0.0)
        assert "MCL Support Specialist" in sent[0]["content"]
        assert any(m["role"] == "system" and "# TEXTUAL CONTEXT" in m["content"] for m in sent)
        assert "[Source: guide.pdf]" in result["response"]
