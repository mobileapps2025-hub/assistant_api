from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.dependencies import get_chat_service
from app.core.flow import flow
from app.core.logging import get_logger
from app.models import (
    AuthContext, Device, PlatformClosureRequest, PlatformClosureResponse, PlatformTurn,
    PlatformTurnRequest, PlatformTurnResponse,
)
from app.platform.auth import require_service_secret
from app.platform.closure import write_closure
from app.services.chat_service import ChatService

logger = get_logger(__name__)
router = APIRouter(prefix="/api/platform", tags=["platform"])


def _conversation(turn: PlatformTurnRequest) -> List[Dict[str, Any]]:
    history = [{"role": m.role, "content": m.content} for m in turn.history if m.content.strip()]
    return [*history, {"role": "user", "content": turn.message.strip()}]


def _auth_context(turn: PlatformTurnRequest) -> AuthContext:
    return AuthContext(
        user_id=turn.actor.user_id,
        company_id=turn.actor.company_id,
        role_ids=turn.actor.role_ids,
        platform_turn=PlatformTurn(
            id=turn.turn.id, token=turn.turn.token, operations_url=turn.turn.operations_url,
        ),
    )


@router.post("/turn", response_model=PlatformTurnResponse, response_model_by_alias=True)
async def turn(
    request: Request,
    body: PlatformTurnRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> PlatformTurnResponse:
    require_service_secret(request)
    flow(f"🏢 MCL.Api → MarieClaire  turn {body.turn.id} · service secret OK")
    if not body.message.strip():
        raise HTTPException(422, "message is required")

    _trace_incoming(body)
    result = await chat_service.process_chat_request(
        _conversation(body),
        session_id=f"platform:{body.actor.company_id}:{body.actor.user_id}",
        auth_context=_auth_context(body),
        device=Device(platform=body.actor.platform),
    )
    if not result.get("success", False):
        flow(f"❌ turn {body.turn.id} failed: {result.get('error', '?')[:80]}")
        raise HTTPException(502, result.get("error") or "MarieClaire could not answer this turn.")

    proposal = result.get("proposal")
    flow(f"📤 MarieClaire → MCL.Api  turn {body.turn.id} · reply {len(result['response'])} chars · "
         f"proposal: {proposal['operation'] if proposal else 'none'}")
    logger.info(f"[PLATFORM] turn={body.turn.id} user={body.actor.user_id} company={body.actor.company_id} answered")
    return PlatformTurnResponse(turn_id=body.turn.id, reply=result["response"], proposal=proposal)


@router.post("/closure", response_model=PlatformClosureResponse)
async def closure(request: Request, body: PlatformClosureRequest) -> PlatformClosureResponse:
    """MCL.Api executed or dropped a proposal; MarieClaire writes the line the user reads next."""
    require_service_secret(request)
    flow(f"🏢 MCL.Api → MarieClaire  closure · {body.outcome.status} ({body.outcome.code}) for user={body.actor.user_id}")
    language = {"de": "German", "es": "Spanish"}.get(body.actor.language, "English")
    history = [{"role": m.role, "content": m.content} for m in body.history]
    message = write_closure(language, history, body.proposal, body.outcome.model_dump(by_alias=True))
    flow(f"📤 MarieClaire → MCL.Api  closure · {'written' if message else 'none (fallback)'}")
    return PlatformClosureResponse(message=message)


def _trace_incoming(body: PlatformTurnRequest) -> None:
    actor = body.actor
    flow(f"👤 actor from session cookie: user={actor.user_id} company={actor.company_id} "
         f"roles={actor.role_ids} lang={actor.language} platform={actor.platform}")
    if body.context and (body.context.page or body.context.record):
        record = f"{body.context.record.type}#{body.context.record.id}" if body.context.record else "none"
        flow(f"🖥 screen context (verified by MCL.Api): page={body.context.page} record={record} "
             f"filters={body.context.filters or {}}")
    else:
        flow("🖥 screen context: none")
    flow(f"🎫 turn grant: token …{body.turn.token[-4:]} · operations at {body.turn.operations_url}")
    flow(f"💬 message ({len(body.history)} prior turns): {body.message.strip()[:60]}")
