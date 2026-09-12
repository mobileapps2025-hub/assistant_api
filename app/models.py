from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


class ContentItem(BaseModel):
    """Content item for multimodal messages (text or image)."""
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None  # {"url": "data:image/png;base64,..."}

    class Config:
        json_schema_extra = {
            "examples": [
                {"type": "text", "text": "What is this screen?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}}
            ]
        }


class Message(BaseModel):
    """Message in a chat conversation, supporting text and images."""
    role: str  # "user", "assistant", "system"
    content: Optional[List[ContentItem] | str] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    annotations: Optional[str] = None

    class Config:
        json_schema_extra = {
            "examples": [
                {"role": "user", "content": "Hello, how can I create a checklist?"},
                {"role": "user", "content": [
                    {"type": "text", "text": "What can I do on this screen?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                ]}
            ]
        }


class PlatformTurn(BaseModel):
    """What MCL.Api hands MarieClaire for one turn so she can ask it for live data.

    The token is bound by MCL.Api to this actor and this turn, expires in about a minute, and
    only opens the operations allowlist. It is never a user credential.
    """
    id: str
    token: str
    operations_url: str


class AuthContext(BaseModel):
    access_token: Optional[str] = None      # legacy callers: the user's MCL bearer token
    user_id: Optional[str] = None
    company_id: Optional[str] = None
    company_name: Optional[str] = None
    full_name: Optional[str] = None
    email: Optional[str] = None
    role_ids: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)   # readable permission keys derived by MCL.Api
    platform_turn: Optional[PlatformTurn] = None    # MCL.Api callers: identity came from the session


class PlatformActor(BaseModel):
    """The signed-in user as MCL.Api derived them from the validated session. Never from the browser."""
    user_id: str = Field(alias="userId")
    company_id: str = Field(alias="companyId")
    role_ids: List[str] = Field(default_factory=list, alias="roleIds")
    language: str = "de"
    platform: str = "web"
    capabilities: List[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class PlatformRecord(BaseModel):
    type: str
    id: str


class PlatformContext(BaseModel):
    """Where the user is in the app. MCL.Api verified any record before forwarding it."""
    page: Optional[str] = None
    record: Optional[PlatformRecord] = None
    filters: Optional[dict] = None


class PlatformTurnGrant(BaseModel):
    id: str
    token: str
    operations_url: str = Field(alias="operationsUrl")

    model_config = {"populate_by_name": True}


class PlatformHistoryMessage(BaseModel):
    role: str
    content: str


class PlatformTurnRequest(BaseModel):
    actor: PlatformActor
    message: str
    history: List[PlatformHistoryMessage] = Field(default_factory=list)
    context: Optional[PlatformContext] = None
    turn: PlatformTurnGrant


class PlatformOutcome(BaseModel):
    status: str
    code: str
    task_id: Optional[str] = Field(default=None, alias="taskId")
    task_number: Optional[int] = Field(default=None, alias="taskNumber")

    model_config = {"populate_by_name": True}


class PlatformFeedbackRequest(BaseModel):
    """A user reporting that an answer was wrong (or withdrawing that report), via MCL.Api."""
    actor: PlatformActor
    question: str
    answer: Optional[str] = None
    note: Optional[str] = None
    withdraw: bool = False


class PlatformFeedbackResponse(BaseModel):
    recorded: bool


class PlatformClosureRequest(BaseModel):
    actor: PlatformActor
    history: List[PlatformHistoryMessage] = Field(default_factory=list)
    proposal: dict
    outcome: PlatformOutcome


class PlatformClosureResponse(BaseModel):
    message: Optional[str] = None


class PlatformTurnResponse(BaseModel):
    turn_id: str = Field(alias="turnId")
    reply: str
    proposal: Optional[dict] = None

    model_config = {"populate_by_name": True}


class SessionRequest(BaseModel):
    """Request to establish a session from a token shared by the MCL app."""
    access_token: str


class SessionResponse(BaseModel):
    """Resolved session identity, derived from the shared token via UserInfo."""
    access_token: str
    user_id: str
    company_id: str
    company_name: str
    full_name: str
    email: str


class MarketInfo(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    soll_bestand: Optional[float] = None
    kasseneinsaetze: Optional[float] = None
    summe_kasseneinsatze: Optional[float] = None


class UserMarketsResponse(BaseModel):
    markets: List[MarketInfo]
    total: int


class Device(BaseModel):
    platform: Optional[str] = None      # iOS | Android | Web
    form_factor: Optional[str] = None   # phone | tablet | desktop
    app_version: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: Optional[str] = None
    auth_context: Optional[AuthContext] = None
    device: Optional[Device] = None
    timezone: Optional[str] = None      # IANA zone from the browser, e.g. "Europe/Berlin"


class FeedbackRequest(BaseModel):
    """Schema for receiving feedback from frontend."""
    response_id: str
    feedback_type: str  # 'positive' or 'negative'
    user_comment: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "response_id": "resp_12345",
                "feedback_type": "positive",
                "user_comment": "Very helpful answer!"
            }
        }


class FeedbackResponse(BaseModel):
    """Schema for feedback response."""
    id: int
    response_id: str
    feedback_type: str
    user_comment: Optional[str]
    created_at: datetime
    processed: bool

    class Config:
        from_attributes = True


class Confirmation(BaseModel):
    id: str
    risk: str                # write | destructive
    action_summary: str


class ChatResponse(BaseModel):
    """Chat response with tracking ID."""
    response: str
    response_id: str
    sources: Optional[List[str]] = None
    requires_confirmation: bool = False
    confirmation: Optional[Confirmation] = None


class ConfirmRequest(BaseModel):
    confirmation_id: str
    decision: str            # approve | reject
    session_id: Optional[str] = None
    auth_context: Optional[AuthContext] = None


# temporary: username/password login for testing, until the MCL shared-session handoff lands.
class LoginRequest(BaseModel):
    user_name: str
    password: str

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Here is your answer...",
                "response_id": "resp_12345",
                "sources": []
            }
        }


def generate_response_id() -> str:
    """Generate a unique response ID for tracking."""
    return f"resp_{uuid.uuid4().hex[:8]}"


class MemorySaveRequest(BaseModel):
    messages: List[Dict[str, Any]]
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class MemoryInfo(BaseModel):
    id: str
    title: str = ""
    category: str = ""
    importance: str = "low"
    tags: List[str] = []
    content: str = ""
    created: str = ""
    updated: str = ""


class MemoryListResponse(BaseModel):
    memories: List[MemoryInfo]


class MemorySaveResponse(BaseModel):
    saved: List[MemoryInfo] = []
    updated: List[MemoryInfo] = []
    deleted: List[str] = []


class MemoryUpdateRequest(BaseModel):
    content: str


class MemoryRecallResponse(BaseModel):
    context: str = ""
    memories: List[MemoryInfo] = []


class MemoryStoreRequest(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    user_id: Optional[str] = None
