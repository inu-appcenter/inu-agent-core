"""
Data schemas and Pydantic models for LLM, Agent Protocols, and Server-Driven UI Cards
"""
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# 1. Chat & Stream Models
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="User prompt or query")
    history: List[ChatMessage] = Field(default_factory=list, description="Recent conversation history")
    session_id: Optional[str] = Field(default=None, description="Client chat session ID for multi-turn callbacks")
    client: Literal["INTIP", "UNIDORM", "WEB"] = Field(default="INTIP", description="Originating client platform")
    client_context: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="clientContext",
        description="Client device context (e.g. linked accounts)",
    )

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


# -----------------------------------------------------------------------------
# 2. Client Action Protocols (Coocon-style On-Device Scraping/Execution)
# -----------------------------------------------------------------------------

class ClientActionRequest(BaseModel):
    url: str
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET"
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    body: Optional[Any] = None


class ClientActionInstruction(BaseModel):
    action_id: str
    session_id: Optional[str] = None
    auth_domain: Literal["LIBRARY", "PORTAL", "LMS", "DORM", "NONE"]
    protocol: Literal["HTTP_REST", "NEXACRO_SSV"] = "HTTP_REST"
    action_type: Literal["QUERY", "MUTATION", "DOWNLOAD"] = "QUERY"
    requires_confirmation: bool = False
    confirmation_message: Optional[str] = None
    request: ClientActionRequest
    extraction_rules: Optional[Dict[str, Any]] = None
    download_metadata: Optional[Dict[str, str]] = None


class ClientActionResult(BaseModel):
    action_id: str
    session_id: Optional[str] = None
    success: bool
    status_code: Optional[int] = None
    data: Optional[Any] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class ChatActionCallbackRequest(BaseModel):
    action_id: str
    session_id: Optional[str] = None
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    original_message: Optional[str] = Field(default="", description="Original user prompt to complete ReAct synthesis")
    client: Literal["INTIP", "UNIDORM", "WEB"] = Field(default="INTIP", description="Client tenant")
    client_context: Optional[Dict[str, Any]] = Field(default=None, alias="clientContext")
    history: List[ChatMessage] = Field(default_factory=list, description="Conversation history")

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


# -----------------------------------------------------------------------------
# 3. Server-Driven UI (SDUI) 4 Universal Card Templates
# -----------------------------------------------------------------------------

class CardBadge(BaseModel):
    text: str
    theme: Literal["primary", "success", "warning", "danger", "neutral"] = "neutral"


class MetricCardItem(BaseModel):
    label: str
    value: str
    highlight: bool = False


class MetricCard(BaseModel):
    card_type: Literal["METRIC_CARD"] = "METRIC_CARD"
    title: str
    badge: Optional[CardBadge] = None
    main_metric: Optional[MetricCardItem] = None
    sub_details: List[MetricCardItem] = Field(default_factory=list)
    action_link: Optional[Dict[str, str]] = None


class StatusCard(BaseModel):
    card_type: Literal["STATUS_CARD"] = "STATUS_CARD"
    title: str
    status: str
    progress_percent: Optional[int] = None
    time_remaining: Optional[str] = None
    primary_action_label: Optional[str] = None
    action_id: Optional[str] = None


class ListItem(BaseModel):
    title: str
    subtitle: Optional[str] = None
    tag: Optional[str] = None
    link: Optional[str] = None


class ListCard(BaseModel):
    card_type: Literal["LIST_CARD"] = "LIST_CARD"
    title: str
    items: List[ListItem] = Field(default_factory=list)
    footer_text: Optional[str] = None


class ActionCard(BaseModel):
    card_type: Literal["ACTION_CARD"] = "ACTION_CARD"
    title: str
    description: str
    confirm_label: str
    cancel_label: Optional[str] = "취소"
    action_payload: Dict[str, Any]


class CardLink(BaseModel):
    label: str
    route: str


class ComponentCard(BaseModel):
    """
    Dedicated Interactive Component Card for INTIP UI (LIBRARY_SEAT_CONFIRM, LIBRARY_ROOMS, ACADEMIC_INFO, etc.)
    """
    card_type: Literal["COMPONENT_CARD"] = "COMPONENT_CARD"
    type: str
    title: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    link: Optional[CardLink] = None


# Union type for all supported Cards
GenerativeCard = Union[MetricCard, StatusCard, ListCard, ActionCard, ComponentCard]


# -----------------------------------------------------------------------------
# 4. Agent Stream Event
# -----------------------------------------------------------------------------

class AgentStreamEvent(BaseModel):
    event_type: Literal["TOKEN", "ACTION_REQUIRED", "CARD", "ERROR", "DONE", "STATUS", "THINKING", "DEBUG"]
    content: Optional[str] = None
    action: Optional[ClientActionInstruction] = None
    card: Optional[GenerativeCard] = None
    error: Optional[str] = None
    # Live tool integration & thinking status
    status_id: Optional[str] = None
    status_title: Optional[str] = None
    status_category: Optional[str] = None
    status_state: Optional[Literal["running", "completed", "failed", "empty"]] = None
    thinking: Optional[str] = None
    # Developer diagnostic / debug trace
    debug: Optional[Dict[str, Any]] = None
