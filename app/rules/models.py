"""
Pydantic data models for Declarative Action Rules (Coocon-style Recipe Engine)
"""
from typing import Dict, Any, Optional, Literal, List
from pydantic import BaseModel, Field


class RuleTarget(BaseModel):
    url: str = Field(..., description="Target external URL to call from on-device runner")
    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH"] = "GET"
    headers: Dict[str, str] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    body_template: Optional[Any] = Field(default=None, description="Request body or SSV template")


class FieldExtractor(BaseModel):
    selector: Optional[str] = Field(default=None, description="CSS selector for HTML parsing")
    json_path: Optional[str] = Field(default=None, description="JSON path key for REST parsing")
    regex: Optional[str] = Field(default=None, description="Optional regex extraction")
    attribute: Optional[str] = Field(default=None, description="HTML attribute to extract (e.g. href, src)")
    type: Literal["string", "number", "boolean", "list"] = "string"


class RuleExtraction(BaseModel):
    root_selector: Optional[str] = Field(default=None, description="Container selector for lists")
    fields: Dict[str, FieldExtractor] = Field(default_factory=dict)


class PrivacyPolicy(BaseModel):
    mask_fields: List[str] = Field(default_factory=lambda: ["studentNumber", "userName", "password"])
    retention: Literal["TRANSIENT", "SESSION", "PERSISTENT"] = "TRANSIENT"


class ActionRule(BaseModel):
    action_id: str = Field(..., description="Unique action identifier e.g. LMS_GET_ASSIGNMENTS")
    domain: Literal["LMS", "LIBRARY", "PORTAL", "DORM", "GENERAL"]
    protocol: Literal["HTTP_REST", "NEXACRO_SSV", "HTML_SCRAPE"] = "HTTP_REST"
    title: str
    description: str
    version: str = "1.0.0"
    target: RuleTarget
    extraction: Optional[RuleExtraction] = None
    privacy: PrivacyPolicy = Field(default_factory=PrivacyPolicy)
    default_card_type: Literal["METRIC_CARD", "STATUS_CARD", "LIST_CARD", "ACTION_CARD"] = "LIST_CARD"
