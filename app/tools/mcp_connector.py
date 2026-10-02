"""
Remote MCP (Model Context Protocol) Connector for inu-portal-server
Connects to inu-portal-server's /mcp JSON-RPC 2.0 endpoint and dynamically mounts its tools.
"""
from typing import Dict, Any, List, Optional
import httpx
from uuid import uuid4

from app.core.config import settings
from app.core.logging import logger
from app.tools.base import BaseTool
from app.tools.coercer import SchemaCoercer


class McpRemoteTool(BaseTool):
    """
    A tool instance backed by a remote MCP (Model Context Protocol) Server.
    """
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        mcp_url: str,
        category: str = "PORTAL",
    ):
        self.name = f"api_{name.lower()}" if not name.lower().startswith("api_") else name.lower()
        self.original_name = name
        self.description = description
        self.input_schema = input_schema
        self.mcp_url = mcp_url
        self.category = category

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """
        Execute tool call against remote MCP Server using JSON-RPC 2.0.
        """
        # Dynamically coerce arguments against schema
        sanitized_arguments = SchemaCoercer.coerce(self.input_schema, arguments)

        req_id = f"mcp_{uuid4().hex[:8]}"
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": self.original_name,
                "arguments": sanitized_arguments,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        raw_token = context.get("auth") or context.get("authorization", "")
        clean_token = raw_token.replace("Bearer ", "").strip() if raw_token else ""
        if clean_token:
            headers["Authorization"] = f"Bearer {clean_token}"

        logger.info(f"Executing McpRemoteTool [{self.name}] on {self.mcp_url} (Auth: {bool(clean_token)})")

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(self.mcp_url, json=payload, headers=headers)
                if resp.status_code >= 400:
                    logger.warning(f"MCP remote call [{self.name}] failed with HTTP {resp.status_code}: {resp.text}")
                    return {"error": f"MCP request failed with status {resp.status_code}"}

                data = resp.json()
                if "error" in data and data["error"]:
                    err = data["error"]
                    logger.warning(f"MCP remote tool [{self.name}] returned JSON-RPC error: {err}")
                    return {"error": err.get("message", "MCP Tool Error"), "code": err.get("code")}

                result = data.get("result", {})
                is_error = result.get("isError", False)
                content_list = result.get("content", [])
                text_content = ""
                if isinstance(content_list, list):
                    for item in content_list:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "")

                is_val_err = "올바르지 않습니다" in text_content or "파라미터" in text_content
                if is_error or is_val_err:
                    logger.warning(f"MCP remote tool [{self.name}] indicated error (val_err={is_val_err}): {text_content}")
                    return {
                        "error": text_content or "MCP execution failed",
                        "summary": text_content,
                        "is_validation_error": is_val_err,
                    }

                return {
                    "summary": text_content,
                    "uiComponent": result.get("_uiComponent"),
                    "rawData": result.get("_rawData"),
                }
            except Exception as e:
                logger.error(f"Failed to execute McpRemoteTool [{self.name}]: {e}", exc_info=True)
                return {"error": str(e)}


class McpConnector:
    """
    Client for inu-portal-server's /mcp endpoint.
    """
    def __init__(self, base_url: Optional[str] = None):
        raw_base = base_url or settings.INU_PORTAL_SERVER_URL
        self.mcp_url = f"{raw_base.rstrip('/')}/mcp"

    async def fetch_tools(self) -> List[McpRemoteTool]:
        """
        Call tools/list on the remote MCP server and return wrapped McpRemoteTools.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": "list_tools",
            "method": "tools/list",
            "params": {},
        }
        logger.info(f"Connecting to remote MCP server at {self.mcp_url}...")

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                resp = await client.post(self.mcp_url, json=payload)
                if resp.status_code != 200:
                    logger.warning(f"Failed to fetch MCP tools: HTTP {resp.status_code}")
                    return []

                body = resp.json()
                result = body.get("result", {})
                tools_data = result.get("tools", [])

                mounted_tools = []
                for t in tools_data:
                    name = t.get("name", "")
                    description = t.get("description", "")
                    input_schema = t.get("inputSchema", {"type": "object", "properties": {}})

                    # Schema Augmentation: enrich parameters and descriptions for LLM compliance
                    input_schema = dict(input_schema or {"type": "object", "properties": {}})
                    props = dict(input_schema.get("properties", {}))
                    name_upper = name.upper()
                    if "UNIFIED_SEARCH" in name_upper and not props:
                        props["query"] = {
                            "type": "string",
                            "description": "교내 통합 검색어 (공지사항, 학사일정, 교수/학과 연락처 등)",
                        }
                    elif "LOST_PROPERTY" in name_upper and not props:
                        props["query"] = {
                            "type": "string",
                            "description": "분실물 검색어 (예: 지갑, 에어팟, 학생증, 우산 등)",
                        }
                    for p_name, p_def in props.items():
                        if isinstance(p_def, dict):
                            p_desc = p_def.get("description", "")
                            p_enum = p_def.get("enum")
                            p_type = p_def.get("type")
                            if p_enum and not any(kw in p_desc for kw in ["허용 값", "Enum", "중 하나"]):
                                p_def["description"] = f"{p_desc} (허용 값: {', '.join(map(str, p_enum))})".strip()
                            elif p_type == "integer" and any(kw in p_name.lower() or kw in p_desc for kw in ["day", "요일"]):
                                if "정수" not in p_desc:
                                    p_def["description"] = f"{p_desc} (반드시 월=1부터 일=7까지의 정수)".strip()
                    input_schema["properties"] = props

                    # Infer category accurately based on tool name and domain
                    name_lower = name.lower()
                    if "timetable_gap" in name_lower:
                        cat = "TIMETABLE_GAP"
                    elif "timetable" in name_lower:
                        cat = "TIMETABLE"
                    elif "cafeteria" in name_lower or "menu" in name_lower:
                        cat = "CAFETERIA"
                    elif "bus" in name_lower or "shuttle" in name_lower:
                        cat = "BUS"
                    elif "library" in name_lower or "seat" in name_lower:
                        cat = "LIBRARY"
                    elif "campus_watch" in name_lower:
                        cat = "CAMPUS_WATCH"
                    elif "keyword" in name_lower:
                        cat = "KEYWORD"
                    elif "reminder" in name_lower:
                        cat = "REMINDER"
                    elif "daily_brief" in name_lower or "brief" in name_lower:
                        cat = "DAILY_BRIEF"
                    elif "settings" in name_lower or "chat_push" in name_lower:
                        cat = "SETTINGS"
                    elif "notice" in name_lower:
                        cat = "NOTICE"
                    elif "schedule" in name_lower or "calendar" in name_lower:
                        cat = "SCHEDULE"
                    elif "directory" in name_lower or "contact" in name_lower:
                        cat = "DIRECTORY"
                    elif "club" in name_lower:
                        cat = "CLUB"
                    elif "lost_property" in name_lower or "lost" in name_lower:
                        cat = "LOST_PROPERTY"
                    elif "search" in name_lower:
                        cat = "SEARCH"
                    elif "weather" in name_lower:
                        cat = "WEATHER"
                    elif "lms" in name_lower:
                        cat = "LMS"
                    elif "inu_ai" in name_lower or "knowledge" in name_lower:
                        cat = "INU_AI_KNOWLEDGE"
                    elif "academic" in name_lower:
                        cat = "PORTAL"
                    else:
                        cat = "GENERAL"

                    tool_instance = McpRemoteTool(
                        name=name,
                        description=description,
                        input_schema=input_schema,
                        mcp_url=self.mcp_url,
                        category=cat,
                    )
                    mounted_tools.append(tool_instance)

                logger.info(f"Successfully mounted {len(mounted_tools)} remote MCP tools from {self.mcp_url}")
                return mounted_tools
            except Exception as e:
                logger.warning(f"Remote MCP server {self.mcp_url} not reachable: {e}")
                return []
