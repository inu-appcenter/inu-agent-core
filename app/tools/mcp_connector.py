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
        req_id = f"mcp_{uuid4().hex[:8]}"
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "tools/call",
            "params": {
                "name": self.original_name,
                "arguments": arguments,
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

        async with httpx.AsyncClient(timeout=10.0) as client:
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

                if is_error:
                    logger.warning(f"MCP remote tool [{self.name}] indicated isError=True: {text_content}")
                    return {"error": text_content or "MCP execution failed", "summary": text_content}

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

        async with httpx.AsyncClient(timeout=5.0) as client:
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

                    # Infer category accurately based on tool name and domain
                    name_lower = name.lower()
                    if "timetable" in name_lower:
                        cat = "TIMETABLE"
                    elif "cafeteria" in name_lower or "menu" in name_lower:
                        cat = "CAFETERIA"
                    elif "bus" in name_lower or "shuttle" in name_lower:
                        cat = "BUS"
                    elif "library" in name_lower or "seat" in name_lower:
                        cat = "LIBRARY"
                    elif "notice" in name_lower:
                        cat = "NOTICE"
                    elif "schedule" in name_lower or "calendar" in name_lower:
                        cat = "SCHEDULE"
                    elif "directory" in name_lower or "contact" in name_lower:
                        cat = "DIRECTORY"
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
