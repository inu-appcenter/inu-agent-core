"""
Model Context Protocol (MCP) Server Implementation for inu-agent-core
Standard JSON-RPC 2.0 over HTTP and SSE transports.
Exposes both Server-Direct Tools and On-Device Client Action Tools to MCP clients and AI servers.
"""
from typing import Dict, Any, List, Optional
import json
import asyncio
from uuid import uuid4
from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse, StreamingResponse

from app.core.logging import logger
from app.tools.registry import tool_registry
from app.tools.action_tool import ClientActionTool

router = APIRouter(prefix="/mcp", tags=["mcp"])

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {
    "name": "inu-portal-mcp-server",
    "version": "1.0.0",
}

# Pre-defined Incheon National University Static Resources
STATIC_RESOURCES = [
    {
        "uri": "inu://calendar/academic",
        "name": "인천대학교 연간 학사일정",
        "description": "2026학년도 인천대학교 수강신청, 시험, 방학 및 주요 학사일정 안내",
        "mimeType": "application/json",
        "content": json.dumps({
            "first_semester": {
                "course_registration": "2월 중순",
                "semester_start": "3월 2일",
                "midterm_exam": "4월 중순",
                "final_exam": "6월 중순",
                "summer_vacation": "6월 말"
            },
            "second_semester": {
                "course_registration": "8월 중순",
                "semester_start": "9월 1일",
                "midterm_exam": "10월 중순",
                "final_exam": "12월 중순",
                "winter_vacation": "12월 말"
            }
        }, ensure_ascii=False)
    },
    {
        "uri": "inu://directory/emergency",
        "name": "교내 주요 긴급/편의 연락처",
        "description": "인천대학교 종합상황실, 보건진료소, 학생지원팀 등 주요 연락처",
        "mimeType": "application/json",
        "content": json.dumps({
            "종합상황실": "032-835-9119",
            "학생지원과": "032-835-9221",
            "학사지원팀": "032-835-9231",
            "학산도서관": "032-835-9411",
            "생활관(기숙사)": "032-835-9800"
        }, ensure_ascii=False)
    }
]


def format_tool_to_mcp(tool: Any) -> Dict[str, Any]:
    """Convert an internal BaseTool or ClientActionTool into an MCP Tool specification."""
    schema = tool.get_schema()
    function_def = schema.get("function", {})
    name = function_def.get("name", tool.name)
    description = function_def.get("description", getattr(tool, "description", ""))
    parameters = function_def.get("parameters", {"type": "object", "properties": {}})

    metadata: Dict[str, Any] = {
        "category": getattr(tool, "category", "GENERAL"),
    }

    if isinstance(tool, ClientActionTool):
        rule = tool.rule
        metadata["executionType"] = "CLIENT_ACTION"
        metadata["authDomain"] = rule.domain
        metadata["protocol"] = rule.protocol
        metadata["actionType"] = getattr(rule, "action_type", "QUERY")
        metadata["requiresConfirmation"] = getattr(rule, "requires_confirmation", False)
        if getattr(rule, "confirmation_message", None):
            metadata["confirmationMessage"] = rule.confirmation_message
        if getattr(rule, "download_metadata", None):
            metadata["downloadMetadata"] = rule.download_metadata
    else:
        metadata["executionType"] = "SERVER_DIRECT"

    return {
        "name": name,
        "description": description,
        "inputSchema": parameters,
        "metadata": metadata,
    }


async def handle_jsonrpc_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """Handle standard JSON-RPC 2.0 methods for MCP."""
    req_id = body.get("id")
    method = body.get("method")
    params = body.get("params", {})

    logger.debug(f"[MCP] Request: method={method}, id={req_id}")

    # 1. Initialize
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"listChanged": False},
                    "prompts": {"listChanged": False},
                },
                "serverInfo": SERVER_INFO,
            }
        }

    # 2. Ping
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    # 3. Tools List
    elif method == "tools/list":
        tools = tool_registry.list_tools()
        formatted_tools = [format_tool_to_mcp(t) for t in tools]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": formatted_tools
            }
        }

    # 4. Tools Call
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        tool = tool_registry.get_tool(tool_name)

        if not tool:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool '{tool_name}' not found in registry",
                }
            }

        try:
            result = await tool.execute(arguments=arguments, context={})
            # Check if result is a ClientActionInstruction
            if hasattr(result, "action_id"):
                instruction_data = result.model_dump()
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Client action dispatched: {result.action_id} (Domain: {result.auth_domain}, Protocol: {result.protocol})",
                            }
                        ],
                        "clientAction": instruction_data,
                        "isError": False,
                    }
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result),
                            }
                        ],
                        "data": result,
                        "isError": False,
                    }
                }
        except Exception as e:
            logger.error(f"[MCP] Tool execution error for '{tool_name}': {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error executing tool: {str(e)}"}],
                    "isError": True,
                }
            }

    # 5. Resources List
    elif method == "resources/list":
        res_list = [
            {
                "uri": r["uri"],
                "name": r["name"],
                "description": r["description"],
                "mimeType": r["mimeType"],
            }
            for r in STATIC_RESOURCES
        ]
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "resources": res_list
            }
        }

    # 6. Resources Read
    elif method == "resources/read":
        uri = params.get("uri")
        found = next((r for r in STATIC_RESOURCES if r["uri"] == uri), None)
        if not found:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32602,
                    "message": f"Resource with URI '{uri}' not found",
                }
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "contents": [
                    {
                        "uri": found["uri"],
                        "mimeType": found["mimeType"],
                        "text": found["content"],
                    }
                ]
            }
        }

    # 7. Unsupported Method
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' not implemented",
            }
        }


@router.post("", summary="MCP JSON-RPC Endpoint")
@router.post("/messages", summary="MCP Messages Endpoint (SSE transport target)")
async def mcp_post(request: Request):
    """Direct POST handler for JSON-RPC 2.0 MCP requests."""
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        )

    response = await handle_jsonrpc_request(body)
    return JSONResponse(content=response)


@router.get("/sse", summary="MCP Server-Sent Events (SSE) Transport")
async def mcp_sse(request: Request):
    """
    Standard MCP SSE Transport Endpoint.
    Sends an initial 'endpoint' event telling the client where to POST messages.
    """
    session_id = uuid4().hex[:12]

    async def event_generator():
        # 1. Emit endpoint event pointing to messages URL
        yield f"event: endpoint\ndata: /mcp/messages?sessionId={session_id}\n\n"
        if request.query_params.get("single") == "true":
            return

        # 2. Keep connection alive with periodic pings
        try:
            while not await request.is_disconnected():
                await asyncio.sleep(1)
                yield ": keepalive\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
