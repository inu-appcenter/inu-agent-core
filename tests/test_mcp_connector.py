import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.tools.mcp_connector import McpConnector, McpRemoteTool


@pytest.mark.asyncio
async def test_mcp_connector_fetch_tools_success():
    mock_tools_data = {
        "jsonrpc": "2.0",
        "id": "list_tools",
        "result": {
            "tools": [
                {
                    "name": "TIMETABLE",
                    "description": "대표 시간표 조회",
                    "inputSchema": {"type": "object", "properties": {"targetDay": {"type": "string"}}},
                },
                {
                    "name": "CAFETERIA",
                    "description": "학식 메뉴 조회",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
        },
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_tools_data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        connector = McpConnector("http://localhost:8080")
        tools = await connector.fetch_tools()

        assert len(tools) == 2
        assert tools[0].name == "api_timetable"
        assert tools[0].category == "TIMETABLE"
        assert tools[1].name == "api_cafeteria"
        assert tools[1].category == "CAFETERIA"


@pytest.mark.asyncio
async def test_mcp_remote_tool_execute():
    tool = McpRemoteTool(
        name="TIMETABLE",
        description="대표 시간표 조회",
        input_schema={"type": "object"},
        mcp_url="http://localhost:8080/mcp",
        category="TIMETABLE",
    )

    mock_result_data = {
        "jsonrpc": "2.0",
        "id": "mcp_12345",
        "result": {
            "content": [{"type": "text", "text": "오늘 수업은 1개 있습니다."}],
            "_uiComponent": {"type": "TIMETABLE", "data": {}},
            "_rawData": {"count": 1},
        },
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_result_data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        res = await tool.execute({"targetDay": "TODAY"}, {"auth": "Bearer test-jwt-token"})

        assert res["summary"] == "오늘 수업은 1개 있습니다."
        assert res["uiComponent"]["type"] == "TIMETABLE"
        assert res["rawData"]["count"] == 1

        # Check call arguments
        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["method"] == "tools/call"
        assert call_kwargs["json"]["params"]["name"] == "TIMETABLE"
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-jwt-token"
