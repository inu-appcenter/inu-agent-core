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


@pytest.mark.asyncio
async def test_mcp_remote_tool_execute_is_error():
    tool = McpRemoteTool(
        name="DIRECTORY",
        description="교내 연락처 조회",
        input_schema={"type": "object"},
        mcp_url="http://localhost:8080/mcp",
        category="DIRECTORY",
    )

    mock_result_data = {
        "jsonrpc": "2.0",
        "id": "mcp_12345",
        "result": {
            "isError": True,
            "content": [{"type": "text", "text": "교내 시스템 오류가 발생했습니다."}],
        },
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_result_data

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        res = await tool.execute({"query": "오류"}, {})

        assert "error" in res
        assert res["error"] == "교내 시스템 오류가 발생했습니다."


def test_card_synthesizer_mcp_directory_and_cafeteria():
    from app.orchestrator.card_synthesizer import CardSynthesizer

    # Test MCP Professor directory card synthesis
    mcp_prof_data = {
        "summary": "교수 연락처",
        "rawData": [
            {
                "id": 1,
                "name": "박문주",
                "position": "교수",
                "detailAffiliation": "컴퓨터공학부",
                "phoneNumber": "8556",
                "email": "mpark@inu.ac.kr",
            }
        ],
    }
    card_prof = CardSynthesizer.synthesize_for_domain("DIRECTORY", "api_directory", mcp_prof_data, "박문주 연락처")
    assert card_prof is not None
    assert card_prof.card_type == "LIST_CARD"
    assert card_prof.items[0].title == "박문주"
    assert "8556" in card_prof.items[0].subtitle

    # Test MCP Office directory card synthesis
    mcp_office_data = {
        "summary": "과사 연락처",
        "rawData": [
            {
                "id": 2,
                "departmentName": "컴퓨터공학부",
                "officePhoneNumber": "032-835-8490",
                "officeLocation": "7호관 410호",
            }
        ],
    }
    card_office = CardSynthesizer.synthesize_for_domain("DIRECTORY", "api_directory", mcp_office_data, "컴공 과사")
    assert card_office is not None
    assert card_office.items[0].title == "컴퓨터공학부"
    assert card_office.items[0].tag == "학과사무실"
    assert "032-835-8490" in card_office.items[0].subtitle

    # Test MCP Cafeteria card synthesis
    mcp_caf_data = {
        "summary": "오늘 학식 메뉴",
        "rawData": {
            "cafeteria": "학생식당",
            "breakfast": "토스트",
            "lunch": "제육볶음\n미역국",
            "dinner": "-",
        },
    }
    card_caf = CardSynthesizer.synthesize_for_domain("CAFETERIA", "api_cafeteria", mcp_caf_data, "학식 메뉴")
    assert card_caf is not None
    assert len(card_caf.items) == 2
    assert "조식" in card_caf.items[0].title
    assert "중식" in card_caf.items[1].title

