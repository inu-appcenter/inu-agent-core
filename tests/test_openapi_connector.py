import asyncio
from pathlib import Path
import json
from unittest.mock import AsyncMock, patch
import httpx

from app.tools.openapi import OpenApiConnector, OpenApiTool
from app.orchestrator.pruner import ToolPruner


def test_openapi_parsing():
    sample_path = Path(__file__).parent.parent / "app" / "tools" / "schemas" / "inu_portal_sample.json"
    with open(sample_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    connector = OpenApiConnector(base_url="http://localhost:8080")
    tools = connector.parse_spec(spec)

    assert len(tools) >= 4
    tool_names = [t.name for t in tools]
    assert "api_getCafeteriaMenu" in tool_names
    assert "api_getBusArrivals" in tool_names
    assert "api_getTodayTimeTable" in tool_names
    assert "api_getNotices" in tool_names

    # Check Cafeteria schema
    cafeteria_tool = next(t for t in tools if t.name == "api_getCafeteriaMenu")
    assert cafeteria_tool.category == "CAFETERIA"
    schema = cafeteria_tool.get_schema()
    params = schema["function"]["parameters"]
    assert "cafeteria" in params["properties"]
    assert "cafeteria" in params["required"]


def test_tool_pruner():
    sample_path = Path(__file__).parent.parent / "app" / "tools" / "schemas" / "inu_portal_sample.json"
    with open(sample_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    connector = OpenApiConnector()
    tools = connector.parse_spec(spec)

    # 1. Cafeteria Query
    cafeteria_query = "오늘 학생식당 점심 메뉴 뭐야?"
    pruned = ToolPruner.prune(cafeteria_query, tools, client="INTIP", max_tools=2)
    assert len(pruned) >= 1
    assert pruned[0].category == "CAFETERIA"

    # 2. Bus Query
    bus_query = "지금 셔틀버스 도착 시간 좀 알려줘"
    pruned = ToolPruner.prune(bus_query, tools, client="INTIP", max_tools=2)
    assert len(pruned) >= 1
    assert pruned[0].category == "BUS"

    # 4. Directory Query
    directory_query = "컴공 과사 전화번호 알려줘"
    pruned = ToolPruner.prune(directory_query, tools, client="INTIP", max_tools=2)
    assert len(pruned) >= 1
    assert any(t.category in ["DIRECTORY", "SEARCH"] for t in pruned)


def test_token_relay_and_execution():
    async def _test():
        tool = OpenApiTool(
            name="api_test_timetable",
            description="시간표 조회",
            method="GET",
            path="/api/timetables/today",
            parameters_schema={"type": "object", "properties": {}},
            category="TIMETABLE",
            requires_auth=True,
        )

        mock_response = httpx.Response(
            status_code=200,
            json={"data": [{"courseName": "운영체제", "room": "7호관 301호"}], "message": "성공"},
            request=httpx.Request("GET", "http://localhost:8080/api/timetables/today"),
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            # Execute with Bearer token in context
            context = {"authorization": "Bearer sample_student_jwt_token_123"}
            result = await tool.execute(arguments={}, context=context)

            assert mock_get.called
            # Verify Token Relay in request headers
            _, kwargs = mock_get.call_args
            headers = kwargs.get("headers", {})
            assert headers.get("Auth") == "sample_student_jwt_token_123"
            assert headers.get("Authorization") == "Bearer sample_student_jwt_token_123"

            # Verify response extraction
            assert isinstance(result, list)
            assert result[0]["courseName"] == "운영체제"

    asyncio.run(_test())


def test_large_result_truncation_guardrail():
    async def _test():
        tool = OpenApiTool(
            name="api_test_large_list",
            description="대량 결과 API",
            method="GET",
            path="/api/large-items",
            parameters_schema={"type": "object", "properties": {}},
        )

        # 20 items returned
        large_list = [{"id": i, "name": f"Item {i}"} for i in range(20)]
        mock_response = httpx.Response(
            status_code=200,
            json={"data": large_list},
            request=httpx.Request("GET", "http://localhost:8080/api/large-items"),
        )

        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await tool.execute(arguments={}, context={})

            # Should be truncated to top 6 to prevent context window explosion
            assert "total_count" in result
            assert result["total_count"] == 20
            assert len(result["items"]) == 6
            assert "notice" in result

    asyncio.run(_test())


def test_unified_search_parsing_and_card_synthesis():
    from app.orchestrator.card_synthesizer import CardSynthesizer
    sample_path = Path(__file__).parent.parent / "app" / "tools" / "schemas" / "inu_portal_sample.json"
    with open(sample_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    connector = OpenApiConnector()
    tools = connector.parse_spec(spec)
    tool_names = [t.name for t in tools]
    assert "api_unifiedSearch" in tool_names

    search_tool = next(t for t in tools if t.name == "api_unifiedSearch")
    assert search_tool.category == "SEARCH"

    # Test Card Synthesis
    mock_search_data = {
        "query": "장학금",
        "tab": "ALL",
        "totalCount": 10,
        "notices": {
            "totalCount": 3,
            "items": [
                {"id": 1, "title": "2026-2학기 교내장학금 신청 안내", "writer": "학생지원과", "createDate": "2026-09-01", "url": "/notice/1"},
            ]
        },
        "directory": {
            "totalCount": 1,
            "items": [
                {"name": "홍길동", "affiliation": "학생처", "detailAffiliation": "학생지원과", "phoneNumber": "032-835-9000", "position": "담당관"}
            ]
        }
    }

    card = CardSynthesizer.synthesize_for_domain(
        domain="SEARCH",
        tool_name="api_unifiedSearch",
        data=mock_search_data,
        query="장학금 신청 기간 알려줘",
    )
    assert card is not None
    assert card.card_type == "LIST_CARD"
    assert "장학금" in card.title
    assert len(card.items) >= 2
    tags = [it.tag for it in card.items]
    assert "학교공지" in tags
    assert "연락처" in tags

