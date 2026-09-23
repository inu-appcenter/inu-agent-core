import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.llm.schemas import ChatRequest, AgentStreamEvent
from app.orchestrator.engine import orchestrator
from app.tools.base import BaseTool


class FailingTool(BaseTool):
    name = "api_getCafeteriaMenu"
    description = "학식 메뉴 조회 도구"
    category = "CAFETERIA"
    parameters_schema = {
        "type": "object",
        "properties": {
            "cafeteria": {"type": "string"},
            "day": {"type": "integer"}
        }
    }

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            }
        }

    async def execute(self, arguments, context):
        # 모의 네트워크/서버 에러 반환
        return {"error": "Connection timed out (504 Gateway Timeout)"}


class EmptyCafeteriaTool(BaseTool):
    name = "api_getCafeteriaMenu"
    description = "학식 메뉴 조회 도구"
    category = "CAFETERIA"
    parameters_schema = {
        "type": "object",
        "properties": {
            "cafeteria": {"type": "string"},
            "day": {"type": "integer"}
        }
    }

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            }
        }

    async def execute(self, arguments, context):
        # 주말/방학 등으로 메뉴가 등록되지 않은 빈 응답
        return []


@pytest.mark.asyncio
async def test_tool_failure_emits_failed_status_and_system_error_summary():
    """도구 실행 중 500/Timeout 등 오류 발생 시 status_state=failed와 투명한 에러 고지 전달 검증"""
    failing_tool = FailingTool()
    req = ChatRequest(message="오늘 학생식당 점심 메뉴 뭐야?", client="INTIP")

    events = []
    async for item in orchestrator._execute_tool(
        tool=failing_tool,
        tool_args={"cafeteria": "학생식당", "day": 1},
        request=req,
        exec_context={},
        emitted_cards=set(),
        emitted_actions=set(),
    ):
        events.append(item)

    # 1. STATUS 이벤트 검증 (running -> failed)
    status_events = [e[0] for e in events if e[0] and e[0].event_type == "STATUS"]
    assert len(status_events) == 2
    assert status_events[0].status_state == "running"
    assert status_events[1].status_state == "failed"
    assert "조회 실패" in status_events[1].status_title

    # 2. summary_out 검증 (환각 금지 지침 및 에러 사유 포함)
    summaries = [e[1] for e in events if e[1]]
    assert len(summaries) == 1
    summary = summaries[0]
    assert "[시스템 오류 고지]" in summary
    assert "Connection timed out" in summary
    assert "가상의 정보" in summary or "지어내지 말고" in summary


@pytest.mark.asyncio
async def test_empty_cafeteria_emits_no_menu_notice_and_prevents_hallucination():
    """식단이 등록되지 않은 경우 가상 메뉴를 지어내지 못하도록 안내 지침 주입 검증"""
    empty_tool = EmptyCafeteriaTool()
    req = ChatRequest(message="학생식당 메뉴 알려줘", client="INTIP")

    events = []
    async for item in orchestrator._execute_tool(
        tool=empty_tool,
        tool_args={"cafeteria": "학생식당", "day": 1},
        request=req,
        exec_context={},
        emitted_cards=set(),
        emitted_actions=set(),
    ):
        events.append(item)

    summaries = [e[1] for e in events if e[1]]
    assert len(summaries) == 1
    summary = summaries[0]
    assert "등록된 식단 메뉴가 없습니다" in summary
    assert "가상의 식단 메뉴를 지어내어 답변하지 마세요" in summary
