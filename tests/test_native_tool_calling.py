import pytest
from unittest.mock import AsyncMock, patch
from app.llm.schemas import ChatRequest, ChatMessage
from app.orchestrator.engine import AgentOrchestrator
from app.tools.registry import tool_registry


@pytest.mark.asyncio
async def test_native_tool_calling_single_turn_academic_and_directory():
    await tool_registry.initialize_all_tools()
    orchestrator = AgentOrchestrator()
    request = ChatRequest(
        message="내 담임교수님 전화번호?",
        history=[],
        client_context={
            "academicDisplay": {
                "advisorProfessorName": "박문주",
                "departmentName": "컴퓨터공학부",
                "koreanName": "배현준",
            }
        },
    )

    # Mock tool calls: Hop 1 calls PORTAL -> Hop 2 calls DIRECTORY -> Hop 3 finishes with no tool calls
    responses = [
        # Hop 1: LLM decides to call PORTAL
        {
            "thought": "사용자의 지도교수 정보를 확인하기 위해 포털 학적 정보를 먼저 조회합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "action_portal_get_academic_record",
                        "arguments": {},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 2: LLM sees advisor 박문주, and calls DIRECTORY with query="박문주"
        {
            "thought": "확인된 박문주 교수님의 연락처를 교내 전화번호부에서 검색합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "api_searchContacts",
                        "arguments": {"query": "박문주"},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 3: Done with tools
        {
            "thought": "정보 수집이 완료되어 최종 답변을 작성합니다.",
            "content": "학우님의 지도교수님은 컴퓨터공학부 박문주 교수님이며, 연락처는 다음과 같습니다.",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": "완료"},
        },
    ]

    async def mock_stream_chat(messages):
        yield "학우님의 지도교수님은 **컴퓨터공학부 박문주 교수님**이시며,"
        yield " 연락처는 032-835-8442 입니다."

    with patch("app.orchestrator.engine.llm_client.chat_with_tools", side_effect=responses), \
         patch("app.orchestrator.engine.llm_client.stream_chat", side_effect=mock_stream_chat):
        
        events = []
        async for ev in orchestrator.run_stream(request):
            events.append(ev)

        event_types = [e.event_type for e in events]
        assert "THINKING" in event_types
        assert "STATUS" in event_types
        assert "CARD" in event_types
        assert "TOKEN" in event_types
        assert "DONE" in event_types

        # Verify card was synthesized for academic
        cards = [e.card for e in events if e.card is not None]
        assert len(cards) >= 1
