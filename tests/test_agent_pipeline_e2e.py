import pytest
from unittest.mock import AsyncMock, patch
from app.llm.schemas import ChatRequest, ChatMessage
from app.orchestrator.engine import AgentOrchestrator
from app.tools.registry import tool_registry


@pytest.mark.asyncio
async def test_e2e_out_of_scope_guardrail():
    await tool_registry.initialize_all_tools()
    orchestrator = AgentOrchestrator()
    request = ChatRequest(
        message="파이썬으로 퀵소트 알고리즘 코드 짜줘",
        history=[],
    )
    events = []
    async for ev in orchestrator.run_stream(request):
        events.append(ev)

    event_types = [e.event_type for e in events]
    assert "TOKEN" in event_types
    assert "DONE" in event_types
    full_text = "".join([e.content for e in events if e.content])
    assert "인천대학교 AI 에이전트 챗불이" in full_text
    assert "학사/캠퍼스 생활과 무관한 질문" in full_text


@pytest.mark.asyncio
async def test_e2e_multi_hop_advisor_and_directory():
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

    responses = [
        # Hop 1: Call PORTAL
        {
            "thought": "사용자의 담임교수(지도교수) 성함을 알기 위해 학적 정보를 조회합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_portal_1",
                    "type": "function",
                    "function": {
                        "name": "action_portal_get_academic_record",
                        "arguments": {},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 2: Call DIRECTORY
        {
            "thought": "지도교수가 박문주 교수님으로 확인되었으므로, 교내 전화번호부에서 박문주 교수님의 연락처를 검색합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_dir_2",
                    "type": "function",
                    "function": {
                        "name": "api_searchContacts",
                        "arguments": {"query": "박문주"},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 3: Done with tool calls
        {
            "thought": "박문주 교수님의 연구실 및 연락처 조회가 완료되었습니다. 최종 답변을 작성합니다.",
            "content": "최종 답변 준비 완료",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": "완료"},
        },
    ]

    async def mock_stream_chat(messages):
        yield "학우님의 지도교수님은 **컴퓨터공학부 박문주 교수님**이십니다.\n"
        yield "- 연구실 번호: 032-835-8442"

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

        thinking_contents = [e.thinking for e in events if e.thinking]
        assert any("박문주" in t for t in thinking_contents)


@pytest.mark.asyncio
async def test_e2e_cafeteria_query():
    await tool_registry.initialize_all_tools()
    orchestrator = AgentOrchestrator()
    request = ChatRequest(
        message="오늘 학생식당 점심 메뉴 뭐야?",
        history=[],
    )

    responses = [
        # Hop 1: Call Cafeteria Tool
        {
            "thought": "학생식당의 오늘 메뉴를 확인하기 위해 학식 조회 도구를 호출합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_cafe_1",
                    "type": "function",
                    "function": {
                        "name": "api_getCafeteriaMenu",
                        "arguments": {"cafeteria": "학생식당", "day": 3},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 2: Complete
        {
            "thought": "메뉴 정보 조회가 완료되었습니다.",
            "content": "메뉴 안내 완료",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": "완료"},
        },
    ]

    async def mock_stream_chat(messages):
        yield "오늘 학생식당 중식 메뉴는 **돈까스 & 쌀밥**입니다!"

    with patch("app.orchestrator.engine.llm_client.chat_with_tools", side_effect=responses), \
         patch("app.orchestrator.engine.llm_client.stream_chat", side_effect=mock_stream_chat):

        events = []
        async for ev in orchestrator.run_stream(request):
            events.append(ev)

        event_types = [e.event_type for e in events]
        assert "THINKING" in event_types
        assert "STATUS" in event_types
        assert "TOKEN" in event_types
        assert "DONE" in event_types


@pytest.mark.asyncio
async def test_e2e_library_seats_query():
    await tool_registry.initialize_all_tools()
    orchestrator = AgentOrchestrator()
    request = ChatRequest(
        message="학산도서관 열람실 자리 있어?",
        history=[],
    )

    responses = [
        # Hop 1: Call Library Tool
        {
            "thought": "도서관 열람실 좌석 현황을 확인합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_lib_1",
                    "type": "function",
                    "function": {
                        "name": "api_library_reading_rooms",
                        "arguments": {},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 2: Done
        {
            "thought": "좌석 현황을 확인했습니다.",
            "content": "좌석 안내 완료",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": "완료"},
        },
    ]

    async def mock_stream_chat(messages):
        yield "현재 학산도서관 **제1열람실은 45석**, **제2열람실은 80석**의 여유가 있습니다."

    with patch("app.orchestrator.engine.llm_client.chat_with_tools", side_effect=responses), \
         patch("app.orchestrator.engine.llm_client.stream_chat", side_effect=mock_stream_chat):

        events = []
        async for ev in orchestrator.run_stream(request):
            events.append(ev)

        event_types = [e.event_type for e in events]
        assert "STATUS" in event_types
        assert "CARD" in event_types
        assert "TOKEN" in event_types
        assert "DONE" in event_types


@pytest.mark.asyncio
async def test_e2e_timetable_dispatched_with_fallback_card():
    await tool_registry.initialize_all_tools()
    orchestrator = AgentOrchestrator()
    request = ChatRequest(
        message="포털에서 내 수강신청 내역 조회해줘",
        history=[],
        client_context={
            "isApp": True,
            "portal": {"linked": True},
        },
    )

    responses = [
        # Hop 1: Call Portal Timetable Tool
        {
            "thought": "학교 포털 수강신청 시간표를 확인하기 위해 action_portal_get_student_timetable을 호출합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_tt_1",
                    "type": "function",
                    "function": {
                        "name": "action_portal_get_student_timetable",
                        "arguments": {},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 2: Synthesis
        {
            "thought": "조회 상태를 확인하고 사용자에게 안내합니다.",
            "content": "안내",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": "완료"},
        },
    ]

    async def mock_stream_chat(messages):
        yield "현재 인팁에 동기화된 이번 학기 수강신청 시간표가 등록되어 있지 않습니다. 인팁 앱의 [시간표] 탭에서 시간표를 추가하거나 관리할 수 있어요."

    with patch("app.orchestrator.engine.llm_client.chat_with_tools", side_effect=responses), \
         patch("app.orchestrator.engine.llm_client.stream_chat", side_effect=mock_stream_chat):

        events = []
        async for ev in orchestrator.run_stream(request):
            events.append(ev)

        event_types = [e.event_type for e in events]
        assert "ACTION_REQUIRED" in event_types
        assert "CARD" in event_types
        cards = [e.card for e in events if e.event_type == "CARD" and e.card]
        assert len(cards) >= 1
        assert cards[0].title == "🗓️ 나의 수업 시간표"
        assert "DONE" in event_types


@pytest.mark.asyncio
async def test_e2e_timetable_web_environment_synthesis():
    """Verify that in web environments (isApp=False), the stream never pauses with empty response; it synthesizes tokens and card seamlessly."""
    await tool_registry.initialize_all_tools()
    orchestrator = AgentOrchestrator()
    request = ChatRequest(
        message="이번학기 수강신청내역",
        history=[],
        client_context={
            "isApp": False,
            "portal": {"linked": True},
        },
    )

    responses = [
        # Hop 1: Call Portal Timetable Tool
        {
            "thought": "사용자의 이번 학기 수강신청 내역을 조회하기 위해 action_portal_get_student_timetable 도구를 호출합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_tt_web_1",
                    "type": "function",
                    "function": {
                        "name": "action_portal_get_student_timetable",
                        "arguments": {},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 2: Synthesis
        {
            "thought": "세션에 동기화된 데이터가 없으므로 대안과 함께 최종 답변을 작성합니다.",
            "content": "완료",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": "완료"},
        },
    ]

    async def mock_stream_chat(messages):
        yield "현재 인팁에 동기화된 이번 학기 수강신청 시간표가 등록되어 있지 않습니다. "
        yield "인팁 앱의 [시간표] 탭에서 이번 학기 시간표를 추가하거나 관리해보세요."

    with patch("app.orchestrator.engine.llm_client.chat_with_tools", side_effect=responses), \
         patch("app.orchestrator.engine.llm_client.stream_chat", side_effect=mock_stream_chat):

        events = []
        async for ev in orchestrator.run_stream(request):
            events.append(ev)

        event_types = [e.event_type for e in events]
        # In web environment, ACTION_REQUIRED is not dispatched and stream does NOT stop early!
        assert "ACTION_REQUIRED" not in event_types
        assert "CARD" in event_types
        assert "TOKEN" in event_types
        assert "DONE" in event_types

        tokens = [e.content for e in events if e.event_type == "TOKEN"]
        assert len(tokens) >= 2
        full_text = "".join(tokens)
        assert "수강신청 시간표가 등록되어 있지 않습니다" in full_text


@pytest.mark.asyncio
async def test_e2e_timetable_from_client_context():
    await tool_registry.initialize_all_tools()
    orchestrator = AgentOrchestrator()
    request = ChatRequest(
        message="내 시간표 보여줘",
        history=[],
        client_context={
            "studentTimetable": [
                {
                    "title": "운영체제",
                    "classroom": "공학관 405호",
                    "time": "화 10:00~11:30",
                }
            ]
        },
    )

    responses = [
        # Hop 1: Call Portal Timetable Tool
        {
            "thought": "시간표 데이터를 확인합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_tt_2",
                    "type": "function",
                    "function": {
                        "name": "action_portal_get_student_timetable",
                        "arguments": {},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 2: Synthesis
        {
            "thought": "시간표 확인 완료",
            "content": "시간표 안내",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": "완료"},
        },
    ]

    async def mock_stream_chat(messages):
        yield "이번 학기 수강 신청된 강의는 **운영체제(화 10:00~11:30, 공학관 405호)**입니다."

    with patch("app.orchestrator.engine.llm_client.chat_with_tools", side_effect=responses), \
         patch("app.orchestrator.engine.llm_client.stream_chat", side_effect=mock_stream_chat):

        events = []
        async for ev in orchestrator.run_stream(request):
            events.append(ev)

        event_types = [e.event_type for e in events]
        assert "CARD" in event_types
        cards = [e.card for e in events if e.event_type == "CARD" and e.card]
        assert len(cards) >= 1
        assert cards[0].title == "🗓️ 나의 수업 시간표"
        assert "ACTION_REQUIRED" not in event_types
        assert "DONE" in event_types

