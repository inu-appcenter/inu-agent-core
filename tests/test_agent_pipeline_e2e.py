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
        assert "CARD" in event_types
        assert "TOKEN" in event_types
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


@pytest.mark.asyncio
async def test_e2e_fallback_cross_search_when_directory_empty():
    """Verify that when api_directory returns empty (0 results), the agent does NOT prematurely give up;
    it deterministically falls back to unifiedSearch to discover contact info from campus web pages and notices."""
    await tool_registry.initialize_all_tools()
    orchestrator = AgentOrchestrator()
    request = ChatRequest(
        message="정보기술대학 행정실 연락처 뭐야?",
        history=[],
    )

    responses = [
        # Hop 1: Model calls api_searchContacts (DIRECTORY) with '정보기술대학 행정실'
        {
            "thought": "정보기술대학 행정실의 연락처를 확인하기 위해 교내 전화번호부(api_searchContacts)를 호출합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_dir_empty_1",
                    "type": "function",
                    "function": {
                        "name": "api_searchContacts",
                        "arguments": {"query": "정보기술대학 행정실"},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 2: Model returns empty tool_calls -> Orchestrator Deterministic Multi-Hop kicks in and chains to SEARCH!
        {
            "thought": "전화번호부에서 결과를 찾지 못했습니다.",
            "content": "",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 3: After auto SEARCH completes, model synthesizes final answer
        {
            "thought": "통합 검색 결과에서 정보기술대학 교학실 연락처를 확인했습니다. 사용자에게 안내합니다.",
            "content": "안내 완료",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": "완료"},
        },
    ]

    async def mock_stream_chat(messages):
        yield "인천대학교 **정보기술대학 교학실(행정실)** 연락처 및 위치 안내입니다:\n"
        yield "- 📞 전화번호: 032-835-8900\n"
        yield "- 📍 위치: 송도캠퍼스 7호관(정보기술대학) 107호"

    # Mock tool execution: api_searchContacts returns empty (0 results), unifiedSearch returns directory/contact hits
    dir_tool = tool_registry.get_tool("api_searchContacts") or tool_registry.get_tool("api_directory")
    search_tool = tool_registry.get_tool("unifiedSearch") or tool_registry.get_tool("api_unified_search")

    orig_dir_exec = dir_tool.execute
    orig_search_exec = search_tool.execute

    async def mock_dir_execute(args, ctx):
        return {"summary": "일치하는 교직원 또는 부서를 찾지 못했습니다.", "data": {"contents": []}}

    async def mock_search_execute(args, ctx):
        return {
            "summary": "정보기술대학 교학실 연락처 (032-835-8900, 7호관 107호)",
            "rawData": {
                "totalCount": 1,
                "directory": {
                    "items": [
                        {
                            "name": "정보기술대학 교학실",
                            "affiliation": "정보기술대학",
                            "phoneNumber": "032-835-8900",
                            "position": "행정실",
                        }
                    ]
                }
            }
        }

    with patch.object(dir_tool, "execute", side_effect=mock_dir_execute), \
         patch.object(search_tool, "execute", side_effect=mock_search_execute), \
         patch("app.orchestrator.engine.llm_client.chat_with_tools", side_effect=responses), \
         patch("app.orchestrator.engine.llm_client.stream_chat", side_effect=mock_stream_chat):

        events = []
        async for ev in orchestrator.run_stream(request):
            events.append(ev)

        event_types = [e.event_type for e in events]
        assert "THINKING" in event_types
        assert "STATUS" in event_types
        assert "TOKEN" in event_types
        assert "DONE" in event_types

        # Verify fallback chaining thought was emitted
        thinking_texts = [e.thinking for e in events if e.event_type == "THINKING" and e.thinking]
        has_fallback_thinking = any("통합 검색" in t for t in thinking_texts)
        assert has_fallback_thinking, f"Expected fallback thinking mentioning 통합 검색, got: {thinking_texts}"

        # Verify final text tokens contain phone number
        tokens = [e.content for e in events if e.event_type == "TOKEN" and e.content]
        full_text = "".join(tokens)
        assert "032-835-8900" in full_text
        assert "정보기술대학" in full_text


@pytest.mark.asyncio
async def test_e2e_zero_silence_token_emission_in_p2p_action_wait():
    """Verify that when a client P2P action (e.g. academic record) is dispatched in Native App,
    the agent NEVER terminates in silence (0 tokens); it immediately streams an informative token
    explaining that secure on-device retrieval is underway."""
    await tool_registry.initialize_all_tools()
    orchestrator = AgentOrchestrator()
    request = ChatRequest(
        message="내 성적 및 취득학점 조회해줘",
        history=[],
        client_context={
            "isApp": True,
            "portal": {"linked": True},
        },
    )

    responses = [
        # Hop 1: Model calls action_portal_get_academic_record
        {
            "thought": "사용자의 학적 및 성적을 조회하기 위해 action_portal_get_academic_record를 호출합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_acad_p2p_1",
                    "type": "function",
                    "function": {
                        "name": "action_portal_get_academic_record",
                        "arguments": {},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
    ]

    with patch("app.orchestrator.engine.llm_client.chat_with_tools", side_effect=responses):
        events = []
        async for ev in orchestrator.run_stream(request):
            events.append(ev)

        event_types = [e.event_type for e in events]
        assert "ACTION_REQUIRED" in event_types
        assert "STATUS" in event_types
        assert "TOKEN" in event_types
        assert "DONE" in event_types

        # Verify token content informs user about P2P secure retrieval
        tokens = [e.content for e in events if e.event_type == "TOKEN" and e.content]
        assert len(tokens) >= 1
        full_text = "".join(tokens)
        assert "단말기 보안 영역" in full_text
        assert "잠시만 기다려 주세요" in full_text


@pytest.mark.asyncio
async def test_e2e_fallback_cross_search_when_notice_empty():
    """Verify that when notice search returns empty, the agent autonomously falls back to unified search."""
    await tool_registry.initialize_all_tools()
    orchestrator = AgentOrchestrator()
    request = ChatRequest(
        message="수강신청 관련 공지 검색해줘",
        history=[],
    )

    responses = [
        # Hop 1: Notice tool returns empty
        {
            "thought": "수강신청 관련 공지사항을 확인하기 위해 api_notice를 호출합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_notice_empty_1",
                    "type": "function",
                    "function": {
                        "name": "api_notice",
                        "arguments": {"query": "수강신청"},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 2: Model returns no tool calls -> Orchestrator chains to unifiedSearch
        {
            "thought": "공지사항 검색 결과가 없습니다.",
            "content": "",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 3: Model synthesizes final response
        {
            "thought": "통합 검색 결과를 바탕으로 수강신청 일정과 공지를 안내합니다.",
            "content": "안내 완료",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": "완료"},
        },
    ]

    async def mock_stream_chat(messages):
        yield "2026학년도 수강신청 일정 및 공지 안내입니다."

    notice_tool = tool_registry.get_tool("api_notice")
    search_tool = tool_registry.get_tool("unifiedSearch") or tool_registry.get_tool("api_unified_search")

    async def mock_notice_exec(args, ctx):
        return {"summary": "조회된 공지사항이 없습니다.", "items": []}

    async def mock_search_exec(args, ctx):
        return {
            "summary": "수강신청 공지 (2026-1학기 수강신청 기간: 2월 10일 ~ 2월 14일)",
            "rawData": {"totalCount": 1, "notices": {"items": [{"title": "2026학년도 1학기 수강신청 안내"}]}}
        }

    with patch.object(notice_tool, "execute", side_effect=mock_notice_exec), \
         patch.object(search_tool, "execute", side_effect=mock_search_exec), \
         patch("app.orchestrator.engine.llm_client.chat_with_tools", side_effect=responses), \
         patch("app.orchestrator.engine.llm_client.stream_chat", side_effect=mock_stream_chat):

        events = []
        async for ev in orchestrator.run_stream(request):
            events.append(ev)

        event_types = [e.event_type for e in events]
        assert "TOKEN" in event_types
        assert "DONE" in event_types

        # Verify fallback chaining thought
        thinking_texts = [e.thinking for e in events if e.event_type == "THINKING" and e.thinking]
        assert any("통합 검색" in t for t in thinking_texts)


@pytest.mark.asyncio
async def test_e2e_fallback_cross_search_when_course_offerings_failed():
    """Verify that when api_getCourseOfferings fails (e.g. 401 or empty), the agent autonomously falls back to unifiedSearch(tab='COURSE')."""
    await tool_registry.initialize_all_tools()
    orchestrator = AgentOrchestrator()
    request = ChatRequest(
        message="이번 학기 개설 강의 중에 파이썬이나 인공지능 관련 수업 있어?",
        history=[],
    )

    responses = [
        # Hop 1: Direct course offering tool fails
        {
            "thought": "2026학년도 2학기 개설 강의 중 인공지능 또는 파이썬 관련 수업을 조회하기 위해 api_getCourseOfferings를 호출합니다.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_course_1",
                    "type": "function",
                    "function": {
                        "name": "api_getCourseOfferings",
                        "arguments": {"year": 2026, "term": "SECOND", "keyword": "인공지능"},
                    },
                }
            ],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 2: Model returns no tool calls after direct tool failure -> Orchestrator chains to unifiedSearch
        {
            "thought": "개설 강의 직접 조회에서 결과를 가져오지 못했습니다.",
            "content": "",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": ""},
        },
        # Hop 3: Model synthesizes final response
        {
            "thought": "통합 검색 결과를 바탕으로 개설된 인공지능 관련 강의를 안내합니다.",
            "content": "안내 완료",
            "tool_calls": [],
            "raw_message": {"role": "assistant", "content": "완료"},
        },
    ]

    async def mock_stream_chat(messages):
        yield "이번 학기에 개설된 관련 강의로 '인공지능개론(컴퓨터공학부, 3학점)'이 개설되어 있습니다."

    course_tool = tool_registry.get_tool("api_getCourseOfferings")
    search_tool = tool_registry.get_tool("unifiedSearch") or tool_registry.get_tool("api_unified_search")

    assert course_tool is not None, "api_getCourseOfferings tool should be registered"
    assert search_tool is not None, "unifiedSearch tool should be registered"

    async def mock_course_exec(args, ctx):
        # Simulating 401 or empty error response from portal server
        return {"error": "API request failed with status 401", "details": "Unauthorized"}

    async def mock_search_exec(args, ctx):
        # Verify fallback chained with COURSE tab
        assert args.get("tab") == "COURSE"
        return {
            "summary": "개설강의 검색 결과 (인공지능개론)",
            "rawData": {
                "courses": {
                    "totalCount": 1,
                    "items": [
                        {
                            "courseName": "인공지능개론",
                            "professor": "김교수",
                            "timeRoom": "공7-204",
                            "credit": 3,
                        }
                    ],
                }
            },
        }

    with patch.object(course_tool, "execute", side_effect=mock_course_exec), \
         patch.object(search_tool, "execute", side_effect=mock_search_exec), \
         patch("app.orchestrator.engine.llm_client.chat_with_tools", side_effect=responses), \
         patch("app.orchestrator.engine.llm_client.stream_chat", side_effect=mock_stream_chat):

        events = []
        async for ev in orchestrator.run_stream(request):
            events.append(ev)

        event_types = [e.event_type for e in events]
        assert "THINKING" in event_types
        assert "TOKEN" in event_types
        assert "DONE" in event_types

        # Verify fallback chaining thought was emitted
        thinking_texts = [e.thinking for e in events if e.event_type == "THINKING" and e.thinking]
        assert any("통합 검색" in t for t in thinking_texts)
        assert any("개설 강의" in t for t in thinking_texts)

        tokens = [e.content for e in events if e.event_type == "TOKEN" and e.content]
        full_text = "".join(tokens)
        assert "인공지능개론" in full_text

