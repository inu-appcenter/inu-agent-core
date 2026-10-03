import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.llm.schemas import ChatActionCallbackRequest, GenerativeCard

client = TestClient(app)


@pytest.mark.asyncio
async def test_action_callback_timetable_success():
    """
    Test on-demand P2P action callback with timetable data.
    Verifies that the callback generates a TIMETABLE card and streams final tokens.
    """
    callback_payload = {
        "action_id": "act_portal_get_student_timetable_6c5a252d",
        "session_id": "session_test_123",
        "success": True,
        "original_message": "포털에서 내 수강신청 내역 조회해줘",
        "client": "INTIP",
        "data": {
            "courses": [
                {
                    "courseName": "운영체제",
                    "professor": "김교수",
                    "time": "화 10:00~11:30",
                    "classroom": "공학관 405호",
                },
                {
                    "courseName": "알고리즘",
                    "professor": "이교수",
                    "time": "수 13:00~15:00",
                    "classroom": "정보기술관 201호",
                },
            ]
        },
    }

    with patch("app.orchestrator.engine.llm_client.stream_chat") as mock_stream:
        async def mock_token_generator(messages):
            yield "이번 "
            yield "학기 "
            yield "수강신청 "
            yield "내역입니다."

        mock_stream.side_effect = mock_token_generator

        response = client.post("/api/v1/chat/action-callback", json=callback_payload)
        assert response.status_code == 200
        text = response.text
        assert "CARD" in text
        assert "운영체제" in text or "나의 수업 시간표" in text
        assert "이번 학기 수강신청 내역입니다." in text or "TOKEN" in text
        assert "DONE" in text


@pytest.mark.asyncio
async def test_action_callback_academic_advisor_multi_hop():
    """
    Test on-demand P2P action callback with academic record containing advisor professor.
    Verifies that it automatically chains to api_directory for contact details.
    """
    callback_payload = {
        "action_id": "act_portal_get_academic_record_12345",
        "session_id": "session_test_advisor",
        "success": True,
        "original_message": "내 지도교수님 연락처좀 알려줘",
        "client": "INTIP",
        "data": {
            "studentName": "배현준",
            "studentId": "202001518",
            "department": "컴퓨터공학부",
            "advisorProfessorName": "박문주",
        },
    }

    with patch("app.orchestrator.engine.llm_client.stream_chat") as mock_stream, \
         patch("app.tools.registry.tool_registry.get_tool") as mock_get_tool:

        async def mock_token_gen(messages):
            yield "박문주 교수님의 연락처를 확인했습니다."

        mock_stream.side_effect = mock_token_gen

        mock_dir_tool = AsyncMock()
        mock_dir_tool.execute.return_value = {
            "contacts": [
                {
                    "name": "박문주",
                    "dept": "컴퓨터공학부",
                    "telephone": "032-835-8400",
                    "email": "mjpark@inu.ac.kr",
                    "location": "7호관 410호",
                }
            ]
        }
        mock_get_tool.return_value = mock_dir_tool

        response = client.post("/api/v1/chat/action-callback", json=callback_payload)
        assert response.status_code == 200
        text = response.text
        assert "CARD" in text
        assert "박문주" in text
        assert "DONE" in text


@pytest.mark.asyncio
async def test_action_callback_failure_handling():
    """
    Test on-demand P2P action callback when client reports failure (e.g. ERP session expired).
    """
    callback_payload = {
        "action_id": "act_portal_get_student_timetable_failed",
        "session_id": "session_fail_123",
        "success": False,
        "error_message": "포털 로그인 세션이 만료되었습니다.",
        "original_message": "포털에서 내 시간표 조회해줘",
        "client": "INTIP",
    }

    response = client.post("/api/v1/chat/action-callback", json=callback_payload)
    assert response.status_code == 200
    text = response.text
    assert "포털 로그인 세션이 만료되었습니다." in text
    assert "DONE" in text
