"""
Unit tests for Autonomous Parallel Deep Drill-Down Engine (DrillDownEvaluator)
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.orchestrator.drill_down import DrillDownEvaluator, DRILL_DOWN_MAP
from app.tools.registry import tool_registry


def test_should_drill_down_intent_keywords():
    """Verify that user queries asking for details, criteria, or applications trigger drill-down."""
    assert DrillDownEvaluator.should_drill_down("장학금 신청 자격 조건 자세히 알려줘", "unifiedSearch", hop=0) is True
    assert DrillDownEvaluator.should_drill_down("컴퓨터공학과 졸업 요건 세부 내용 확인해줘", "api_getSchoolDepartmentNotices", hop=1) is True
    assert DrillDownEvaluator.should_drill_down("도서관 몇 시에 열어?", "library", hop=0) is False
    # Guardrail: Hop limit > 2 should return False
    assert DrillDownEvaluator.should_drill_down("장학금 신청 방법 알려줘", "unifiedSearch", hop=3) is False


def test_extract_top_candidates_from_unified_search():
    """Verify that extract_top_candidates pulls Top-K candidate IDs from unified search structure."""
    raw_unified = {
        "data": {
            "notices": {
                "items": [
                    {"id": 101, "title": "2026 국가장학금 신청 공고"},
                    {"id": 102, "title": "2026 교내 장학금 선발 계획"},
                    {"id": 103, "title": "이공계 인재 장학금 안내"},
                    {"id": 104, "title": "추가 선발 안내"},
                ]
            },
            "departmentNotices": {
                "items": [
                    {"id": 201, "title": "컴퓨터공학부 졸업작품 제출 양식 안내"}
                ]
            }
        }
    }

    candidates = DrillDownEvaluator.extract_top_candidates("unifiedSearch", "SEARCH", raw_unified, max_k=3)
    assert len(candidates) == 3
    assert candidates[0] == ("notices", 101, "2026 국가장학금 신청 공고")
    assert candidates[1] == ("notices", 102, "2026 교내 장학금 선발 계획")
    assert candidates[2] == ("notices", 103, "이공계 인재 장학금 안내")


def test_extract_top_candidates_from_notice_list():
    """Verify candidate extraction from standard notice endpoint response."""
    raw_notice_list = {
        "contents": [
            {"id": 301, "title": "수강신청 편람 안내"},
            {"id": 302, "title": "수강정정 기간 안내"},
        ]
    }
    candidates = DrillDownEvaluator.extract_top_candidates("api_getAllNotice", "NOTICE", raw_notice_list, max_k=2)
    assert len(candidates) == 2
    assert candidates[0][1] == 301
    assert candidates[1][1] == 302


@pytest.mark.asyncio
async def test_fetch_details_in_parallel_via_gather():
    """Verify parallel asynchronous fetching of details across candidates."""
    mock_detail_tool = MagicMock()
    mock_detail_tool.execute = AsyncMock(return_value={
        "data": {
            "content": "<p>2026년도 장학금 선발 기준: 직전 학기 평점 3.5 이상, 소득 8구간 이하 학생 대상.</p>"
        }
    })

    with patch.object(tool_registry, "get_tool", return_value=mock_detail_tool):
        candidates = [
            ("notices", 101, "2026 국가장학금 신청 공고"),
            ("notices", 102, "2026 교내 장학금 선발 계획"),
        ]
        results = await DrillDownEvaluator.fetch_details_in_parallel(candidates, {})
        assert len(results) == 2
        assert mock_detail_tool.execute.call_count == 2
        assert results[0]["id"] == 101
        assert results[1]["id"] == 102


def test_format_drill_down_summary():
    """Verify formatting and HTML tag cleaning for synthesis LLM ground truth context."""
    documents = [
        {
            "id": 101,
            "title": "국가장학금 1차 신청 안내",
            "data": {
                "content": "<p>신청 마감일: <strong>2026년 11월 30일 18:00</strong>까지입니다. 서류 미제출 시 탈락합니다.</p>"
            }
        },
        {
            "id": 102,
            "title": "교내 성적우수 장학금 안내",
            "data": {
                "content": "선발 기준: 평점평균 3.75 이상 취득자 중 학과 상위 10% 이내"
            }
        }
    ]

    summary = DrillDownEvaluator.format_drill_down_summary(documents)
    assert "총 2건 병렬 확인 완료" in summary
    assert "국가장학금 1차 신청 안내 (ID: 101)" in summary
    assert "2026년 11월 30일 18:00" in summary
    assert "<p>" not in summary  # Cleaned HTML
    assert "교내 성적우수 장학금 안내 (ID: 102)" in summary
    assert "평점평균 3.75" in summary
