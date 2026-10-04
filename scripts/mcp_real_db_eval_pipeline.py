"""
MCP Standard Compliance & Live DB Ground-Truth Evaluation Pipeline.

Validates inu-agent-core against live INU Portal DB (portal-dev.inuappcenter.kr)
across diverse realistic scenarios:
1. Course Offering detailed attributes (gradeEvaluationName, capacity, isuName, hyName, credit)
2. Course Offering credit and target grade
3. Department name course offerings (e.g. 컴퓨터공학부)
4. Cafeteria menu inquiries
5. Campus Weather inquiries
6. Department/Office contact directory
7. University Notices
8. Central Clubs
9. Lost & Found Property
10. Composite multi-intent query (Course + Cafeteria)
"""
import sys
import os
import asyncio
import json
from typing import List, Dict, Any, Optional

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Add repo root to pythonpath
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.orchestrator.engine import AgentOrchestrator
from app.llm.schemas import ChatRequest, AgentStreamEvent
from app.core.logging import logger

ADMIN_DEV_TOKEN = (
    "eyJhbGciOiJIUzI1NiJ9."
    "eyJzdWIiOiIzIiwicm9sZXMiOlsiUk9MRV9BRE1JTiJdLCJpYXQiOjE3OTEwMDMyMDIsImV4cCI6MTc5MTA4OTYwMn0."
    "9oyGF4ahhWa7n0Me_kxybevXtOc7ttm4zSaOU-aG3bU"
)


class EvalScenario:
    def __init__(
        self,
        id: str,
        name: str,
        query: str,
        expected_tool_categories: List[str],
        expected_ground_truth_keywords: List[str],
        prohibited_phrases: Optional[List[str]] = None,
        notes: str = "",
    ):
        self.id = id
        self.name = name
        self.query = query
        self.expected_tool_categories = expected_tool_categories
        self.expected_ground_truth_keywords = expected_ground_truth_keywords
        self.prohibited_phrases = prohibited_phrases or [
            "기능을 지원하지 않습니다",
            "직접 조회하여 안내해 드리는 기능을 지원하지 않습니다",
            "제공하지 않는 기능입니다",
            "교내 시스템 일시 오류",
        ]

        self.notes = notes


# 10 Ground-Truth Test Scenarios Grounded in portal-dev DB
SCENARIOS: List[EvalScenario] = [
    EvalScenario(
        id="SCENARIO_01_COURSE_EVAL_TYPE",
        name="모바일소프트웨어 성적 평가 방식 (상대/절대평가)",
        query="모바일소프트웨어 수업 평가 방식이 상대평가야 절대평가야?",
        expected_tool_categories=["COURSE"],
        expected_ground_truth_keywords=["상대평가"],
        prohibited_phrases=["기능을 지원하지 않습니다", "직접 조회하여 안내해 드리는 기능을 지원하지 않습니다"],
        notes="API 응답의 gradeEvaluationName이 무손실 릴레이되어 정확히 식별되는지 검증",
    ),
    EvalScenario(
        id="SCENARIO_02_COURSE_GRADE_CREDIT",
        name="모바일소프트웨어 학점 및 학년 확인",
        query="모바일소프트웨어 수업 몇 학점이고 대상 학년이 어떻게 돼?",
        expected_tool_categories=["COURSE"],
        expected_ground_truth_keywords=["3학점"],
        notes="credit: 3(3학점) 및 hyName(전학년) 등 실제 DB 필드가 온전하게 반영되는지 검증",
    ),

    EvalScenario(
        id="SCENARIO_03_DEPT_COURSES",
        name="컴퓨터공학부 개설과목 조회",
        query="컴퓨터공학부에 개설된 전공과목들 알려줘",
        expected_tool_categories=["COURSE"],
        expected_ground_truth_keywords=["컴퓨터공학부"],
        notes="deptName: 컴퓨터공학부 필터링 및 강의 목록 제공 검증",
    ),
    EvalScenario(
        id="SCENARIO_04_CAFETERIA_MENU",
        name="학생식당 메뉴 정보",
        query="오늘 학생식당 학식 메뉴 뭐야?",
        expected_tool_categories=["CAFETERIA"],
        expected_ground_truth_keywords=["학생식당"],
        notes="학식 API 연동 및 메뉴/운영안내 정상 반환 검증",
    ),
    EvalScenario(
        id="SCENARIO_05_CAMPUS_WEATHER",
        name="캠퍼스 실시간 날씨",
        query="송도 캠퍼스 지금 날씨랑 기온 어때?",
        expected_tool_categories=["WEATHER"],
        expected_ground_truth_keywords=["송도", "기온", "°C"],
        notes="기상청 실시간 날씨 데이터 수신 및 친절 안내 검증",
    ),
    EvalScenario(
        id="SCENARIO_06_DEPT_DIRECTORY",
        name="학과 사무실 연락처 조회",
        query="컴퓨터공학부 과사 전화번호나 위치 알려줘",
        expected_tool_categories=["DIRECTORY", "SEARCH"],
        expected_ground_truth_keywords=["032-835", "7호관"],
        notes="전화번호부 DB 또는 포털 연락처에서 032-835 번호 추출 검증",
    ),
    EvalScenario(
        id="SCENARIO_07_UNIVERSITY_NOTICE",
        name="학교 공지사항 검색",
        query="등록금 관련 학교 공지사항 찾아줘",
        expected_tool_categories=["NOTICE", "SEARCH"],
        expected_ground_truth_keywords=["등록금"],
        notes="공지사항 도구 호출 및 마크다운 링크 정상 포함 검증",
    ),
    EvalScenario(
        id="SCENARIO_08_CENTRAL_CLUB",
        name="교내 동아리 목록 조회",
        query="인천대학교 중앙동아리 목록 알려줘",
        expected_tool_categories=["CLUB", "SEARCH"],
        expected_ground_truth_keywords=["동아리"],
        notes="동아리 API 정상 호출 및 목록 제공 검증",
    ),
    EvalScenario(
        id="SCENARIO_09_LOST_PROPERTY",
        name="학내 분실물 습득 내역",
        query="최근에 캠퍼스에서 습득된 분실물 있어?",
        expected_tool_categories=["LOST_PROPERTY", "SEARCH"],
        expected_ground_truth_keywords=["분실물"],
        notes="분실물 API 정상 호출 및 내역 안내 검증",
    ),
    EvalScenario(
        id="SCENARIO_10_COMPOSITE_MULTI_INTENT",
        name="복합 질의 (모바일소프트웨어 평가방식 + 오늘 학생식당)",
        query="모바일소프트웨어 평가방식이랑 오늘 학생식당 점심 알려줘",
        expected_tool_categories=["COURSE", "CAFETERIA"],
        expected_ground_truth_keywords=["상대평가", "학생식당"],
        notes="두 개의 독립 도구를 모두 호출하여 종합 답변 생성 검증",
    ),
]


async def evaluate_single_scenario(
    orchestrator: AgentOrchestrator,
    scenario: EvalScenario,
) -> Dict[str, Any]:
    req = ChatRequest(
        message=scenario.query,
        client="INTIP",
        client_context={
            "auth": ADMIN_DEV_TOKEN,
            "authorization": f"Bearer {ADMIN_DEV_TOKEN}",
            "accessToken": ADMIN_DEV_TOKEN,
        },
    )

    events: List[AgentStreamEvent] = []
    collected_tokens: List[str] = []
    executed_tools: List[str] = []
    executed_categories: List[str] = []

    try:
        async for event in orchestrator.run_stream(req):
            events.append(event)
            if event.event_type == "TOKEN" and event.content:
                collected_tokens.append(event.content)
            elif event.event_type == "STATUS":
                st_cat = getattr(event, "status_category", None)
                st_title = getattr(event, "status_title", None)
                if st_cat:
                    executed_categories.append(st_cat)
                if st_title:
                    executed_tools.append(st_title)


        full_answer = "".join(collected_tokens).strip()

        # Validation Checks
        errors = []

        # 1. Expected tool category called
        for exp_cat in scenario.expected_tool_categories:
            if exp_cat not in executed_categories:
                # If alternative category exists (e.g. SEARCH instead of NOTICE)
                if not any(cat in executed_categories for cat in scenario.expected_tool_categories):
                    errors.append(f"필수 도구 카테고리 누락: {exp_cat} (실행된 카테고리: {executed_categories})")

        # 2. Prohibited phrases check
        for phrase in scenario.prohibited_phrases:
            if phrase in full_answer:
                errors.append(f"금지된 거절/오류 문구 검출: '{phrase}'")

        # 3. Ground truth keyword check
        for kw in scenario.expected_ground_truth_keywords:
            if kw not in full_answer:
                errors.append(f"필수 정답 키워드 누락: '{kw}'")

        is_passed = len(errors) == 0

        return {
            "id": scenario.id,
            "name": scenario.name,
            "query": scenario.query,
            "passed": is_passed,
            "errors": errors,
            "executed_categories": list(set(executed_categories)),
            "full_answer_snippet": full_answer[:250] + ("..." if len(full_answer) > 250 else ""),
        }

    except Exception as ex:
        return {
            "id": scenario.id,
            "name": scenario.name,
            "query": scenario.query,
            "passed": False,
            "errors": [f"실행 예외 발생: {str(ex)}"],
            "executed_categories": executed_categories,
            "full_answer_snippet": "",
        }


async def main():
    print("=" * 70)
    print("🚀 MCP 표준 준수 및 실 DB 기반 다상황 자가검증 파이프라인 가동")
    print("=" * 70)

    orchestrator = AgentOrchestrator()
    total = len(SCENARIOS)
    passed_count = 0
    results = []

    for idx, sc in enumerate(SCENARIOS, 1):
        print(f"\n[{idx}/{total}] 검증 중: {sc.name} ({sc.id})")
        print(f"  질의: '{sc.query}'")
        res = await evaluate_single_scenario(orchestrator, sc)
        results.append(res)

        if res["passed"]:
            passed_count += 1
            print(f"  결과: ✅ PASS")
            print(f"  실행 카테고리: {res['executed_categories']}")
            print(f"  응답 요약: {res['full_answer_snippet'].replace(chr(10), ' ')}")
        else:
            print(f"  결과: ❌ FAIL")
            print(f"  오류 목록: {res['errors']}")
            print(f"  실행 카테고리: {res['executed_categories']}")
            print(f"  응답 요약: {res['full_answer_snippet'].replace(chr(10), ' ')}")

    print("\n" + "=" * 70)
    print(f"📊 검증 리포트: 총 {total}개 시나리오 중 {passed_count}개 통과 (성공률: {passed_count/total*100:.1f}%)")
    print("=" * 70)

    # Save detailed JSON report
    report_path = os.path.join(REPO_ROOT, "scripts", "mcp_eval_pipeline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"📄 상세 리포트 저장 완료: {report_path}")

    if passed_count < total:
        print("\n⚠️ 일부 시나리오 검증 실패. 원인을 분석하고 보정합니다.")
        sys.exit(1)
    else:
        print("\n🎉 모든 실 DB 기반 MCP 시나리오가 100% 통과되었습니다!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
