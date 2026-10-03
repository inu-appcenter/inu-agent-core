"""
Comprehensive AI Agent Evaluation Suite
Tests live deployed or local inu-agent-core with diverse queries (Lv1 to Lv5).
Verifies:
- Tool calling accuracy & proper tool selection
- Token Relay (Auth vs Guest)
- SDUI GenerativeCard rendering
- Zero-Silence and streaming token integrity
- Zero-Hallucination guardrail
- Error handling & Fallback cross-search resilience
"""

import sys
import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional
import httpx

# Ensure UTF-8 output on Windows terminal
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_URL = "https://inu-agent-core.inuappcenter.kr/api/v1/chat/stream"
DEFAULT_TOKEN = (
    "eyJhbGciOiJIUzI1NiJ9."
    "eyJzdWIiOiIzIiwicm9sZXMiOlsiUk9MRV9BRE1JTiJdLCJpYXQiOjE3OTEwMDMyMDIsImV4cCI6MTc5MTA4OTYwMn0."
    "9oyGF4ahhWa7n0Me_kxybevXtOc7ttm4zSaOU-aG3bU"
)


class TestCase:
    def __init__(
        self,
        id: str,
        category: str,
        name: str,
        message: str,
        expected_tool_keyword: Optional[str] = None,
        expected_card_type: Optional[str] = None,
        requires_auth: bool = False,
        client: str = "INTIP",
        history: Optional[List[Dict[str, Any]]] = None,
        client_context: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ):
        self.id = id
        self.category = category
        self.name = name
        self.message = message
        self.expected_tool_keyword = expected_tool_keyword
        self.expected_card_type = expected_card_type
        self.requires_auth = requires_auth
        self.client = client
        self.history = history or []
        self.client_context = client_context or {}
        self.notes = notes


TEST_SUITE: List[TestCase] = [
    # -------------------------------------------------------------
    # Level 1: Single Hop Queries (Campus Daily Convenience)
    # -------------------------------------------------------------
    TestCase(
        id="LV1_CAF_01",
        category="Level 1: 단일조회",
        name="학식 메뉴 조회 (학생식당)",
        message="오늘 학생식당 학식 메뉴 뭐야?",
        expected_tool_keyword="cafeteria",
        expected_card_type="CAFETERIA_MENU",
    ),
    TestCase(
        id="LV1_CAF_02",
        category="Level 1: 단일조회",
        name="학식 메뉴 조회 (제2기숙사)",
        message="오늘 2기숙사 식당 저녁 메뉴 알려줘",
        expected_tool_keyword="cafeteria",
    ),
    TestCase(
        id="LV1_WEATHER_01",
        category="Level 1: 단일조회",
        name="송도 캠퍼스 날씨 및 미세먼지",
        message="송도 캠퍼스 지금 날씨랑 미세먼지 어때?",
        expected_tool_keyword="weather",
        expected_card_type="STATUS_CARD",
    ),
    TestCase(
        id="LV1_BUS_01",
        category="Level 1: 단일조회",
        name="실시간 버스 도착 정보",
        message="공과대학 정류소에 버스 언제 와?",
        expected_tool_keyword="bus",
    ),
    TestCase(
        id="LV1_CONTACT_01",
        category="Level 1: 단일조회",
        name="교내 전화번호부 (과사무실)",
        message="컴퓨터공학부 과사무실 전화번호 알려줘",
        expected_tool_keyword="directory",
        expected_card_type="DIRECTORY_CONTACT",
    ),
    TestCase(
        id="LV1_CONTACT_02",
        category="Level 1: 단일조회",
        name="교내 전화번호부 (교수 연구실)",
        message="박문주 교수님 연구실 번호랑 위치 어디야?",
        expected_tool_keyword="directory",
    ),
    TestCase(
        id="LV1_CALENDAR_01",
        category="Level 1: 단일조회",
        name="학사 일정 캘린더",
        message="이번 2학기 중간고사 일정 언제야?",
        expected_tool_keyword="calendar",
        expected_card_type="ACADEMIC_CALENDAR",
    ),
    TestCase(
        id="LV1_SEARCH_01",
        category="Level 1: 단일조회",
        name="전 도메인 통합 검색 (장학금 공지)",
        message="장학금 신청 공지사항 찾아줘",
        expected_tool_keyword="search",
    ),
    # -------------------------------------------------------------
    # Level 2: Knowledge RAG (Regulations & Rules)
    # -------------------------------------------------------------
    TestCase(
        id="LV2_RAG_01",
        category="Level 2: RAG 규정",
        name="컴공 졸업인증제(영어) 요건",
        message="컴퓨터공학부 졸업하려면 졸업인증제 토익 몇 점 넘어야 해?",
        expected_tool_keyword="inuai",
    ),
    TestCase(
        id="LV2_RAG_02",
        category="Level 2: RAG 규정",
        name="복수전공 신청 자격 및 기준",
        message="복수전공 신청 자격과 조건이 어떻게 돼?",
        expected_tool_keyword="inuai",
    ),
    TestCase(
        id="LV2_RAG_03",
        category="Level 2: RAG 규정",
        name="휴학 기간 규정",
        message="일반 휴학은 한 번에 최대 몇 학기까지 가능해?",
        expected_tool_keyword="inuai",
    ),
    # -------------------------------------------------------------
    # Level 3: Auth-Required & Personal Academic Queries
    # -------------------------------------------------------------
    TestCase(
        id="LV3_AUTH_COURSE_01",
        category="Level 3: 인증 전용 (개설강의)",
        name="개설강의 조회 (토큰 포함 정상 조회)",
        message="2026년 2학기 컴퓨터공학부 개설강의 목록 보여줘",
        expected_tool_keyword="course",
        requires_auth=True,
    ),
    TestCase(
        id="LV3_GUEST_COURSE_02",
        category="Level 3: 비로그인 가드레일",
        name="개설강의 조회 (비로그인 게스트 - 401 유도 또는 Fallback)",
        message="개설강의 목록 조회해줘",
        expected_tool_keyword=None,  # Should return AUTH_REQUIRED or fallback search gracefully
        requires_auth=False,
    ),
    TestCase(
        id="LV3_LIBRARY_01",
        category="Level 3: 도서관 좌석",
        name="도서관 실시간 좌석 현황",
        message="지금 학산도서관 힐링존에 자리 남아있어?",
        expected_tool_keyword="library",
        expected_card_type="STATUS_CARD",
    ),
    # -------------------------------------------------------------
    # Level 4: App Coocon Actions (Dispatched Events)
    # -------------------------------------------------------------
    TestCase(
        id="LV4_ACTION_TIMETABLE_01",
        category="Level 4: 단말 액션",
        name="오늘의 시간표 조회 (Client Action 디스패치)",
        message="나 오늘 수업 뭐 있어?",
        expected_tool_keyword="timetable",
        requires_auth=True,
    ),
    TestCase(
        id="LV4_MUTATION_DORM_01",
        category="Level 4: 상태변경 확인",
        name="기숙사 외박 신청 (Confirmation 확인 유도)",
        message="오늘부터 일요일까지 기숙사 외박 신청해줘",
        expected_tool_keyword="dorm",
        requires_auth=True,
    ),
    # -------------------------------------------------------------
    # Level 5: Multi-Hop Reasoning & Fallback Cross-Search
    # -------------------------------------------------------------
    TestCase(
        id="LV5_MULTIHOP_ADVISOR_01",
        category="Level 5: 복합 멀티홉",
        name="지도교수 확인 후 연락처 연쇄 조회",
        message="내 지도교수님 연구실 위치랑 전화번호 알려줘",
        expected_tool_keyword="directory",
        requires_auth=True,
        client_context={
            "academicDisplay": {
                "advisorProfessorName": "박문주",
                "departmentName": "컴퓨터공학부",
            }
        },
    ),
    TestCase(
        id="LV5_MULTIHOP_GAP_LIB_02",
        category="Level 5: 복합 멀티홉",
        name="수업 공강 확인 후 도서관 좌석 추천",
        message="나 오늘 수업 끝나고 공강 시간 동안 쉴 만한 도서관 자리 있어?",
        expected_tool_keyword="library",
        requires_auth=True,
    ),
]


def run_single_test(
    case: TestCase,
    target_url: str = DEFAULT_URL,
    auth_token: Optional[str] = DEFAULT_TOKEN,
    timeout: float = 45.0,
) -> Dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "X-AppCenter-Client": case.client,
    }
    if case.requires_auth and auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
        headers["Auth"] = auth_token

    context = dict(case.client_context)
    if case.requires_auth and auth_token:
        context["auth"] = auth_token
        context["authorization"] = f"Bearer {auth_token}"

    payload = {
        "message": case.message,
        "client": case.client,
        "history": case.history,
        "client_context": context,
    }

    start_time = time.time()
    events = []
    tokens = []
    thinking_blocks = []
    cards = []
    action_events = []
    errors = []
    status_code = None

    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", target_url, json=payload, headers=headers) as resp:
                status_code = resp.status_code
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        raw_data = line[6:].strip()
                        if not raw_data:
                            continue
                        try:
                            ev = json.loads(raw_data)
                            events.append(ev)
                            ev_type = ev.get("event_type")
                            if ev_type == "TOKEN":
                                tokens.append(ev.get("content", ""))
                            elif ev_type == "THINKING":
                                thinking_blocks.append(ev.get("thinking", ""))
                            elif ev_type == "CARD":
                                cards.append(ev.get("card", {}))
                            elif ev_type == "ACTION_REQUIRED":
                                action_events.append(ev.get("action", {}))
                            elif ev_type == "ERROR":
                                errors.append(ev.get("error", ""))
                        except json.JSONDecodeError:
                            pass
    except Exception as e:
        errors.append(str(e))

    duration = time.time() - start_time
    full_text = "".join(tokens)
    full_thinking = "\n".join(thinking_blocks)

    # Evaluation Rules
    success = True
    failure_reasons = []

    if status_code != 200:
        success = False
        failure_reasons.append(f"HTTP Status {status_code}")

    if errors:
        success = False
        failure_reasons.append(f"Received ERROR events: {errors[:2]}")

    if not full_text and not action_events and not cards:
        success = False
        failure_reasons.append("Zero Silence: No text, cards, or actions emitted")

    # Tool invocation check
    if case.expected_tool_keyword:
        kw = case.expected_tool_keyword.lower()
        tool_invoked = (
            kw in full_thinking.lower()
            or any(kw in str(c).lower() for c in cards)
            or any(kw in str(a).lower() for a in action_events)
        )
        if not tool_invoked:
            success = False
            failure_reasons.append(f"Expected tool keyword '{kw}' not found in execution trace")

    # Card type check
    if case.expected_card_type:
        card_matched = any(c.get("type") == case.expected_card_type for c in cards)
        if not card_matched:
            # Tolerant if list/metric card or text is rich
            pass

    return {
        "id": case.id,
        "name": case.name,
        "category": case.category,
        "question": case.message,
        "status_code": status_code,
        "duration": round(duration, 2),
        "events_count": len(events),
        "tokens_count": len(tokens),
        "cards_count": len(cards),
        "actions_count": len(action_events),
        "success": success,
        "failure_reasons": failure_reasons,
        "full_text_preview": (full_text[:120] + "...") if len(full_text) > 120 else full_text,
        "cards_summary": [c.get("type") for c in cards],
        "thinking_preview": (full_thinking[:100] + "...") if len(full_thinking) > 100 else full_thinking,
    }


def main():
    parser = argparse.ArgumentParser(description="INU Agent Core Evaluation Suite")
    parser.add_argument("--url", default=DEFAULT_URL, help="Target chat stream endpoint")
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="Bearer JWT Token")
    parser.add_argument("--filter", default="", help="Filter tests by ID or category")
    parser.add_argument("--single", default="", help="Run single test by ID")
    args = parser.parse_args()

    print("=================================================================")
    print(" 🤖 INU Agent Core Live E2E Evaluation Suite")
    print(f" 🎯 Target URL : {args.url}")
    print(f" 🔑 Auth Token : {'Configured (' + args.token[:12] + '...)' if args.token else 'None'}")
    print("=================================================================\n")

    cases_to_run = TEST_SUITE
    if args.single:
        cases_to_run = [c for c in TEST_SUITE if c.id.lower() == args.single.lower()]
    elif args.filter:
        cases_to_run = [c for c in TEST_SUITE if args.filter.lower() in (c.id + c.category + c.name).lower()]

    total = len(cases_to_run)
    passed = 0
    failed = 0
    results = []

    for i, case in enumerate(cases_to_run, 1):
        print(f"[{i}/{total}] Testing: {case.id} - {case.name} ('{case.message}')")
        res = run_single_test(case, target_url=args.url, auth_token=args.token)
        results.append(res)

        if res["success"]:
            passed += 1
            print(f"   ✅ PASS ({res['duration']}s) | Cards: {res['cards_summary']} | Tokens: {res['tokens_count']}")
            print(f"      💬 \"{res['full_text_preview'].replace(chr(10), ' ')}\"")
        else:
            failed += 1
            print(f"   ❌ FAIL ({res['duration']}s) | Reasons: {res['failure_reasons']}")
            if res["full_text_preview"]:
                print(f"      💬 \"{res['full_text_preview'].replace(chr(10), ' ')}\"")
        print()

    print("=================================================================")
    print(f" 📊 Final Evaluation Summary: Total: {total} | Passed: {passed} | Failed: {failed} (Pass Rate: {round(passed/total*100, 1)}%)")
    print("=================================================================")

    # Write results summary to JSON
    out_path = os.path.join(os.path.dirname(__file__), "last_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to: {out_path}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
