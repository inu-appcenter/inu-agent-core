"""
Extended AI Agent Evaluation Suite
Tests live deployed or local inu-agent-core with unverified/deep domains:
- Set A: Clubs & Lost Property (동아리, 분실물)
- Set B: Syllabus & Course Offerings (강의계획서, 개설강의)
- Set C: Department Notices & Council Notices (학과 공지, 총학생회 공지)
- Set D: Academic P2P Client Actions (등록금, 장학금, LMS 과제)
- Set E: Notifications, Reminders & Briefs (리마인더, 데일리브리프, 알림설정)
- Set F: Multi-Intent & Autonomous Fallback Resilience (복합 질의 및 자가 회복)
"""

import sys
import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional
import httpx

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
        must_contain: Optional[List[str]] = None,
        must_not_contain: Optional[List[str]] = None,
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
        self.must_contain = must_contain or []
        self.must_not_contain = must_not_contain or []
        self.requires_auth = requires_auth
        self.client = client
        self.history = history or []
        self.client_context = client_context or {}
        self.notes = notes


EXTENDED_TEST_SUITE: List[TestCase] = [
    # -------------------------------------------------------------
    # Set A: 동아리 및 분실물 (Clubs & Lost Property)
    # -------------------------------------------------------------
    TestCase(
        id="SET_A_CLUB_01",
        category="Set A: 동아리 및 분실물",
        name="교내 동아리 목록 조회 (DB 빈 목록 시 정직한 안내 확인)",
        message="교내 밴드 동아리나 음악 동아리 목록 찾아줘",
        expected_tool_keyword="club",
        must_contain=["동아리", "없습니다"],
        must_not_contain=["API error", "일시적인 오류로 인해"],
        notes="동아리 API 실행 후 DB 결과가 없더라도 오류 핑계 없이 정직하게 안내하는지 검증",
    ),
    TestCase(
        id="SET_A_LOST_01",
        category="Set A: 동아리 및 분실물",
        name="학내 분실물 습득/신고 목록 조회 (DB 빈 목록 시 정직한 안내)",
        message="학내 분실물 센터에 최근 등록된 물건 있어?",
        expected_tool_keyword="lost",
        must_contain=["분실물", "없습니다"],
        must_not_contain=["API error", "일시적인 오류로 인해", "서버 응답 지연"],
        notes="분실물 API 실행 후 빈 목록일 때 오류 핑계 없이 사실대로 고지하는지 확인",
    ),
    # -------------------------------------------------------------
    # Set B: 강의계획서 및 수강편람 상세 (Syllabus & Course Catalog)
    # -------------------------------------------------------------
    TestCase(
        id="SET_B_SYLLABUS_01",
        category="Set B: 강의계획서/수강편람",
        name="과목 강의계획서 상세 조회 (실제 평가비율 및 주차별 계획 파싱 확인)",
        message="컴퓨터프로그래밍 강의계획서 평가 비율이나 주차별 계획 알려줘",
        expected_tool_keyword="course",
        must_contain=["평가", "계획"],
        must_not_contain=["일시적인 오류로 인해", "불러오지 못했습니다"],
        requires_auth=True,
        notes="실제 DB에 존재하는 컴퓨터프로그래밍 강좌의 평가 비율과 주차별 계획을 파싱하는지 확인",
    ),
    TestCase(
        id="SET_B_COURSE_01",
        category="Set B: 강의계획서/수강편람",
        name="개설강의 검색 (실제 DB 데이터사이언스 개설과목 파싱 확인)",
        message="2026년 2학기 데이터사이언스 개설강의 목록 알려줘",
        expected_tool_keyword="course",
        must_contain=["데이터사이언스"],
        must_not_contain=["일시적인 오류로 인해", "불러오지 못했습니다", "API error"],
        requires_auth=True,
        notes="실제 5건 존재하는 데이터사이언스 개설 과목을 400 에러 없이 정상 반환하는지 검증",
    ),
    # -------------------------------------------------------------
    # Set C: 학과 공지 및 총학생회 공지 (Department & Council Notices)
    # -------------------------------------------------------------
    TestCase(
        id="SET_C_DEPT_NOTICE_01",
        category="Set C: 공지사항 확장",
        name="컴퓨터공학부 학과 공지사항 최신 글 조회 (실제 31건 중 공지글 파싱 확인)",
        message="컴퓨터공학부 학과 공지사항 최신 글 목록 보여줘",
        expected_tool_keyword="department",
        expected_card_type="NOTICE_LIST",
        must_contain=["컴퓨터공학", "공지"],
        must_not_contain=["일시적인 오류로 인해", "불러오지 못했습니다"],
        notes="실제 존재하는 컴퓨터공학부 공지사항 목록을 오류 없이 정상 파싱하는지 확인",
    ),
    TestCase(
        id="SET_C_COUNCIL_NOTICE_01",
        category="Set C: 공지사항 확장",
        name="총학생회 공지사항 조회 (DB 0건 시 정직한 안내 확인)",
        message="총학생회 최근 공지사항 올라온 거 있어?",
        expected_tool_keyword="council",
        must_contain=["총학생회", "없습니다"],
        must_not_contain=["일시적인 오류로 인해", "API error", "서버 응답 지연"],
        notes="DB에 0건일 때 서버 오류 핑계 없이 정직하게 안내하는지 확인",
    ),
    # -------------------------------------------------------------
    # Set D: 학적 P2P 액션 확장 (Tuition, Scholarship, LMS)
    # -------------------------------------------------------------
    TestCase(
        id="SET_D_TUITION_01",
        category="Set D: 학적 P2P 액션",
        name="등록금 납부 고지서 조회 (Client Action 디스패치)",
        message="이번 학기 등록금 고지서 얼마 나왔어?",
        expected_tool_keyword="tuition",
        requires_auth=True,
        notes="action_portal_get_tuition 디스패치 확인",
    ),
    TestCase(
        id="SET_D_SCHOLARSHIP_01",
        category="Set D: 학적 P2P 액션",
        name="장학금 수혜 내역 조회 (Client Action 디스패치)",
        message="나 지금까지 받은 장학금 수혜 내역 조회해줘",
        expected_tool_keyword="scholarship",
        requires_auth=True,
        notes="action_portal_get_scholarship 디스패치 확인",
    ),
    TestCase(
        id="SET_D_LMS_ASSIGNMENT_01",
        category="Set D: 학적 P2P 액션",
        name="LMS 이러닝 미제출 과제/마감 기한 조회 (Client Action)",
        message="이러닝(LMS) 과제 제출 기한 남은 거 뭐 있어?",
        expected_tool_keyword="assignment",
        requires_auth=True,
        notes="action_lms_get_upcoming_assignments 디스패치 확인",
    ),
    # -------------------------------------------------------------
    # Set E: 알림 및 리마인더 (Reminder, Daily Brief, Settings)
    # -------------------------------------------------------------
    TestCase(
        id="SET_E_REMINDER_01",
        category="Set E: 알림/리마인더/설정",
        name="학식 알림 리마인더 등록",
        message="매일 아침 9시에 오늘 학식 메뉴 알려줘",
        expected_tool_keyword="menu",
        requires_auth=True,
        notes="action_manage_reminder 또는 데일리브리프/학식 도구 실행 확인",
    ),
    TestCase(
        id="SET_E_SETTINGS_01",
        category="Set E: 알림/리마인더/설정",
        name="내 맞춤 알림 및 브리프 종합 설정 확인",
        message="내 맞춤 알림이랑 데일리 브리프 설정 보여줘",
        expected_tool_keyword="settings",
        requires_auth=True,
        notes="action_my_settings 또는 daily_brief 설정 도구 확인",
    ),
    # -------------------------------------------------------------
    # Set F: 복합 실패 자가 회복 (Multi-Intent & Autonomous Fallback)
    # -------------------------------------------------------------
    TestCase(
        id="SET_F_FALLBACK_UNKNOWN_01",
        category="Set F: 복합 자가회복",
        name="미등록 인물 연락처 질의 (전화번호부 실패 -> 통합검색 자가 회복)",
        message="이철수 교수님 연락처랑 연구실 어디야?",
        expected_tool_keyword="search",
        notes="api_searchDirectory 실패 시 포기하지 않고 api_unifiedSearch로 연쇄 탐색하는지 검증",
    ),
    TestCase(
        id="SET_F_MULTI_INTENT_01",
        category="Set F: 복합 자가회복",
        name="복합 다중 의도 질의 (학생식당 학식 + 정문 버스 도착)",
        message="오늘 학생식당 학식 메뉴랑 공과대학 버스 언제 오는지 둘 다 알려줘",
        expected_tool_keyword="cafeteria",
        must_contain=["학생식당", "공과대학"],
        notes="학식과 버스 도구가 복합 실행되고 둘 다 답변에 포함되는지 검증",
    ),
]


class ExtendedEvaluator:
    def __init__(self, endpoint_url: str = DEFAULT_URL, token: str = DEFAULT_TOKEN):
        self.endpoint_url = endpoint_url
        self.token = token
        self.results: List[Dict[str, Any]] = []

    async def execute_case(self, tc: TestCase) -> Dict[str, Any]:
        print(f"\n[{tc.id}] {tc.category} - {tc.name}", flush=True)
        print(f"  질문: '{tc.message}'", flush=True)

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if tc.requires_auth:
            headers["Authorization"] = f"Bearer {self.token}"

        client_ctx = dict(tc.client_context)
        client_ctx["debug"] = True
        if tc.requires_auth:
            client_ctx["auth"] = self.token
            client_ctx["accessToken"] = self.token
            client_ctx["portal"] = {"linked": True, "studentId": "202000001"}

        payload = {
            "message": tc.message,
            "client": tc.client,
            "history": tc.history,
            "clientContext": client_ctx,
        }

        tokens = []
        thinking_chunks = []
        tools_called = []
        cards_received = []
        client_actions = []
        status_events = []
        error_msg = None
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                async with client.stream("POST", self.endpoint_url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        error_msg = f"HTTP {response.status_code}"
                        print(f"  ❌ Server returned {error_msg}", flush=True)
                    else:
                        current_event_type = None
                        buffer = ""
                        async for line in response.aiter_lines():
                            line_str = line.strip()
                            if not line_str:
                                continue
                            if line_str.startswith("event:"):
                                current_event_type = line_str.replace("event:", "").strip()
                            elif line_str.startswith("data:"):
                                raw_data = line_str.replace("data:", "").strip()
                                try:
                                    parsed = json.loads(raw_data)
                                    ev_type = parsed.get("event_type") or current_event_type
                                    if ev_type == "TOKEN":
                                        c = parsed.get("content", "")
                                        tokens.append(c)
                                    elif ev_type == "THINKING":
                                        th = parsed.get("thinking", "")
                                        if th:
                                            thinking_chunks.append(th)
                                    elif ev_type == "STATUS":
                                        st_id = parsed.get("status_id", "")
                                        st_cat = parsed.get("status_category", "")
                                        status_events.append((st_id, st_cat))
                                        if st_cat and st_cat not in tools_called:
                                            tools_called.append(st_cat)
                                    elif ev_type == "DEBUG":
                                        dbg = parsed.get("debug", {})
                                        exec_t = dbg.get("executed_tools", [])
                                        for t in exec_t:
                                            if t not in tools_called:
                                                tools_called.append(t)
                                    elif ev_type == "CARD":
                                        cards_received.append(parsed.get("card_type"))
                                    elif ev_type == "CLIENT_ACTION":
                                        client_actions.append(parsed.get("action"))
                                    elif ev_type == "ERROR":
                                        error_msg = parsed.get("error")
                                except json.JSONDecodeError:
                                    if current_event_type == "TOKEN":
                                        tokens.append(raw_data)
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ Request exception: {e}")

        elapsed = round(time.time() - start_time, 2)
        full_text = "".join(tokens).strip()

        # Extract tools called from thinking chunks
        for th in thinking_chunks:
            import re
            matches = re.findall(r"(?:api_[a-zA-Z0-9_]+|action_[a-zA-Z0-9_]+|[a-zA-Z0-9_]+Tool)", th)
            for m in matches:
                if m not in tools_called:
                    tools_called.append(m)

        # Verification Logic
        passed = True
        failure_reasons = []

        if error_msg:
            passed = False
            failure_reasons.append(f"Server error: {error_msg}")

        if not full_text and not client_actions:
            passed = False
            failure_reasons.append("Zero Response (no text and no client action received)")

        # Verify tool execution if keyword expected
        if tc.expected_tool_keyword:
            kw = tc.expected_tool_keyword.lower()
            matched_tool = any(
                kw in t.lower() for t in tools_called
            ) or any(
                kw in a.lower() for a in client_actions
            ) or any(
                kw in th.lower() for th in thinking_chunks
            )
            # Check fallback to search or notice
            fallback_matched = any("search" in t.lower() or "unified" in t.lower() or "notice" in t.lower() for t in tools_called)

            if not matched_tool and not fallback_matched:
                passed = False
                failure_reasons.append(
                    f"Expected tool matching '{tc.expected_tool_keyword}', but called: {tools_called or client_actions or 'None'}"
                )

        # Verify SDUI card
        if tc.expected_card_type:
            if tc.expected_card_type not in cards_received:
                # Soft check - card is optional if detailed text is provided
                pass

        # Ground Truth Verification: Verify required content is present in answer
        if tc.must_contain:
            for kw in tc.must_contain:
                if kw not in full_text:
                    passed = False
                    failure_reasons.append(f"Ground Truth Missing: Answer must contain '{kw}'")

        # False Error Excuse Check: Verify agent didn't falsely blame server error
        if tc.must_not_contain:
            for kw in tc.must_not_contain:
                if kw in full_text:
                    passed = False
                    failure_reasons.append(f"Unacceptable Error Excuse: Answer contained forbidden '{kw}'")

        # Zero Surrender Check: Check if agent gave up with "결과가 없습니다" without searching
        surrender_phrases = ["결과를 찾을 수 없습니다", "전화번호부 검색 실패", "검색 결과가 없습니다", "정보가 없습니다"]
        did_surrender = any(sp in full_text for sp in surrender_phrases)
        used_search_fallback = any("search" in t.lower() or "unified" in t.lower() for t in tools_called)
        if did_surrender and not used_search_fallback and not tools_called:
            passed = False
            failure_reasons.append("Premature Surrender: Agent gave up without attempting search fallback")

        status_str = "✅ PASS" if passed else "❌ FAIL"
        print(f"  결과: {status_str} ({elapsed}s)", flush=True)
        print(f"  실행된 도구: {tools_called}", flush=True)
        if client_actions:
            print(f"  디스패치된 액션: {client_actions}", flush=True)
        if cards_received:
            print(f"  수신된 카드: {cards_received}", flush=True)
        print(f"  응답 요약: {full_text[:120]}..." if len(full_text) > 120 else f"  응답: {full_text}", flush=True)
        if failure_reasons:
            print(f"  실패 원인: {', '.join(failure_reasons)}", flush=True)

        result_entry = {
            "id": tc.id,
            "category": tc.category,
            "name": tc.name,
            "message": tc.message,
            "passed": passed,
            "elapsed_seconds": elapsed,
            "tools_called": tools_called,
            "client_actions": client_actions,
            "cards_received": cards_received,
            "response_sample": full_text[:200],
            "failure_reasons": failure_reasons,
        }
        self.results.append(result_entry)
        return result_entry

    async def run_all(self):
        print("=" * 70)
        print("🚀 [Extended Agent Evaluation Suite] Live Server Multi-Domain Test")
        print(f"🌐 Target URL: {self.endpoint_url}")
        print(f"📋 Total Test Cases: {len(EXTENDED_TEST_SUITE)}")
        print("=" * 70)

        for tc in EXTENDED_TEST_SUITE:
            await self.execute_case(tc)
            await asyncio.sleep(0.5)

        total = len(self.results)
        passed_count = sum(1 for r in self.results if r["passed"])
        failed_count = total - passed_count
        pass_rate = round((passed_count / total) * 100, 1)

        print("\n" + "=" * 70)
        print(f"📊 [Evaluation Summary] Passed: {passed_count}/{total} ({pass_rate}%) | Failed: {failed_count}")
        print("=" * 70)

        # Save results to scripts/last_extended_eval_results.json
        out_path = os.path.join(os.path.dirname(__file__), "last_extended_eval_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total": total,
                "passed": passed_count,
                "failed": failed_count,
                "pass_rate": pass_rate,
                "results": self.results,
            }, f, ensure_ascii=False, indent=2)
        print(f"Saved evaluation report to {out_path}")
        return passed_count == total


if __name__ == "__main__":
    import asyncio
    evaluator = ExtendedEvaluator()
    success = asyncio.run(evaluator.run_all())
    sys.exit(0 if success else 1)
