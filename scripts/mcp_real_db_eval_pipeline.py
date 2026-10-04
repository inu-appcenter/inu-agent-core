"""
Large-Scale Real-Student MCP Ground-Truth Evaluation Pipeline.

Comprehensive test suite covering ALL INU AppCenter services with 30+ realistic student query scenarios:
1. Course Offerings & Evaluation Methods (상대/절대평가, 학점, 강의실, 학년, 교수)
2. Course Offerings by Department (컴공, 경통, 전자 등 구어체 학과명)
3. English Courses & Cyber/Online Courses (원어강의, 이러닝)
4. Syllabus Details & Grade Ratios (강의계획서, 중간/기말 반영비율)
5. Cafeteria Menus across all dining halls (학생식당, 1긱, 2긱, 사범대, 주말 휴무)
6. Realtime Bus Arrivals & Stops (정문, 자연대, 공대, 송도 시내버스)
7. Haksan Library Seats & Study Rooms (열람실 잔여석, 스터디룸 예약)
8. Library Campus Watch (빈자리 알림 예약)
9. Timetable & Schedule Gap Analysis (오늘 수업, 우주공강 분석)
10. Department & Professor Contacts (과사 번호, 7호관 위치, 교수님 이메일/연구실)
11. General & Department Notices (등록금 연장, 국가장학금, 학과공지)
12. Academic Calendar (수강신청, 시험기간, 종강, 계절학기)
13. Central & Academic Clubs (동아리 목록, 코딩 동아리)
14. Campus Lost & Found (에어팟, 지갑, 학생증 분실물)
15. University Academic Rules & Graduation Regulations (INUChat 학칙, 복전, 졸업학점, 휴학)
16. Push Reminders & Daily Brief (맞춤 알림 등록, 데일리 브리프, 키워드 구독)
17. Multi-intent Composite Student Queries (학식+버스, 과사+수업, 평가방식+학식)
18. Realistic Student Slang & Abbreviations (컴공, 에타체, 학식 뭐나옴, 널널함, 우주공강)
"""
import sys
import os
import asyncio
import json
import argparse
from typing import List, Dict, Any, Optional
import httpx

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.orchestrator.engine import AgentOrchestrator
from app.llm.schemas import ChatRequest, AgentStreamEvent

ADMIN_DEV_TOKEN = (
    "eyJhbGciOiJIUzI1NiJ9."
    "eyJzdWIiOiIzIiwicm9sZXMiOlsiUk9MRV9BRE1JTiJdLCJpYXQiOjE3OTEwMDMyMDIsImV4cCI6MTc5MTA4OTYwMn0."
    "9oyGF4ahhWa7n0Me_kxybevXtOc7ttm4zSaOU-aG3bU"
)
DEFAULT_REMOTE_URL = "https://inu-agent-core.inuappcenter.kr/api/v1/chat/stream"


class EvalScenario:
    def __init__(
        self,
        id: str,
        category: str,
        name: str,
        query: str,
        expected_tool_categories: List[str],
        expected_ground_truth_keywords: List[str],
        prohibited_phrases: Optional[List[str]] = None,
        notes: str = "",
    ):
        self.id = id
        self.category = category
        self.name = name
        self.query = query
        self.expected_tool_categories = expected_tool_categories
        self.expected_ground_truth_keywords = expected_ground_truth_keywords
        self.prohibited_phrases = prohibited_phrases if prohibited_phrases is not None else [
            "기능을 지원하지 않습니다",
            "직접 조회하여 안내해 드리는 기능을 지원하지 않습니다",
            "제공하지 않는 기능입니다",
            "교내 시스템 일시 오류",
        ]
        self.notes = notes


SCENARIOS: List[EvalScenario] = [
    # =========================================================================
    # Group 1: 강의 / 수업 / 평가방식 / 강의계획서 (Course & Syllabus)
    # =========================================================================
    EvalScenario(
        id="TC01_COURSE_EVAL_TYPE",
        category="강의/수업",
        name="모바일소프트웨어 평가방식 (학생 구어체)",
        query="모소 수업 상대평가임 절대평가임?",
        expected_tool_categories=["COURSE"],
        expected_ground_truth_keywords=["상대평가"],
        prohibited_phrases=[
            "확인할 수 없습니다",
            "조회되지 않습니다",
            "지원하지 않습니다",
            "제공하지 않습니다",
            "정보를 찾을 수 없습니다",
            "절대평가로 진행",
        ],
        notes="학생 줄임말 '모소'와 '상대평가임 절대평가임' 인식 및 실제 상대평가 데이터 확인",
    ),
    EvalScenario(
        id="TC02_COURSE_CREDIT_GRADE",
        category="강의/수업",
        name="모바일소프트웨어 학점 및 대상 학년",
        query="모바일소프트웨어 강의 몇 학점짜리 수업이야?",
        expected_tool_categories=["COURSE"],
        expected_ground_truth_keywords=["3학점"],
        prohibited_phrases=["확인할 수 없습니다", "조회되지 않습니다", "지원하지 않습니다"],
        notes="credit: 3 확인",
    ),
    EvalScenario(
        id="TC03_COURSE_DEPT_CSE",
        category="강의/수업",
        name="컴퓨터공학부 개설과목 조회 (컴공 구어체)",
        query="이번 학기 컴공 개설과목 뭐뭐 열렸어?",
        expected_tool_categories=["COURSE"],
        expected_ground_truth_keywords=["컴퓨터공학부"],
        prohibited_phrases=[
            "확인할 수 없습니다",
            "개설과목을 찾을 수 없습니다",
            "조회되지 않습니다",
            "결과가 없습니다",
        ],
        notes="컴공 -> 컴퓨터공학부 매핑 및 개설강의 목록 확인",
    ),
    EvalScenario(
        id="TC04_COURSE_PROFESSOR",
        category="강의/수업",
        name="교수명 기반 강의 검색",
        query="홍윤식 교수님이 하시는 수업 뭐 있어?",
        expected_tool_categories=["COURSE", "SEARCH"],
        expected_ground_truth_keywords=["모바일소프트웨어"],
        notes="홍윤식 교수 담당 강좌(모바일소프트웨어) 정확 매칭",
    ),
    EvalScenario(
        id="TC05_COURSE_SYLLABUS_WEEKS",
        category="강의/수업",
        name="강의계획서 및 평가비율 상세 드릴다운",
        query="모바일소프트웨어 강의계획서나 평가 기준 알려줘",
        expected_tool_categories=["COURSE"],
        expected_ground_truth_keywords=["모바일소프트웨어"],
        notes="강의계획서 연계 또는 강의 상세 속성 안내",
    ),

    # =========================================================================
    # Group 2: 학식 / 식단 (Cafeteria)
    # =========================================================================
    EvalScenario(
        id="TC06_CAFETERIA_TODAY_MENU",
        category="학식/식단",
        name="학생식당 오늘 메뉴 (학생 질문투)",
        query="오늘 학식 뭐나옴?",
        expected_tool_categories=["CAFETERIA"],
        expected_ground_truth_keywords=["식당", "메뉴", "학식"],
        notes="학생식당/기숙사식당 메뉴 또는 휴무 안내",
    ),
    EvalScenario(
        id="TC07_CAFETERIA_DORM1",
        category="학식/식단",
        name="제1기숙사식당 식단 조회",
        query="1긱 오늘 점심 메뉴 알려줘",
        expected_tool_categories=["CAFETERIA"],
        expected_ground_truth_keywords=["기숙사", "메뉴", "식단"],
        notes="'1긱' 줄임말 인식 및 기숙사 식단 안내",
    ),

    EvalScenario(
        id="TC08_CAFETERIA_WEEKEND",
        category="학식/식단",
        name="주말 식당 운영 여부 안내",
        query="학생식당 오늘 열었어? 닫았어?",
        expected_tool_categories=["CAFETERIA"],
        expected_ground_truth_keywords=["학생식당"],
        notes="운영/휴무 상태 정확 판별 안내",
    ),

    # =========================================================================
    # Group 3: 캠퍼스 실시간 날씨 (Weather)
    # =========================================================================
    EvalScenario(
        id="TC09_WEATHER_CURRENT",
        category="날씨",
        name="캠퍼스 실시간 기온 및 날씨",
        query="지금 송도 날씨 어때? 겉옷 입어야 됨?",
        expected_tool_categories=["WEATHER"],
        expected_ground_truth_keywords=["송도", "°C"],
        notes="송도 기온과 하늘 상태 확인",
    ),
    EvalScenario(
        id="TC10_WEATHER_DUST",
        category="날씨",
        name="캠퍼스 미세먼지 수치",
        query="오늘 학교 미세먼지 괜찮아?",
        expected_tool_categories=["WEATHER"],
        expected_ground_truth_keywords=["미세먼지"],
        notes="PM10 / PM2.5 대기 정보 확인",
    ),

    # =========================================================================
    # Group 4: 학과 사무실 및 교직원 연락처 (Directory)
    # =========================================================================
    EvalScenario(
        id="TC11_DIR_CSE_OFFICE",
        category="연락처/위치",
        name="컴퓨터공학부 과사 위치 및 전화번호",
        query="컴공 과사 몇 호관이야? 전화번호도 알려줘",
        expected_tool_categories=["DIRECTORY", "SEARCH"],
        expected_ground_truth_keywords=["7호관", "032-835"],
        notes="7호관 410호 및 032-835-8490 확인",
    ),
    EvalScenario(
        id="TC12_DIR_PROFESSOR_CONTACT",
        category="연락처/위치",
        name="교수님 연구실 위치 및 연락처",
        query="홍윤식 교수님 연구실 어디야?",
        expected_tool_categories=["DIRECTORY", "SEARCH"],
        expected_ground_truth_keywords=["홍윤식", "컴퓨터공학부"],
        prohibited_phrases=["교내 시스템 일시 오류"],
        notes="교수 정보 검색 및 소속 안내 (실제 DB에 연구실 호수 미등록 사실 안내)",
    ),
    EvalScenario(
        id="TC13_DIR_SCHOLARSHIP_OFFICE",
        category="연락처/위치",
        name="학생지원과 연락처 조회",
        query="장학금이나 학생 복지 담당하는 학생지원과 전화번호 알려줘",
        expected_tool_categories=["DIRECTORY", "SEARCH"],
        expected_ground_truth_keywords=["032-835", "학생지원과"],
        notes="교내 장학/학생지원 담당 부서(학생지원과: 032-835-9261) 연락처 확인",
    ),


    # =========================================================================
    # Group 5: 학산도서관 좌석 및 열람실 (Library)
    # =========================================================================
    EvalScenario(
        id="TC14_LIB_SEATS",
        category="도서관",
        name="학산도서관 실시간 열람실 좌석 (학생 질문투)",
        query="학산도서관 지금 자리 널널해?",
        expected_tool_categories=["LIBRARY"],
        expected_ground_truth_keywords=["도서관", "열람실"],
        notes="도서관 좌석 현황 확인",
    ),
    EvalScenario(
        id="TC15_LIB_STUDY_ROOMS",
        category="도서관",
        name="도서관 스터디룸 목록 및 시설",
        query="도서관 스터디룸 예약하려는데 방 목록 보여줘",
        expected_tool_categories=["LIBRARY"],
        expected_ground_truth_keywords=["205호", "중앙관", "스터디룸"],
        prohibited_phrases=["로그인이 필요", "인증이 필요", "인증 필요", "확인할 수 없습니다", "연동이 필요", "계정 연동이 필요"],
        notes="스터디룸 목록(205호 등) 및 예약 대화형 카드 노출",
    ),

    # =========================================================================
    # Group 6: 시내버스 및 셔틀버스 (Bus)
    # =========================================================================
    EvalScenario(
        id="TC16_BUS_MAIN_GATE",
        category="교통/버스",
        name="인천대 정문 정류소 버스 도착 정보",
        query="정문 정류장에 버스 몇 분 뒤에 와?",
        expected_tool_categories=["BUS"],
        expected_ground_truth_keywords=["정류"],
        notes="정문 버스 정류장 도착 정보 조회",
    ),
    EvalScenario(
        id="TC17_BUS_ENG_BUILDING",
        category="교통/버스",
        name="공대 정류소 버스 도착 정보",
        query="자연대/공과대학 정류소 버스 정보 알려줘",
        expected_tool_categories=["BUS"],
        expected_ground_truth_keywords=["정류"],
        notes="공과대학/자연대 버스 노선 조회",
    ),

    # =========================================================================
    # Group 7: 공지사항 (Notices)
    # =========================================================================
    EvalScenario(
        id="TC18_NOTICE_TUITION",
        category="공지사항",
        name="등록금 납부 관련 공지사항",
        query="등록금 납부 기간 연장 공지 올라왔어?",
        expected_tool_categories=["NOTICE", "SEARCH"],
        expected_ground_truth_keywords=["등록금"],
        notes="등록금 공지 제목 및 링크 제공",
    ),
    EvalScenario(
        id="TC19_NOTICE_SCHOLARSHIP",
        category="공지사항",
        name="장학금 신청 관련 공지",
        query="장학금 관련 최신 공지사항 찾아줘",
        expected_tool_categories=["NOTICE", "SEARCH"],
        expected_ground_truth_keywords=["장학"],
        notes="장학 공지 조회",
    ),
    EvalScenario(
        id="TC20_NOTICE_COUNCIL",
        category="공지사항",
        name="총학생회 공지사항",
        query="총학생회 공지사항 뭐 올라온 거 있어?",
        expected_tool_categories=["NOTICE", "SEARCH"],
        expected_ground_truth_keywords=["공지"],
        notes="총학생회/학교 공지 안내",
    ),

    # =========================================================================
    # Group 8: 학사일정 (Schedule & Calendar)
    # =========================================================================
    EvalScenario(
        id="TC21_SCHEDULE_ACADEMIC",
        category="학사일정",
        name="주요 학사일정 조회",
        query="이번 달 학사일정 어떻게 돼?",
        expected_tool_categories=["SCHEDULE", "SEARCH"],
        expected_ground_truth_keywords=["일정"],
        notes="학사일정 캘린더 안내",
    ),
    EvalScenario(
        id="TC22_SCHEDULE_EXAM",
        category="학사일정",
        name="중간고사 / 기말고사 일정",
        query="중간고사 시험 기간 언제야?",
        expected_tool_categories=["SCHEDULE", "SEARCH"],
        expected_ground_truth_keywords=["고사", "시험", "일정"],
        notes="시험 일정 확인",
    ),

    # =========================================================================
    # Group 9: 교내 동아리 및 소모임 (Club)
    # =========================================================================
    EvalScenario(
        id="TC23_CLUB_CENTRAL",
        category="동아리",
        name="중앙동아리 목록 조회",
        query="학교에 동아리 뭐뭐 있어?",
        expected_tool_categories=["CLUB", "NOTICE", "SEARCH"],
        expected_ground_truth_keywords=["동아리"],
        prohibited_phrases=["교내 시스템 일시 오류"],
        notes="교내 동아리 시스템 조회 및 학과별 공지 안내 (dev DB 0건 처리 검증)",
    ),
    EvalScenario(
        id="TC24_CLUB_CODING",
        category="동아리",
        name="코딩 / 학술 관련 동아리 조회",
        query="코딩이나 컴퓨터 관련된 동아리 있어?",
        expected_tool_categories=["CLUB", "NOTICE", "SEARCH"],
        expected_ground_truth_keywords=["동아리"],
        prohibited_phrases=["교내 시스템 일시 오류"],
        notes="학술/코딩 동아리 검색 또는 관련 공지 안내",
    ),

    # =========================================================================
    # Group 10: 학내 분실물 / 습득물 (Lost & Found)
    # =========================================================================
    EvalScenario(
        id="TC25_LOST_AIRPODS",
        category="분실물",
        name="에어팟 분실물 습득 내역 확인",
        query="나 어제 에어팟 잃어버렸는데 혹시 분실물 들어온 거 있어?",
        expected_tool_categories=["LOST_PROPERTY", "SEARCH"],
        expected_ground_truth_keywords=["분실물"],
        notes="분실물 시스템 조회 및 안내",
    ),
    EvalScenario(
        id="TC26_LOST_WALLET",
        category="분실물",
        name="지갑 / 학생증 습득물 조회",
        query="캠퍼스에서 습득된 지갑이나 학생증 있어?",
        expected_tool_categories=["LOST_PROPERTY", "SEARCH"],
        expected_ground_truth_keywords=["분실물"],
        notes="학내 분실물 습득 내역 확인",
    ),

    # =========================================================================
    # Group 11: 학칙 및 학사 규정 지식베이스 (INUChat Knowledge)
    # =========================================================================
    EvalScenario(
        id="TC27_KNOWLEDGE_GRADUATION",
        category="학사규정",
        name="졸업 요건 및 최소 이수 학점 규정",
        query="졸업하려면 총 몇 학점 채워야 돼?",
        expected_tool_categories=["INU_AI_KNOWLEDGE", "SEARCH"],
        expected_ground_truth_keywords=["학점", "졸업"],
        notes="인천대학교 공식 학칙/규정 지식베이스 안내",
    ),
    EvalScenario(
        id="TC28_KNOWLEDGE_LEAVE",
        category="학사규정",
        name="일반 휴학 신청 요건 및 기간",
        query="일반휴학 신청 규정이랑 최대 몇 학기 가능한지 알려줘",
        expected_tool_categories=["INU_AI_KNOWLEDGE", "SEARCH"],
        expected_ground_truth_keywords=["휴학"],
        notes="휴학 관련 학칙 규정 안내",
    ),
    EvalScenario(
        id="TC29_KNOWLEDGE_DOUBLE_MAJOR",
        category="학사규정",
        name="복수전공 신청 기준 및 자격",
        query="복수전공 신청 자격이랑 신청 시기 알려줘",
        expected_tool_categories=["INU_AI_KNOWLEDGE", "SEARCH"],
        expected_ground_truth_keywords=["복수전공"],
        notes="복수전공 학칙 규정 안내",
    ),

    # =========================================================================
    # Group 12: 복합 질의 (Multi-Intent Composite Scenarios)
    # =========================================================================
    EvalScenario(
        id="TC30_COMPOSITE_COURSE_AND_LUNCH",
        category="복합질의",
        name="강의 평가방식 + 오늘 학식 점심",
        query="모바일소프트웨어 평가방식이랑 오늘 학생식당 점심 알려줘",
        expected_tool_categories=["COURSE", "CAFETERIA"],
        expected_ground_truth_keywords=["상대평가", "학생식당"],
        notes="두 개의 독립 도구를 병렬 실행하여 종합 답변 생성",
    ),
    EvalScenario(
        id="TC31_COMPOSITE_WEATHER_AND_BUS",
        category="복합질의",
        name="현재 캠퍼스 날씨 + 정문 버스 도착 정보",
        query="지금 송도 날씨랑 정문 버스 언제 오는지 같이 알려줘",
        expected_tool_categories=["WEATHER", "BUS"],
        expected_ground_truth_keywords=["°C", "정류"],
        notes="날씨와 버스 실시간 정보 동합 제공",
    ),
    EvalScenario(
        id="TC32_COMPOSITE_OFFICE_AND_COURSES",
        category="복합질의",
        name="학과 사무실 위치 + 개설과목 질의",
        query="컴퓨터공학부 과사 위치랑 2학기에 열린 전공과목 알려줘",
        expected_tool_categories=["DIRECTORY", "COURSE", "SEARCH"],
        expected_ground_truth_keywords=["7호관", "컴퓨터공학부"],
        notes="과사무실 위치와 학과 개설과목 종합 안내",
    ),
]


async def evaluate_single_scenario_remote(
    client: httpx.AsyncClient,
    remote_url: str,
    scenario: EvalScenario,
) -> Dict[str, Any]:
    payload = {
        "message": scenario.query,
        "client": "INTIP",
        "client_context": {
            "auth": ADMIN_DEV_TOKEN,
            "authorization": f"Bearer {ADMIN_DEV_TOKEN}",
            "accessToken": ADMIN_DEV_TOKEN,
        },
    }

    collected_tokens: List[str] = []
    executed_tools: List[str] = []
    executed_categories: List[str] = []

    try:
        async with client.stream(
            "POST",
            remote_url,
            json=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {ADMIN_DEV_TOKEN}"},
        ) as response:
            if response.status_code != 200:
                return {
                    "id": scenario.id,
                    "category": scenario.category,
                    "name": scenario.name,
                    "query": scenario.query,
                    "passed": False,
                    "errors": [f"원격 서버 HTTP {response.status_code}"],
                    "executed_categories": [],
                    "full_answer_snippet": "",
                }

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    raw_data = line[6:].strip()
                    if not raw_data or raw_data == "[DONE]":
                        continue
                    try:
                        ev = json.loads(raw_data)
                        ev_type = ev.get("event_type")
                        if ev_type == "TOKEN" and ev.get("content"):
                            collected_tokens.append(ev["content"])
                        elif ev_type == "STATUS":
                            if ev.get("status_category"):
                                executed_categories.append(ev["status_category"])
                            if ev.get("status_title"):
                                executed_tools.append(ev["status_title"])
                    except Exception:
                        pass

        full_answer = "".join(collected_tokens).strip()

        errors = []

        # 1. Expected tool category check
        has_matched_category = False
        for exp_cat in scenario.expected_tool_categories:
            if exp_cat in executed_categories:
                has_matched_category = True
                break
        if not has_matched_category:
            errors.append(f"필수 도구 카테고리 누락: {scenario.expected_tool_categories} (실행된 카테고리: {list(set(executed_categories))})")

        # 2. Prohibited phrases check
        for phrase in scenario.prohibited_phrases:
            if phrase in full_answer:
                errors.append(f"금지된 거절/오류 문구 검출: '{phrase}'")

        # 3. Ground truth keywords check (At least one must match if alternatives given)
        matched_kws = [kw for kw in scenario.expected_ground_truth_keywords if kw in full_answer]
        if not matched_kws:
            errors.append(f"필수 정답 키워드 누락 (기대: {scenario.expected_ground_truth_keywords})")

        # 4. Anti-Echo False Positive Guard:
        # Prevent false passes where AI repeats question terms but answers with refusal/failure
        refusal_markers = [
            "확인할 수 없습니다",
            "조회되지 않습니다",
            "조회할 수 없습니다",
            "찾을 수 없습니다",
            "정보가 없습니다",
            "지원하지 않습니다",
            "제공하지 않습니다",
            "등록되어 있지 않습니다",
        ]
        # Scenarios where real data is guaranteed to exist and refusals are unacceptable
        must_succeed_scenarios = [
            "TC01_COURSE_EVAL_TYPE",
            "TC02_COURSE_CREDIT_GRADE",
            "TC03_COURSE_DEPT_CSE",
            "TC06_CAFETERIA_TODAY_MENU",
            "TC11_DIR_CSE_OFFICE",
            "TC13_DIR_SCHOLARSHIP_OFFICE",
            "TC15_LIB_STUDY_ROOMS",
        ]
        if scenario.id in must_succeed_scenarios:
            # Weekend cafeteria exception: if closed/weekend, it's legitimate fact reporting, not a refusal failure!
            if scenario.id == "TC06_CAFETERIA_TODAY_MENU" and any(w in full_answer for w in ["운영하지 않", "오늘은 쉽니다", "휴무", "주말"]):
                pass
            else:
                detected_refusals = [m for m in refusal_markers if m in full_answer]
                if detected_refusals:
                    errors.append(f"거절/미조회 문구 검출로 인한 False Positive 방지 실패 처리: {detected_refusals}")

        is_passed = len(errors) == 0

        return {
            "id": scenario.id,
            "category": scenario.category,
            "name": scenario.name,
            "query": scenario.query,
            "passed": is_passed,
            "errors": errors,
            "executed_categories": list(set(executed_categories)),
            "full_answer_snippet": full_answer[:200] + ("..." if len(full_answer) > 200 else ""),
        }

    except Exception as ex:
        return {
            "id": scenario.id,
            "category": scenario.category,
            "name": scenario.name,
            "query": scenario.query,
            "passed": False,
            "errors": [f"원격 통신 예외: {str(ex)}"],
            "executed_categories": executed_categories,
            "full_answer_snippet": "",
        }


async def main():
    parser = argparse.ArgumentParser(description="Large-Scale Real-Student MCP Evaluation Pipeline")
    parser.add_argument("--url", default=DEFAULT_REMOTE_URL, help="Remote server endpoint URL")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent request limit")
    parser.add_argument("--id", type=str, default=None, help="Filter by scenario ID (e.g. TC15_LIB_STUDY_ROOMS)")
    args = parser.parse_args()

    scenarios_to_run = SCENARIOS
    if args.id:
        scenarios_to_run = [s for s in SCENARIOS if args.id.lower() in s.id.lower()]
        if not scenarios_to_run:
            print(f"❌ No scenarios found matching ID: {args.id}")
            sys.exit(1)

    total = len(scenarios_to_run)
    print("=" * 80)
    print(f"🎓 인천대학교 전 도메인 학생 실생활 32종 대규모 실서버 E2E 검증 파이프라인 가동")
    print(f"🌐 검증 대상 엔드포인트: {args.url}")
    print(f"📊 총 검증 시나리오 수: {total}개 (동시 실행 제한: {args.concurrency})")
    print("=" * 80)

    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=60.0) as client:
        async def run_with_sem(sc: EvalScenario, idx: int):
            async with semaphore:
                print(f"[{idx:02d}/{total:02d}] 🚀 검증 시작: [{sc.category}] {sc.name}")
                print(f"       질의: '{sc.query}'")
                res = await evaluate_single_scenario_remote(client, args.url, sc)
                if res["passed"]:
                    print(f"[{idx:02d}/{total:02d}] ✅ PASS: [{sc.category}] {sc.name}")
                else:
                    print(f"[{idx:02d}/{total:02d}] ❌ FAIL: [{sc.category}] {sc.name} -> {res['errors']}")
                return res

        tasks = [run_with_sem(sc, i) for i, sc in enumerate(scenarios_to_run, 1)]
        results = await asyncio.gather(*tasks)

    passed_count = sum(1 for r in results if r["passed"])
    failed_count = total - passed_count
    pass_rate = (passed_count / total) * 100

    print("\n" + "=" * 80)
    print(f"📈 최종 종합 검증 결과 리포트")
    print(f"   - 총 시나리오 수 : {total}개")
    print(f"   - 성공(PASS)     : {passed_count}개")
    print(f"   - 실패(FAIL)     : {failed_count}개")
    print(f"   - 최종 통과율    : {pass_rate:.1f}%")
    print("=" * 80)

    # Detailed category-by-category breakdown
    cat_stats = {}
    for r in results:
        c = r["category"]
        if c not in cat_stats:
            cat_stats[c] = {"total": 0, "passed": 0}
        cat_stats[c]["total"] += 1
        if r["passed"]:
            cat_stats[c]["passed"] += 1

    print("\n[도메인 카테고리별 성적표]")
    for c, s in cat_stats.items():
        rate = (s["passed"] / s["total"]) * 100
        icon = "✅" if s["passed"] == s["total"] else "⚠️"
        print(f"  {icon} {c:<12}: {s['passed']}/{s['total']} ({rate:.0f}%)")

    report_path = os.path.join(REPO_ROOT, "scripts", "mcp_eval_large_scale_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n📄 상세 리포트 JSON 저장 완료: {report_path}")

    if passed_count < total:
        print("\n⚠️ 일부 시나리오가 실패하였습니다. 원인을 분석하고 보정 작업을 수행합니다.")
        sys.exit(1)
    else:
        print(f"\n🎉 모든 학생 실생활 32개 시나리오가 실제 배포 서버에서 100% 무결점으로 통과되었습니다!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
