"""
Advanced Tool Pruning Engine for Gemma 27B
Selects the 3-5 most relevant tools from dozens of candidates to maximize attention & accuracy.
"""
from typing import List, Dict, Any
from app.tools.base import BaseTool
from app.core.logging import logger


from typing import List, Dict, Any, Optional
from app.tools.base import BaseTool
from app.core.logging import logger


class ToolPruner:
    CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "CAFETERIA": ["학식", "식당", "밥", "메뉴", "점심", "저녁", "아침", "조식", "중식", "석식", "카페테리아", "식단"],
        "BUS": ["버스", "셔틀", "셔틀버스", "인입런", "정류장", "노선", "도착", "막차", "첫차", "출발", "몇 분 후"],
        "TIMETABLE": ["시간표", "강의", "수업", "공강", "강좌", "교수", "과목", "시간표조회", "내시간표", "다음 수업", "강의실"],
        "NOTICE": ["공지", "공지사항", "학사공지", "장학공지", "모집", "대회", "행사"],
        "SCHEDULE": ["학사일정", "일정", "시험", "중간고사", "기말고사", "개강", "종강", "등록금납부"],
        "RESERVATION": ["예약", "시설", "세미나실", "강의실대여", "운동장", "풋살장"],
        "WEATHER": ["날씨", "미세먼지", "비", "우산", "기온", "온도", "초미세먼지"],
        "LMS": ["과제", "lms", "이러닝", "사이버캠퍼스", "사캠", "출석", "강의영상", "퀴즈", "제출"],
        "LIBRARY": ["도서관", "열람실", "좌석", "자리", "연장", "반납", "책", "도서", "대출", "스터디룸", "배정"],
        "PORTAL": ["성적", "학점", "학적", "장학금", "휴학", "복학", "졸업", "이수학점", "증명서", "포털", "지도교수", "담임교수"],
        "DIRECTORY": ["교수", "교수님", "과사", "학과사무실", "사무실", "전화번호", "연락처", "이메일", "연구실", "조교", "부서"],
        "INU_AI_KNOWLEDGE": [
            "학칙", "규정", "졸업", "졸업요건", "졸업 요건", "조기졸업", "조기 졸업", "복수전공", "부전공",
            "전과", "학사경고", "공학인증", "휴학", "복학", "수료", "이수", "장학금규정", "졸업학점",
            "졸업논문", "영어졸업인증", "졸업인증", "편입", "재입학", "학사제도"
        ],
        "DORM": ["외박", "기숙사", "생활관", "유니돔", "상벌점", "세탁기", "건조기", "점호", "호실"],
    }

    @classmethod
    def prune(
        cls,
        query: str,
        tools: List[BaseTool],
        history: Optional[List[Any]] = None,
        client: str = "INTIP",
        max_tools: int = 4,
    ) -> List[BaseTool]:
        """
        Filter down tools to the top most relevant for Gemma 27B with multi-turn context inheritance.
        """
        q = query.lower()
        matched_categories = set()

        # Augment query with recent user turns for context inheritance (e.g. '20학번은?', '연락처는?')
        augmented_text = q
        if history and len(history) > 0:
            user_prev_messages = [
                (h.content if hasattr(h, "content") else h.get("content", "")).lower()
                for h in history[-4:]
                if (h.role if hasattr(h, "role") else h.get("role")) == "user"
            ]
            if user_prev_messages:
                augmented_text += " " + " ".join(user_prev_messages)

        # 1. First-person graduation/academic eligibility check -> require both PORTAL and INU_AI_KNOWLEDGE
        is_first_person_grad = any(k in q for k in ["나 졸업", "내 졸업", "내 학점", "졸업 가능", "졸업 요건 돼", "졸업할 수 있어"])
        if is_first_person_grad:
            matched_categories.add("PORTAL")
            matched_categories.add("INU_AI_KNOWLEDGE")

        # 2. Advisor professor inquiry -> require both PORTAL and DIRECTORY
        is_advisor_inquiry = any(k in q for k in ["지도교수", "담임교수", "지도 교수"])
        if is_advisor_inquiry:
            matched_categories.add("PORTAL")
            matched_categories.add("DIRECTORY")

        # 3. Match categories from augmented text
        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if any(kw in augmented_text for kw in keywords):
                matched_categories.add(category)

        # If no domain keywords match, treat as general conversation
        if not matched_categories:
            return []

        # Score tools
        scored_tools = []
        for tool in tools:
            score = 0
            if tool.category in matched_categories:
                score += 10

            # Match directly on tool name or description
            desc = tool.description.lower()
            name = tool.name.lower()
            for cat in matched_categories:
                for kw in cls.CATEGORY_KEYWORDS.get(cat, []):
                    if kw in desc or kw in name:
                        score += 5

            if score > 0:
                scored_tools.append((score, tool))

        # Sort by score descending
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        selected = [t for _, t in scored_tools[:max_tools]]

        # If still empty, fall back to first N tools
        if not selected and tools:
            selected = tools[:max_tools]

        logger.debug(
            f"ToolPruner selected {len(selected)} tools ({[t.name for t in selected]}) for query: '{query[:30]}...'"
        )
        return selected
