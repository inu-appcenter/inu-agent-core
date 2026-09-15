"""
Advanced Tool Pruning Engine for Gemma 27B
Selects the 3-5 most relevant tools from dozens of candidates to maximize attention & accuracy.
"""
from typing import List, Dict, Any
from app.tools.base import BaseTool
from app.core.logging import logger


class ToolPruner:
    CATEGORY_KEYWORDS: Dict[str, List[str]] = {
        "CAFETERIA": ["학식", "식당", "밥", "메뉴", "점심", "저녁", "아침", "조식", "중식", "석식", "카페테리아", "식단"],
        "BUS": ["버스", "셔틀", "셔틀버스", "인입런", "정류장", "노선", "도착", "막차", "첫차", "출발"],
        "TIMETABLE": ["시간표", "강의", "수업", "공강", "강좌", "교수", "과목", "시간표조회", "내시간표"],
        "NOTICE": ["공지", "공지사항", "학사공지", "장학공지", "모집", "대회", "행사"],
        "SCHEDULE": ["학사일정", "일정", "시험", "중간고사", "기말고사", "개강", "종강", "등록금납부"],
        "RESERVATION": ["예약", "시설", "세미나실", "강의실대여", "운동장", "풋살장"],
        "WEATHER": ["날씨", "미세먼지", "비", "우산", "기온", "온도"],
        "LMS": ["과제", "lms", "사이버캠퍼스", "사캠", "출석", "강의영상", "퀴즈", "제출"],
        "LIBRARY": ["도서관", "열람실", "좌석", "자리", "연장", "반납", "책", "도서", "대출", "스터디룸"],
        "PORTAL": ["성적", "학점", "학적", "장학금", "휴학", "복학", "졸업", "이수학점", "증명서", "포털"],
        "DORM": ["외박", "기숙사", "생활관", "유니돔", "상벌점", "세탁기", "건조기", "점호", "호실"],
    }

    @classmethod
    def prune(
        cls,
        query: str,
        tools: List[BaseTool],
        client: str = "INTIP",
        max_tools: int = 4,
    ) -> List[BaseTool]:
        """
        Filter down tools to the top most relevant for Gemma 27B.
        """
        q = query.lower()
        matched_categories = set()

        for category, keywords in cls.CATEGORY_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                matched_categories.add(category)

        # If no domain keywords match, treat as general conversation (no tools needed)
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
