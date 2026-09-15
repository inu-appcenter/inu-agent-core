"""
Tool Parameter Resolver
Intelligently resolves and defaults parameters for OpenAPI tools based on user queries and temporal context.
"""
from typing import Dict, Any
from datetime import datetime


class ToolParameterResolver:
    """
    Extracts or supplies robust defaults for OpenAPI tools (Bus, Cafeteria, Notice, Schedule, etc.)
    """
    @classmethod
    def resolve_arguments(cls, tool_name: str, category: str, query: str) -> Dict[str, Any]:
        q = query.lower()
        now = datetime.now()
        args: Dict[str, Any] = {}

        if category == "BUS":
            # 1. Bus Stop ID resolution
            if any(k in q for k in ["인천대입구역", "인입", "인입런", "1번출구", "2번출구", "지하철역"]):
                args["bstopId"] = "164000396"
            elif any(k in q for k in ["정문", "인천대정문", "인천대 정문", "본관", "대학본부"]):
                args["bstopId"] = "164000385"
            elif any(k in q for k in ["공대", "공과대학", "공학관", "공대앞", "공학부"]):
                args["bstopId"] = "164000377"
            elif any(k in q for k in ["자연대", "자연과학대학", "자연과학부"]):
                args["bstopId"] = "164000378"
            elif any(k in q for k in ["기숙사", "생활관", "유니돔"]):
                args["bstopId"] = "164000751"
            elif any(k in q for k in ["지식정보단지역", "지정단", "지식정보"]):
                args["bstopId"] = "164000404"
            else:
                # Default by time of day: morning -> 인천대입구역, afternoon/night -> 인천대 정문
                if now.hour < 14:
                    args["bstopId"] = "164000396"  # 인천대입구역 (등교)
                else:
                    args["bstopId"] = "164000385"  # 인천대 정문 (하교)

        elif category == "CAFETERIA":
            # Cafeteria restaurant name
            if any(k in q for k in ["기숙사", "1기숙사", "제1기숙사"]):
                args["cafeteria"] = "제1기숙사식당"
            elif any(k in q for k in ["2기숙사", "제2기숙사", "이룸관"]):
                args["cafeteria"] = "2기숙사 식당"
            elif any(k in q for k in ["27호관", "사범대", "교육관"]):
                args["cafeteria"] = "27호관식당"
            else:
                args["cafeteria"] = "학생식당"

            # Day of week: 1=월 ~ 7=일
            if "내일" in q:
                args["day"] = (now.weekday() + 1) % 7 + 1
            elif "월요일" in q or "월" in q:
                args["day"] = 1
            elif "화요일" in q or "화" in q:
                args["day"] = 2
            elif "수요일" in q or "수" in q:
                args["day"] = 3
            elif "목요일" in q or "목" in q:
                args["day"] = 4
            elif "금요일" in q or "금" in q:
                args["day"] = 5
            elif "토요일" in q or "토" in q:
                args["day"] = 6
            elif "일요일" in q or "일" in q:
                args["day"] = 7
            else:
                args["day"] = now.weekday() + 1  # Today

        elif category == "NOTICE":
            if "장학" in q:
                args["category"] = "장학"
            elif "행사" in q or "축제" in q:
                args["category"] = "행사"
            elif "일반" in q or "모집" in q:
                args["category"] = "일반"
            else:
                args["category"] = "학사"

        elif category == "SCHEDULE":
            args["year"] = now.year
            args["month"] = now.month

        return args
