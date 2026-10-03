"""
Schema Coercer Module
Generic, schema-driven parameter validation, type coercion, and semantic synonym resolution
for MCP and native tools, eliminating brittle domain-specific hardcodings in the orchestrator.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from app.core.logging import logger

# Semantic synonym mappings for common campus enum values
CAMPUS_SYNONYMS: Dict[str, list[str]] = {
    # Meal types
    "LUNCH": ["점심", "중식", "점심식사", "런치", "점심메뉴", "중식메뉴"],
    "BREAKFAST": ["아침", "조식", "아침식사", "모닝", "아침메뉴", "조식메뉴"],
    "DINNER": ["저녁", "석식", "저녁식사", "디너", "저녁메뉴", "석식메뉴"],
    "AUTO": ["전체", "자동", "all", "ALL", "모두", "상관없음"],
    # Cafeterias
    "학생식당": ["학식", "1학식", "제1학생식당", "학생식당메뉴"],
    "제1기숙사식당": ["1긱", "1기숙사", "제1기숙사", "1기숙사식당", "제1기숙사식당메뉴"],
    "2기숙사 식당": ["2긱", "2기숙사", "제2기숙사", "2기숙사식당", "2기숙사 식당메뉴"],
    "2호관(교직원)식당": ["2호관", "교직원", "교직원식당", "2호관식당", "2호관(교직원)식당"],
    "27호관식당": ["27호관", "이공계", "이공계식당", "27호관식당메뉴"],
    "사범대식당": ["사범대", "사범관", "사범대식당메뉴"],
    "전체": ["모두", "all", "ALL", "전체식당"],
    # Bus Stops & Terminals
    "공대": ["공과대학", "공대", "공학관", "공대앞", "공과대"],
    "자연대": ["자연과학대학", "자연대", "자연과학대"],
    "정문": ["대학본부", "본부", "정문"],
    "인천대입구역": ["인입", "인천대입구", "인천대입구역", "2번출구", "1번출구", "인입런"],
    # Semester Terms
    "FIRST": ["1학기", "1", "first", "FIRST", "10", "1학기수업", "1st"],
    "SECOND": ["2학기", "2", "second", "SECOND", "20", "2학기수업", "2nd"],
    "SUMMER": ["여름학기", "여름계절학기", "계절학기", "summer", "30"],
    "WINTER": ["겨울학기", "겨울계절학기", "winter", "40"],
    # Academic Departments (OpenAPI Enum)
    "COMPUTER_ENGINEERING": ["컴퓨터공학부", "컴퓨터공학과", "컴공", "컴퓨터공학", "컴퓨터", "computer_engineering"],
    "INFORMATION_COMMUNICATION_ENGINEERING": ["정보통신공학과", "정통", "정보통신", "정통과"],
    "EMBEDDED_SYSTEM": ["임베디드시스템공학과", "임베디드", "임베"],
    "BUSINESS_ADMINISTRATION": ["경영학부", "경영학과", "경영"],
    "ECONOMICS": ["경제학과", "경제학부", "경제"],
    "DATA_SCIENCE": ["데이터사이언스학과", "데이터사이언스", "데사"],
    "SAFETY_ENGINEERING": ["안전공학과", "안전공학부", "안전"],
    "MECHANICAL_ENGINEERING": ["기계공학과", "기계공학부", "기계"],
    "ELECTRICAL_ENGINEERING": ["전기공학과", "전기공학부", "전기"],
    "ELECTRONICS_ENGINEERING": ["전자공학과", "전자공학부", "전자"],
    "MEDIA_COMMUNICATION": ["미디어커뮤니케이션학과", "미디어커뮤니케이션", "미컴"],
    "SOCIAL_WELFARE": ["사회복지학과", "사회복지", "사복"],
    "FASHION": ["패션산업학과", "패션산업", "패디"],
    "LIFE_SCIENCE": ["생명과학부", "생명과학과", "생과"],
    "BIOENGINEERING": ["생명공학부", "생명공학과", "생공"],
}


class SchemaCoercer:
    """
    Validates and coerces tool call arguments dynamically against a JSON Schema.
    Ensures LLM inputs comply with tool definitions without hardcoded engine branches.
    """

    @classmethod
    def coerce(cls, schema: Optional[Dict[str, Any]], args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Dynamically coerces argument types and enum values based on the tool's inputSchema.
        """
        if not schema or not isinstance(args, dict):
            return dict(args or {})

        properties = schema.get("properties", {})
        if not properties:
            return dict(args)

        coerced = dict(args)

        # 0. Parameter Alias Coercion (e.g. query <-> q <-> keyword)
        if "q" in properties and (not coerced.get("q")):
            for alias in ["query", "keyword", "searchWord", "searchTerm", "search"]:
                if coerced.get(alias):
                    coerced["q"] = coerced.pop(alias)
                    break
        elif "query" in properties and (not coerced.get("query")):
            for alias in ["q", "keyword", "searchWord", "searchTerm", "search"]:
                if coerced.get(alias):
                    coerced["query"] = coerced.pop(alias)
                    break
        elif "keyword" in properties and (not coerced.get("keyword")):
            for alias in ["q", "query", "searchWord", "searchTerm", "search"]:
                if coerced.get(alias):
                    coerced["keyword"] = coerced.pop(alias)
                    break

        # Academic Year & Term Auto-Defaults
        if "year" in properties and (not coerced.get("year")):
            coerced["year"] = datetime.now(timezone(timedelta(hours=9))).year
        if "term" in properties and (not coerced.get("term")):
            curr_month = datetime.now(timezone(timedelta(hours=9))).month
            coerced["term"] = "SECOND" if curr_month >= 7 else "FIRST"

        # Course offerings parameter resilience:
        # If deptName is passed, always populate keyword as fallback to avoid 400 Bad Request
        if "deptName" in properties and coerced.get("deptName"):
            raw_dept = str(coerced["deptName"]).strip()
            if "keyword" in properties and not coerced.get("keyword"):
                coerced["keyword"] = raw_dept
            dept_prop = properties.get("deptName", {})
            dept_enums = dept_prop.get("enum", [])
            # If backend OpenAPI enum is mangled or doesn't match raw_dept, pop deptName so keyword handles search
            if dept_enums and raw_dept not in dept_enums:
                coerced.pop("deptName", None)

        for param, prop in properties.items():
            if param not in coerced:
                continue

            val = coerced[param]
            p_type = prop.get("type")
            p_enum = prop.get("enum")
            p_desc = str(prop.get("description", "")).lower()

            # 1. Type Coercion: Integer
            if p_type == "integer" and val is not None:
                if str(val).strip().upper() in ["TODAY", "오늘", "NOW", "지금"]:
                    # If parameter is related to day of week, coerce to KST weekday (1=Mon ... 7=Sun)
                    if any(kw in p_desc for kw in ["요일", "day", "week"]):
                        coerced[param] = datetime.now(timezone(timedelta(hours=9))).weekday() + 1
                    else:
                        coerced[param] = datetime.now(timezone(timedelta(hours=9))).day
                else:
                    try:
                        coerced[param] = int(val)
                    except (ValueError, TypeError):
                        logger.debug(f"SchemaCoercer: could not coerce {param}={val} to integer")

            # 2. Type Coercion: Boolean
            elif p_type == "boolean" and val is not None:
                if isinstance(val, str):
                    clean = val.strip().lower()
                    if clean in ["true", "1", "yes", "y", "t"]:
                        coerced[param] = True
                    elif clean in ["false", "0", "no", "n", "f"]:
                        coerced[param] = False

            # 3. Enum Coercion (Case-insensitivity & Semantic Synonyms)
            if p_enum and isinstance(val, str):
                val_clean = val.strip()

                # Exact match
                if val_clean in p_enum:
                    continue

                # Case-insensitive match
                ci_matched = False
                for e in p_enum:
                    if str(e).lower() == val_clean.lower():
                        coerced[param] = e
                        ci_matched = True
                        break
                if ci_matched:
                    continue

                # Synonym matching
                syn_matched = False
                for target_val, syn_list in CAMPUS_SYNONYMS.items():
                    if target_val in p_enum and (val_clean in syn_list or any(syn in val_clean for syn in syn_list)):
                        coerced[param] = target_val
                        syn_matched = True
                        break

                if not syn_matched:
                    # If value still invalid and "AUTO" or "전체" exists in enum, fallback gracefully
                    for fallback in ["AUTO", "전체", "ALL"]:
                        if fallback in p_enum:
                            coerced[param] = fallback
                            break

        return coerced
