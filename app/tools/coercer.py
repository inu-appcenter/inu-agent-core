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
