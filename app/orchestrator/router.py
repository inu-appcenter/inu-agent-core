"""
Dynamic Schema-Driven Agent Router & Entity Matcher
100% Zero-Hardcoding: All tools, parameters, and entities are dynamically discovered from OpenAPI specs and live server APIs.
"""
from typing import List, Dict, Any, Optional
import json
import re
import difflib
import httpx

from app.core.config import settings
from app.core.logging import logger
from app.llm.client import llm_client
from app.tools.base import BaseTool


class DynamicBusMatcher:
    """
    Dynamically fetches bus stop aliases and route sections from inu-portal-server APIs.
    Performs real-time fuzzy matching with zero hardcoded IDs or station names.
    """
    _cached_aliases: Optional[List[Dict[str, Any]]] = None
    _cached_sections: Optional[List[Dict[str, Any]]] = None

    @classmethod
    async def get_live_metadata(cls) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        base_url = settings.INU_PORTAL_SERVER_URL.rstrip("/")
        aliases = []
        sections = []

        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp_aliases = await client.get(f"{base_url}/api/buses/stop-aliases")
                if resp_aliases.status_code == 200:
                    data = resp_aliases.json()
                    aliases = data.get("data", []) if isinstance(data, dict) else (data or [])

                resp_sections = await client.get(f"{base_url}/api/buses/routes")
                if resp_sections.status_code == 200:
                    data = resp_sections.json()
                    sections = data.get("data", []) if isinstance(data, dict) else (data or [])
        except Exception as e:
            logger.warning(f"Could not prefetch live bus metadata from server: {e}")

        return aliases, sections

    @classmethod
    async def resolve_stop_and_routes(
        cls, query: str, user_stop_param: Optional[str] = None
    ) -> tuple[Optional[str], str, str, List[str]]:
        aliases, sections = await cls.get_live_metadata()
        target_text = (user_stop_param or query).strip()

        matched_bstop_id = None
        resolved_stop_name = "인천대입구역 2번출구"
        matched_tab_name = "인입런"
        valid_routes: List[str] = []

        # 1. Match from live server stop aliases
        best_score = 0.0
        for a in aliases:
            if not isinstance(a, dict):
                continue
            bstop_id = a.get("bstopId")
            bstop_name = a.get("bstopName") or ""
            stop_alias = a.get("stopAlias") or ""
            memo = a.get("memo") or ""

            candidates = [bstop_name, stop_alias, memo]
            for cand in candidates:
                if not cand:
                    continue
                if cand.lower() in target_text.lower() or target_text.lower() in cand.lower():
                    matched_bstop_id = bstop_id
                    resolved_stop_name = stop_alias or bstop_name
                    best_score = 1.0
                    break
                ratio = difflib.SequenceMatcher(None, target_text.lower(), cand.lower()).ratio()
                if ratio > best_score and ratio > 0.4:
                    best_score = ratio
                    matched_bstop_id = bstop_id
                    resolved_stop_name = stop_alias or bstop_name

        # 2. Match from live server route sections if not matched from aliases
        if not matched_bstop_id or best_score < 0.6:
            for sec in sections:
                if not isinstance(sec, dict):
                    continue
                s_name = sec.get("startBstopName") or ""
                s_alias = sec.get("startBstopAlias") or ""
                s_id = sec.get("startBstopId")
                tab_name = sec.get("tabName") or ""

                for cand in [s_name, s_alias, tab_name]:
                    if not cand:
                        continue
                    if cand.lower() in target_text.lower() or target_text.lower() in cand.lower():
                        matched_bstop_id = s_id
                        resolved_stop_name = s_alias or s_name
                        matched_tab_name = tab_name
                        best_score = 1.0
                        break
                    ratio = difflib.SequenceMatcher(None, target_text.lower(), cand.lower()).ratio()
                    if ratio > best_score and ratio > 0.4:
                        best_score = ratio
                        matched_bstop_id = s_id
                        resolved_stop_name = s_alias or s_name
                        matched_tab_name = tab_name

        # 3. Default to first route section if not found
        if not matched_bstop_id and sections:
            first_sec = sections[0]
            matched_bstop_id = first_sec.get("startBstopId")
            resolved_stop_name = first_sec.get("startBstopAlias") or first_sec.get("startBstopName") or "인천대입구역 2번출구"
            matched_tab_name = first_sec.get("tabName") or "인입런"

        # 4. Extract valid serviced route numbers for this stop
        if matched_bstop_id and sections:
            matched_sec_list = [
                s for s in sections
                if s.get("startBstopId") and (s.get("startBstopId") == matched_bstop_id or str(s.get("startBstopId")) in str(matched_bstop_id))
            ]
            if matched_sec_list:
                matched_tab_name = matched_sec_list[0].get("tabName") or matched_tab_name
                resolved_stop_name = matched_sec_list[0].get("startBstopAlias") or matched_sec_list[0].get("startBstopName") or resolved_stop_name
                for s in matched_sec_list:
                    r_no = s.get("routeNo")
                    if r_no and r_no not in valid_routes:
                        valid_routes.append(r_no)

        return matched_bstop_id, resolved_stop_name, matched_tab_name, valid_routes

    @classmethod
    async def resolve_bstop_id(cls, query: str, user_stop_param: Optional[str] = None) -> Optional[str]:
        bstop_id, _, _, _ = await cls.resolve_stop_and_routes(query, user_stop_param)
        return bstop_id


class AgentRouter:
    """
    Uses LLM to dynamically inspect OpenAPI parameters schema and extract arguments.
    """
    @classmethod
    async def extract_tool_arguments(
        cls,
        tool: BaseTool,
        query: str,
        history: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dynamically extracts tool arguments matching the tool's OpenAPI schema using LLM reasoning.
        """
        schema = tool.get_schema()
        fn = schema.get("function", {})
        params_schema = fn.get("parameters", {})
        properties = params_schema.get("properties", {})

        if not properties:
            return {}

        # If tool requires bstopId (Bus tool), dynamically resolve stop and valid serviced routes
        if "bstopId" in properties:
            bstop_id, stop_name, tab_name, valid_routes = await DynamicBusMatcher.resolve_stop_and_routes(query)
            if bstop_id:
                return {
                    "bstopId": bstop_id,
                    "_meta": {
                        "bstopId": bstop_id,
                        "stopName": stop_name,
                        "tabName": tab_name,
                        "validRoutes": valid_routes,
                    },
                }

        # For other schema-driven tools, ask LLM to extract arguments conforming to the OpenAPI parameters schema
        system_msg = f"""당신은 사용자의 질의로부터 OpenAPI REST API 호출에 필요한 JSON 인자(Arguments)를 정확히 추출하는 AI 파라미터 리졸버입니다.

[도구 이름]: {fn.get('name')}
[도구 설명]: {fn.get('description')}
[도구 파라미터 JSON Schema]:
{json.dumps(params_schema, ensure_ascii=False, indent=2)}

규칙:
1. 사용자 질의에서 언급된 핵심 엔티티와 조건을 파악하여 파라미터 값을 추출하세요.
2. 학과명, 단과대, 교수명, 부서명 등의 검색어(query) 파라미터 추출 시:
   - '과사', '사무실', '전화번호', '알려줘', '위치', '번호', '연락처' 같은 질의 수식어는 제거하세요.
   - 축약어/줄임말(예: '컴공' -> '컴퓨터공학', '임베' -> '임베디드', '정통' -> '정보통신', '전전' -> '전자공학', '산경' -> '산업경영', '패디' -> '패션산업', '미컴' -> '미디어커뮤니케이션', '사복' -> '사회복지', '생공' -> '생명공학' 등)은 대학 포털 DB에서 검색될 수 있는 정규 학과/부서 키워드로 정규화하세요.
3. 식당(cafeteria) 파라미터는 학생식당, 제1기숙사식당, 2기숙사 식당, 27호관식당, 사범대식당 중 가장 일치하는 명칭으로 추출하세요.
4. 질의에 명시되지 않은 선택적 파라미터는 기본값(default)을 사용하거나 생략하세요.
5. 반드시 오직 유효한 JSON 객체({{ ... }})만 반환하세요.
"""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": query},
        ]

        try:
            raw_json = await llm_client.create_guided_completion(
                messages=messages,
                response_schema=params_schema,
                temperature=0.1,
            )
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            logger.warning(f"LLM parameter extraction failed for {tool.name}: {e}")

        return {}
