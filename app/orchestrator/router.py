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
    async def resolve_bstop_id(cls, query: str, user_stop_param: Optional[str] = None) -> Optional[str]:
        aliases, sections = await cls.get_live_metadata()
        target_text = (user_stop_param or query).strip()

        # 1. Direct or fuzzy match against live server aliases
        best_bstop_id = None
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
                    return bstop_id
                # Levenshtein ratio
                ratio = difflib.SequenceMatcher(None, target_text.lower(), cand.lower()).ratio()
                if ratio > best_score and ratio > 0.4:
                    best_score = ratio
                    best_bstop_id = bstop_id

        if best_bstop_id:
            return best_bstop_id

        # 2. Match against live server route sections
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
                    return s_id
                ratio = difflib.SequenceMatcher(None, target_text.lower(), cand.lower()).ratio()
                if ratio > best_score and ratio > 0.4:
                    best_score = ratio
                    best_bstop_id = s_id

        if best_bstop_id:
            return best_bstop_id

        # 3. Default: return the primary active section from live server data if available
        if sections:
            first_sec = sections[0]
            if isinstance(first_sec, dict) and first_sec.get("startBstopId"):
                return first_sec.get("startBstopId")

        return None


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

        # If tool requires bstopId (Bus tool), dynamically resolve from live server metadata
        if "bstopId" in properties:
            resolved_id = await DynamicBusMatcher.resolve_bstop_id(query)
            if resolved_id:
                return {"bstopId": resolved_id}

        # For other schema-driven tools, ask LLM to extract arguments conforming to the OpenAPI parameters schema
        system_msg = f"""당신은 사용자의 질의로부터 API 호출에 필요한 JSON 인자(Arguments)를 정확히 추출하는 AI 파라미터 리졸버입니다.

[도구 이름]: {fn.get('name')}
[도구 설명]: {fn.get('description')}
[도구 파라미터 JSON Schema]:
{json.dumps(params_schema, ensure_ascii=False, indent=2)}

규칙:
1. 사용자 질의에서 언급된 정보를 바탕으로 파라미터 값을 추출하세요.
2. 질의에 명시되지 않은 선택적 파라미터는 기본값(default)을 사용하거나 생략하세요.
3. 반드시 오직 유효한 JSON 객체({{ ... }})만 반환하세요.
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
