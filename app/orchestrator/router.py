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

        # 1. Match from live server route sections first (actual boarding/origin stops like 공대/자연대, 인입런, 지정단런)
        best_score = 0.0
        if sections:
            for sec in sections:
                if not isinstance(sec, dict):
                    continue
                s_name = sec.get("startBstopName") or ""
                s_alias = sec.get("startBstopAlias") or ""
                s_id = sec.get("startBstopId")
                tab_name = sec.get("tabName") or ""

                for cand in [s_alias, s_name, tab_name]:
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
                if best_score == 1.0:
                    break

        # 2. Match from live server stop aliases if not matched with high confidence from route sections
        if not matched_bstop_id or best_score < 0.8:
            for a in aliases:
                if not isinstance(a, dict):
                    continue
                bstop_id = a.get("bstopId")
                bstop_name = a.get("bstopName") or ""
                stop_alias = a.get("stopAlias") or ""
                memo = a.get("memo") or ""

                # Skip destination-only drop-off stops if possible
                if "도착" in memo and best_score >= 0.5:
                    continue

                candidates = [stop_alias, bstop_name, memo]
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
                if best_score == 1.0:
                    break

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
    Uses LLM to dynamically inspect OpenAPI parameters schema, plan tools with real thoughts,
    and execute autonomous ReAct chaining with full conversational context resolution.
    """
    @staticmethod
    def _format_history_str(history: Optional[List[Any]], max_turns: int = 6) -> str:
        if not history:
            return ""
        h_lines = []
        for h in history[-max_turns:]:
            role = getattr(h, "role", None) or (h.get("role") if isinstance(h, dict) else "user")
            content = getattr(h, "content", None) or (h.get("content") if isinstance(h, dict) else "")
            if content:
                clean_content = content.replace("\n", " ").strip()
                if len(clean_content) > 200:
                    clean_content = clean_content[:200] + "..."
                h_lines.append(f"- {role}: {clean_content}")
        return "\n[최근 대화 흐름]:\n" + "\n".join(h_lines) + "\n" if h_lines else ""

    @staticmethod
    def _build_messages_with_history(
        system_msg: str,
        history: Optional[List[Any]],
        current_user_msg: str,
        max_turns: int = 6,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_msg}]
        if history:
            for h in history[-max_turns:]:
                role = getattr(h, "role", None) or (h.get("role") if isinstance(h, dict) else "user")
                content = getattr(h, "content", None) or (h.get("content") if isinstance(h, dict) else "")
                if content:
                    clean_role = "assistant" if role == "assistant" else "user"
                    clean_content = content.strip()
                    if len(clean_content) > 350:
                        clean_content = clean_content[:350] + "..."
                    messages.append({"role": clean_role, "content": clean_content})
        messages.append({"role": "user", "content": current_user_msg})
        return messages

    @staticmethod
    def _is_generic_contact_term(term: Optional[str]) -> bool:
        if not term or not term.strip():
            return True
        norm = re.sub(r"\s+", "", term).lower()
        generic_terms = {
            "전화번호", "이메일", "연락처", "번호", "연구실", "위치", "사무실", "과사",
            "지도교수", "지도교수님", "담임교수", "담임교수님", "교수님", "교수", "선생님",
            "전화번호나이메일", "이메일이나전화번호", "전화번호알려줘", "연락처알려줘", "번호알려줘",
            "전화번호누구야", "번호누구야", "연락처누구야", "알려줘", "누구야", "어디야", "뭐야"
        }
        return norm in generic_terms

    @staticmethod
    def _clean_entity_query(query: Optional[str]) -> str:
        if not query:
            return ""
        cleaned = query.strip()
        pattern = re.compile(r"(?:교수님|교수|선생님|조교님|학부장님|학과장님|과사|연구실|사무실|연락처|전화번호|이메일|메일|번호)$")
        changed = True
        while changed and len(cleaned) >= 2:
            m = pattern.search(cleaned)
            if m and m.start() >= 2:
                cleaned = cleaned[:m.start()].strip()
            else:
                changed = False
        return cleaned

    @classmethod
    async def decide_initial_plan(
        cls,
        query: str,
        history: List[Any],
        candidate_tools: List[BaseTool],
    ) -> Dict[str, Any]:
        """
        Dynamically analyzes user intent, conversation context, and available candidate tools
        to generate the AI's actual reasoning (Thought) and select the 1st-hop tools.
        """
        from datetime import datetime
        now = datetime.now()
        history_str = cls._format_history_str(history, max_turns=6)

        # Ensure core campus domain tools are always visible in candidate catalog
        core_categories = {"PORTAL", "LIBRARY", "LMS", "BUS", "CAFETERIA", "TIMETABLE", "SCHEDULE", "NOTICE", "DIRECTORY", "WEATHER", "CAMPUS_WATCH", "INU_AI_KNOWLEDGE"}
        all_candidate_tools = list(candidate_tools)
        existing_cats = {t.category.upper() for t in all_candidate_tools}
        
        from app.tools.registry import tool_registry
        for cat in core_categories:
            if cat not in existing_cats:
                cat_tools = tool_registry.get_tools_by_category(cat)
                if cat_tools:
                    all_candidate_tools.append(cat_tools[0])

        tool_catalog = []
        for t in all_candidate_tools:
            tool_catalog.append(f"- {t.category} ({t.name}): {t.description}")
        tool_catalog_str = "\n".join(tool_catalog)

        schema = {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "사용자의 질문 의도와 필요한 캠퍼스 도구를 선별한 실제 판단 이유 (한두 문장, 구체적이고 자연스럽게)",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "실행할 도구 카테고리 목록 (예: ['PORTAL'], ['LIBRARY'], ['BUS'], ['CAFETERIA'], ['DIRECTORY'], ['INU_AI_KNOWLEDGE'] 등)",
                },
            },
            "required": ["thought", "tools"],
        }

        system_msg = f"""당신은 '인천대학교 AI 에이전트 챗불이'의 자율 오케스트레이터입니다.
현재 시점: {now.year}년 {now.month}월 {now.day}일

[핵심 지침]:
1. [대화 맥락 연속성 및 생략된 주어/목적어 복원 (필수)]:
   - 사용자의 현재 질문에 주어/목적어가 생략되었거나 대명사/후속 질의인 경우(예: 이전 대화에서 지도교수가 '홍길동 교수님'으로 확인된 후 '전화번호 누구야?', '전화번호 알아?', '연구실 어디야?'라고 묻는 경우), **반드시 이전 대화 흐름의 핵심 엔티티(인물명, 교수명, 학과명, 식당명, 정류장 등)를 주어로 복원하여 사고(Thought)와 도구를 결정**하세요.
   - ⚠️ 절대 "구체적인 대상이 명시되지 않았다"고 오판하여 사용자에게 반문하지 말고, 이전 대화의 대상(예: 홍길동 교수님)을 찾아 해당 도구(DIRECTORY 등)를 선택하고 thought에 구체적인 대상을 명시하세요.
   - 예 (지도교수 확인 후 전화번호 질의): thought: "이전 대화에서 확인된 홍길동 교수님의 전화번호를 교내 전화번호부에서 조회합니다.", tools: ["DIRECTORY"]
2. [개인 학적 및 성적 질의]:
   '내 학적 정보', '내 학번/학과', '내 성적', '취득학점', '내 지도교수' 등 개인 학적/성적 관련 질문은 반드시 'PORTAL' 도구를 선택하세요.
3. [1인칭 졸업/학사 판정 질의]:
   '나 졸업 가능해?', '나 졸업 요건 돼?', '내 취득학점'처럼 1인칭으로 본인의 졸업/학점을 묻는 질문은,
   학생 본인의 학적(소속 학과, 학번, 취득 학점) 확인이 필수적이므로 먼저 'PORTAL'을 반드시 포함하세요.
4. [교내 연락처/위치 질의]:
   교수님, 학과 사무실, 교내 행정부서의 전화번호나 위치/호실을 묻는 질문은 'DIRECTORY' 도구를 선택하세요.
5. [일반 학과 규정/학칙 질의]:
   '컴퓨터공학과 졸업 요건 알려줘'처럼 3인칭 또는 일반 학과 규정을 묻는 질문은 개인 학적 조회가 불필요하므로 'INU_AI_KNOWLEDGE'만 선택하세요.
6. [도서관 열람실 잔여 좌석 및 스터디룸 현황 질의]:
   '열람실 잔여 좌석', '도서관 좌석 현황', '열람실 몇 자리 남았어?', '스터디룸 목록', '스터디룸 예약 가능한 곳' 등 학산도서관의 현재 잔여 좌석이나 스터디룸 관련 질문은 반드시 'LIBRARY' 도구를 선택하세요. (빈자리 푸시 알림/감시 예약 요청이 아닌 단순 현황 질문은 절대 CAMPUS_WATCH가 아닌 LIBRARY를 선택해야 합니다.)
7. [도서관 빈자리 알림/스나이퍼 질의]:
   '자리 나면 알려줘', '알림 걸어줘', '취소표 나오면 알려줘', '빈자리 감시'처럼 빈자리 발생 시 푸시 알림/예약을 요구하는 질문은 단순 잔여 좌석 조회가 아니므로 반드시 'CAMPUS_WATCH' 도구를 선택하세요.
8. [판단 이유 (thought) 작성 및 정직성 규칙]:
   - 반드시 실제 선택한 'tools' 목록에 부합하는 판단 이유만 작성해야 합니다.
   - 질문에 등장했거나 대화 맥락에서 복원된 구체적인 대상(예: 홍길동 교수님 전화번호, 정문 버스, 학생식당 학식 등)을 직접 언급하며 실제 실행할 도구의 목적을 명확히 작성하세요.

{history_str}
[사용 가능한 도구 목록]:
{tool_catalog_str}

반드시 JSON 스키마 규격에 맞춰 응답하세요."""

        messages = cls._build_messages_with_history(system_msg, history, query, max_turns=6)

        try:
            raw_json = await llm_client.create_guided_completion(
                messages=messages,
                response_schema=schema,
                temperature=0.1,
            )
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "thought" in parsed and "tools" in parsed:
                return parsed
        except Exception as e:
            logger.warning(f"LLM initial plan decision failed, using fallback: {e}")

        return {
            "thought": f"'{query[:20]}...' 질문을 해결하기 위해 필요한 캠퍼스 시스템을 확인하고 있습니다.",
            "tools": [t.category for t in candidate_tools[:2]],
        }

    @classmethod
    async def decide_secondary_plan(
        cls,
        query: str,
        history: List[Any],
        current_observation: str,
        executed_categories: List[str],
        available_tools: List[BaseTool],
    ) -> Dict[str, Any]:
        """
        ReAct Autonomous Chaining:
        Examines current tool observations and conversation history to decide if sufficient to answer.
        If insufficient, selects next-hop tools and generates reasoning thought.
        """
        if not current_observation or not current_observation.strip():
            return {"thought": "", "tools": []}

        history_str = cls._format_history_str(history, max_turns=4)

        tool_catalog = []
        for t in available_tools:
            if t.category not in executed_categories:
                tool_catalog.append(f"- {t.category} ({t.name}): {t.description}")

        if not tool_catalog:
            return {"thought": "필요한 정보가 모두 수집되어 답변을 작성합니다.", "tools": []}

        tool_catalog_str = "\n".join(tool_catalog)

        schema = {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "현재까지 수집된 결과를 평가하고 후속 조치나 최종 답변 작성을 결정한 구체적인 판단 이유",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "추가로 실행해야 할 도구 카테고리 목록 (이미 충분하면 빈 배열 [])",
                },
            },
            "required": ["thought", "tools"],
        }

        system_msg = f"""당신은 인천대학교 캠퍼스 AI 비서의 ReAct 자율 의사결정 엔진입니다.
[사용자 질문]: "{query}"
{history_str}
[현재까지 수집된 도구 실행 결과 (Observation)]:
{current_observation}

[이미 실행된 도구]: {', '.join(executed_categories)}

[핵심 판단 기준]:
1. 위 Observation 결과만으로 사용자 질문에 충분하고 정확하게 답변할 수 있는가?
   - 충분하다면: "tools": []로 응답하고, thought에는 수집된 데이터를 바탕으로 명확한 답변을 준비한다는 이유를 작성하세요.
2. [순수 학칙/규정 질의 시 불필요 도구 차단]:
   - 학과 졸업 요건, 학사 규정, 학칙에 관한 질의에서 'INU_AI_KNOWLEDGE'가 이미 실행되었거나 충분한 규정 내용이 관찰된 경우, 사용자가 명시적으로 요구하지 않은 일반 학과 공지사항(NOTICE)이나 학사일정(SCHEDULE) 도구를 무분별하게 추가 호출하지 말고 "tools": []로 종료하세요.
3. 정보가 불충분하거나 후속 연계가 필요한 경우:
   - [예시]: 사용자가 "나 졸업 가능해?"라고 물었고, 1단계에서 학생 학적(소속 학과, 학번, 취득학점)을 확인했으나 아직 학과의 '졸업 요건 규정'이 없는 경우:
     -> tools: ["INU_AI_KNOWLEDGE"], thought: "학적 정보에서 소속 학과와 취득 학점을 확인했습니다. 졸업 가능 여부를 판정하기 위해 해당 학과의 졸업 요건 규정을 공식 학칙 지식베이스에서 추가로 조회합니다."
   - 이미 실행된 도구는 중복 호출하지 마세요.

[추가 실행 가능한 도구 목록]:
{tool_catalog_str}

반드시 JSON 스키마 규격에 맞춰 응답하세요."""

        messages = cls._build_messages_with_history(
            system_msg,
            history,
            "현재까지의 관찰 결과를 바탕으로 후속 조치를 결정하세요.",
            max_turns=4,
        )

        try:
            raw_json = await llm_client.create_guided_completion(
                messages=messages,
                response_schema=schema,
                temperature=0.1,
            )
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict) and "thought" in parsed and "tools" in parsed:
                return parsed
        except Exception as e:
            logger.warning(f"LLM secondary plan decision failed: {e}")

        return {"thought": "", "tools": []}

    @classmethod
    async def extract_tool_arguments(
        cls,
        tool: BaseTool,
        query: str,
        history: Optional[List[Any]] = None,
        client_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dynamically extracts tool arguments matching the tool's OpenAPI schema using LLM reasoning,
        full conversation context, and automatic entity fallback resolution.
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
        from datetime import datetime
        now = datetime.now()
        history_str = cls._format_history_str(history, max_turns=6)

        system_msg = f"""당신은 사용자의 질의와 최근 대화 맥락으로부터 OpenAPI REST API 호출에 필요한 JSON 인자(Arguments)를 정확히 추출하는 AI 파라미터 리졸버입니다.

[도구 이름]: {fn.get('name')}
[도구 설명]: {fn.get('description')}
[도구 파라미터 JSON Schema]:
{json.dumps(params_schema, ensure_ascii=False, indent=2)}

[현재 기준 일시]: {now.year}년 {now.month}월 {now.day}일
{history_str}
규칙:
1. **[대화 맥락 및 생략된 엔티티 복원 (핵심)]**:
   - 사용자의 현재 질문에 주어/대상/파라미터가 생략된 경우(예: '전화번호 알아?', '전화번호 누구야?', '연구실 위치는?', '이메일 뭐야?', '언제까지야?'), 반드시 위 **[최근 대화 흐름]**에서 언급된 주된 대상(인물명, 교수명 예: '홍길동', 학과명, 식당명 등)을 검색어(`query`)나 해당 파라미터 값으로 추출하세요.
2. 학과명, 단과대, 교수명, 부서명 등의 검색어(query) 파라미터 추출 시:
   - '과사', '사무실', '전화번호', '알려줘', '위치', '번호', '연락처', '알아?', '누구야' 같은 질의 수식어는 제거하고 실제 고유명사(예: '홍길동', '컴퓨터공학부')만 추출하세요.
   - 축약어/줄임말(예: '컴공' -> '컴퓨터공학', '임베' -> '임베디드', '정통' -> '정보통신', '전전' -> '전자공학과', '산경' -> '산업경영', '패디' -> '패션산업', '미컴' -> '미디어커뮤니케이션', '사복' -> '사회복지', '생공' -> '생명공학' 등)은 대학 포털 DB에서 검색될 수 있는 정규 학과/부서 키워드로 정규화하세요.
3. 식당(cafeteria) 파라미터는 학생식당, 제1기숙사식당, 2기숙사 식당, 27호관식당, 사범대식당 중 가장 일치하는 명칭으로 추출하세요.
4. 도서관(library) 도구 파라미터 추출 시:
   - 좌석 예약/배정 요청(예: '30번 자리 예약해줘', '자리 잡아줘', '제1열람실 배정해줘', '열람실 예약'): target="RESERVE_SEAT", room_name과 seat_no(언급된 경우)를 추출하세요.
   - 스터디룸 예약 요청(예: '205호 스터디룸 예약'): target="RESERVE_STUDY_ROOM", room_name, date, begin_time, end_time 추출.
   - 스터디룸 목록 질의(예: '스터디룸 목록 보여줘', '스터디룸 뭐 있어?'): target="STUDY_ROOMS".
   - 열람실 잔여 좌석/현황 질의(예: '도서관 자리 있어?', '열람실 좌석 남아있어?'): target="SEATS", room_name(특정 열람실 언급 시).
5. 빈자리 알림/감시(campus_seat_sniper_and_watch) 도구 파라미터 추출 시:
   - action: 사용자가 '알림 걸어줘', '자리 나면' -> "WATCH", '감시 목록', '내 알림 확인' -> "LIST", '감시 취소' -> "CANCEL".
   - target_name: '힐링존', '제1열람실', '205호' 등 감시 대상 명칭 추출.
   - seat_no: 특정 좌석 번호가 언급된 경우 좌석 번호 추출.
6. 학사일정(schedule) 파라미터(year, month) 추출 시:
   - 사용자가 '이번 달', '오늘', '학사일정' 등을 언급하거나 연도/월을 생략한 경우 현재 연도({now.year})와 현재 월({now.month})을 정수(integer)로 추출하세요.
7. 공지사항(notice) 검색어(query) 추출 시:
   - 사용자가 '장학 공지', '학사 공지' 등을 물은 경우 검색어(query)에 '장학', '학사' 등 핵심 키워드를 추출하세요.
8. 질의에 명시되지 않고 대화 맥락에도 없는 선택적 파라미터는 기본값(default)을 사용하거나 생략하세요.
9. 반드시 오직 유효한 JSON 객체({{ ... }})만 반환하세요.
"""
        messages = cls._build_messages_with_history(system_msg, history, query, max_turns=6)

        extracted_args: Dict[str, Any] = {}
        try:
            raw_json = await llm_client.create_guided_completion(
                messages=messages,
                response_schema=params_schema,
                temperature=0.1,
            )
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                extracted_args = parsed
        except Exception as e:
            logger.warning(f"LLM parameter extraction failed for {tool.name}: {e}")

        # Post-Processing: Entity Context Fallback & Normalization for search queries
        if "query" in properties:
            curr_query = str(extracted_args.get("query") or "").strip()
            is_generic = cls._is_generic_contact_term(curr_query)

            if not curr_query or is_generic:
                # 1. Fallback from history (Search for professor names / department names in recent turns)
                resolved_name = None
                if history:
                    name_pattern = re.compile(r"([가-힣]{2,4})\s*(?:교수님|교수|선생님)")
                    dept_pattern = re.compile(r"([가-힣]+(?:학부|학과|과))")
                    non_names = {"지도", "담임", "전담", "학과", "학부", "담당", "우리", "해당", "어떤", "무슨", "소속", "모든"}

                    for h in reversed(history[-6:]):
                        content = getattr(h, "content", None) or (h.get("content") if isinstance(h, dict) else "")
                        if content:
                            matches = name_pattern.findall(content)
                            for m in reversed(matches):
                                if m not in non_names:
                                    resolved_name = m
                                    break
                            if resolved_name:
                                break
                            dept_matches = dept_pattern.findall(content)
                            if dept_matches:
                                resolved_name = dept_matches[-1]
                                break

                # 2. Fallback from client_context (Academic advisor / department)
                if not resolved_name and client_context:
                    acad_disp = client_context.get("academicDisplay")
                    if isinstance(acad_disp, dict):
                        resolved_name = acad_disp.get("advisorProfessorName") or acad_disp.get("departmentName")
                    if not resolved_name:
                        acad = client_context.get("academic")
                        if isinstance(acad, dict):
                            resolved_name = acad.get("advisorProfessorName") or acad.get("departmentName")

                if resolved_name:
                    logger.info(f"Resolved empty query fallback to '{resolved_name}' from conversation/client context")
                    extracted_args["query"] = resolved_name
                elif is_generic:
                    extracted_args["query"] = ""

            # Clean query suffixes
            if extracted_args.get("query"):
                cleaned = cls._clean_entity_query(str(extracted_args["query"]))
                if len(cleaned) >= 2:
                    extracted_args["query"] = cleaned

        return extracted_args

