"""
Core Agent Orchestration Engine
Integrates Tool Registry, Tool Pruner (Gemma 27B optimization), and ReAct execution with SSE streaming.
"""
from typing import AsyncGenerator, List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import json
import re

from app.core.logging import logger
from app.core.anonymizer import (
    sanitize_text,
    sanitize_academic_record,
    build_anonymized_academic_summary,
)
from app.llm.client import llm_client
from app.llm.schemas import ChatRequest, AgentStreamEvent, ChatActionCallbackRequest
from app.orchestrator.prompts import (
    get_tool_orchestration_prompt,
    get_system_prompt_for_client,
    is_out_of_scope_question,
    OUT_OF_SCOPE_REFUSAL_MESSAGE,
)
from app.orchestrator.retriever import tool_retriever
from app.orchestrator.card_synthesizer import CardSynthesizer
from app.orchestrator.router import AgentRouter
from app.tools.base import BaseTool
from app.tools.registry import tool_registry
from app.tools.coercer import SchemaCoercer


def resolve_tool_display_name(category: str, name: str) -> str:
    category_map = {
        "PORTAL": "포털 종합정보(ERP) 학적 데이터",
        "LMS": "이러닝(LMS) 강의 및 과제",
        "LIBRARY": "학산도서관 좌석/열람실 시스템",
        "CAMPUS_WATCH": "학산도서관 실시간 빈자리 알림 예약",
        "CAFETERIA": "학생식당 메뉴 정보",
        "BUS": "인천시 실시간 시내버스 도착 정보",
        "NOTICE": "학교 공식 공지사항",
        "SCHEDULE": "학사 일정 정보",
        "TIMETABLE": "학사 강의 시간표 정보",
        "DIRECTORY": "교내 부서/학과 연락처",
        "WEATHER": "캠퍼스 날씨 정보",
        "INU_AI_KNOWLEDGE": "인천대학교 학칙·규정 지식베이스",
        "REMINDER": "맞춤 푸시 알림 예약 및 관리",
        "DAILY_BRIEF": "데일리 브리프 아침 일정 설정",
        "KEYWORD": "공지사항 키워드 알림 구독",
        "SETTINGS": "내 맞춤 알림 및 브리프 설정 종합",
        "SEARCH": "인천대학교 전 도메인 고도화 통합 검색",
        "CLUB": "교내 동아리 목록 정보",
        "LOST_PROPERTY": "학내 분실물 습득/신고 정보",
        "TIMETABLE_GAP": "시간표 공강 및 여유 시간 분석",
    }
    if category in category_map:
        return category_map[category]

    name_lower = name.lower()
    if "unified" in name_lower or "search" in name_lower:
        return "인천대학교 전 도메인 고도화 통합 검색"
    elif "watch" in name_lower or "sniper" in name_lower:
        return "학산도서관 실시간 빈자리 알림 예약"
    elif "timetable" in name_lower:
        return "학사 강의 시간표 정보"
    elif "cafeteria" in name_lower or "menu" in name_lower:
        return "학생식당 메뉴 정보"
    elif "bus" in name_lower:
        return "인천시 실시간 시내버스 도착 정보"
    elif "notice" in name_lower:
        return "학교 공식 공지사항"
    elif "weather" in name_lower:
        return "캠퍼스 날씨 정보"
    elif "directory" in name_lower or "contact" in name_lower:
        return "교내 부서 및 연락처"
    elif "library" in name_lower or "seat" in name_lower:
        return "학산도서관 시스템"

    return name


class AgentOrchestrator:
    async def run_stream(self, request: ChatRequest) -> AsyncGenerator[AgentStreamEvent, None]:
        """
        Execute streaming chat with Native OpenAI Tool Calling, autonomous ReAct chaining,
        SDUI generative card synthesis, and INUChat direct pass-through.
        """
        # 0. Out-of-scope hard guardrail (Coding, pure math, unrelated universities)
        if is_out_of_scope_question(request.message):
            logger.info(f"Out-of-scope question detected: '{request.message[:30]}...' -> streaming refusal.")
            yield AgentStreamEvent(event_type="TOKEN", content=OUT_OF_SCOPE_REFUSAL_MESSAGE)
            yield AgentStreamEvent(event_type="DONE")
            return

        # 1. Index and retrieve candidate tools
        if not tool_registry.list_tools():
            await tool_registry.initialize_all_tools()
        if not tool_retriever.is_indexed:
            await tool_retriever.index_tools(tool_registry.list_tools())

        candidate_tools = await tool_retriever.retrieve(
            query=request.message,
            history=request.history,
            top_k=8,
        )

        # Ensure essential core domain tools are available in the candidate pool
        core_categories = [
            "SEARCH", "PORTAL", "LMS", "DIRECTORY", "INU_AI_KNOWLEDGE", "LIBRARY",
            "CAFETERIA", "BUS", "TIMETABLE", "CAMPUS_WATCH", "NOTICE", "SCHEDULE"
        ]
        existing_cats = {t.category.upper() for t in candidate_tools}
        for cat in core_categories:
            if cat not in existing_cats:
                cat_tools = tool_registry.get_tools_by_category(cat)
                if cat_tools:
                    candidate_tools.append(cat_tools[0])
                    existing_cats.add(cat)

        client_ctx = request.client_context or {}
        is_app = bool(client_ctx.get("isApp", False))
        raw_token = client_ctx.get("auth") or client_ctx.get("authorization", "")
        clean_token = raw_token.replace("Bearer ", "").strip() if raw_token else ""

        portal_meta = client_ctx.get("portal") if isinstance(client_ctx.get("portal"), dict) else {}
        logger.info(
            f"[AGENT_START] client={request.client} query='{request.message[:60]}' "
            f"auth={bool(clean_token)} portal_linked={portal_meta.get('linked')} "
            f"context_keys={list(client_ctx.keys())}"
        )
        logger.info(
            f"[TOOL_RETRIEVAL] candidates={[f'{t.name}({t.category})' for t in candidate_tools]}"
        )

        exec_context = {
            "auth": clean_token,
            "authorization": f"Bearer {clean_token}" if clean_token else "",
            "client": request.client,
            "query": request.message,
        }

        # 2. Build initial conversation messages & OpenAI tool schemas
        openai_tools = [t.get_schema() for t in candidate_tools]

        orchestration_prompt = get_tool_orchestration_prompt(request.client)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": orchestration_prompt}
        ]

        if request.history:
            for h in request.history[-6:]:
                role = getattr(h, "role", None) or (h.get("role") if isinstance(h, dict) else "user")
                content = getattr(h, "content", None) or (h.get("content") if isinstance(h, dict) else "")
                if content:
                    clean_role = "assistant" if role == "assistant" else "user"
                    messages.append({"role": clean_role, "content": content})

        messages.append({"role": "user", "content": request.message})

        executed_categories = set()
        executed_tool_names = set()
        emitted_cards = set()
        emitted_actions = set()
        tool_summary_text = ""
        academic_data_dict = {}
        inuchat_rag_data = None
        executed_tool_signatures = set()
        successful_search_queries = set()
        failed_tool_categories: Dict[str, str] = {}
        empty_directory_queries: Set[str] = set()
        empty_tool_categories: Dict[str, str] = {}
        directory_found_results: bool = False

        # 3. Native Tool Calling ReAct Loop (Autonomous Multi-Hop with Streaming Thought)
        MAX_HOPS = 4
        from unittest.mock import Mock
        for hop in range(MAX_HOPS):
            response = None
            streamed_any_thought = False
            if isinstance(llm_client.chat_with_tools, Mock):
                try:
                    response = await llm_client.chat_with_tools(
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto",
                        temperature=0.1,
                    )
                except Exception as ex:
                    logger.warning(f"Mocked tool calling failed: {ex}")
                    break
            else:
                try:
                    async for event_type, data in llm_client.stream_chat_with_tools(
                        messages=messages,
                        tools=openai_tools,
                        tool_choice="auto",
                        temperature=0.1,
                    ):
                        if event_type == "thinking_chunk" and data:
                            streamed_any_thought = True
                            yield AgentStreamEvent(
                                event_type="THINKING",
                                thinking=data,
                            )
                        elif event_type == "done":
                            response = data
                except Exception as ex:
                    logger.warning(f"LLM streaming tool calling step {hop + 1} failed: {ex}. Trying non-streaming fallback...")
                    try:
                        response = await llm_client.chat_with_tools(
                            messages=messages,
                            tools=openai_tools,
                            tool_choice="auto",
                            temperature=0.1,
                        )
                    except Exception as fb_ex:
                        logger.warning(f"Fallback also failed: {fb_ex}")
                        break

            if not response:
                break

            thought = response.get("thought") or ""
            content = response.get("content") or ""
            tool_calls = response.get("tool_calls") or []

            logger.info(
                f"[REACT_HOP_{hop + 1}] thought='{thought[:80]}...' "
                f"tool_calls={[tc.get('function', {}).get('name') for tc in tool_calls]}"
            )

            # If thought was not streamed token-by-token, yield it now
            if thought and not streamed_any_thought:
                yield AgentStreamEvent(
                    event_type="THINKING",
                    thinking=thought,
                )

            if not tool_calls:
                # Deterministic Multi-Hop: 지도교수 연락처 질의 시 학적 조회 후 api_directory 미실행 상태면 자동 연쇄 호출
                is_contact_query = any(kw in request.message for kw in ["연락처", "전화번호", "전화", "연구실", "번호", "이메일", "메일", "교수님", "교수", "담임교수", "지도교수", "과사", "사무실", "찾아줘"])
                advisor_name = academic_data_dict.get("advisor") or academic_data_dict.get("advisorProfessorName")
                if not advisor_name and request.client_context:
                    acad_disp = request.client_context.get("academicDisplay")
                    if isinstance(acad_disp, dict):
                        advisor_name = acad_disp.get("advisorProfessorName") or acad_disp.get("profNm")

                if is_contact_query and advisor_name and "DIRECTORY" not in executed_categories and "api_directory" not in executed_tool_names:
                    logger.info(f"Deterministic Multi-Hop: Chaining to api_directory for advisor '{advisor_name}'")
                    dir_tool = tool_registry.get_tool("api_directory")
                    if dir_tool:
                        yield AgentStreamEvent(
                            event_type="THINKING",
                            thinking=f"학적 정보에서 확인된 지도교수({advisor_name} 교수님)의 연락처 및 연구실 정보를 확인하기 위해 교내 전화번호부(api_directory)를 조회합니다.",
                        )
                        from uuid import uuid4
                        tool_calls = [{
                            "id": f"call_auto_dir_{uuid4().hex[:6]}",
                            "type": "function",
                            "function": {
                                "name": "api_directory",
                                "arguments": {"query": advisor_name},
                            }
                        }]
                elif "SEARCH" in failed_tool_categories and "NOTICE" not in executed_categories and "api_notice" not in executed_tool_names:
                    search_query = failed_tool_categories.get("SEARCH") or request.message
                    logger.info(f"Deterministic Multi-Hop: Chaining from failed SEARCH to api_notice for query '{search_query}'")
                    notice_tool = tool_registry.get_tool("api_notice") or tool_registry.get_tool("notice") or tool_registry.get_tool("action_portal_search_notices")
                    if notice_tool:
                        clean_q = AgentRouter._clean_entity_query(search_query) or search_query
                        yield AgentStreamEvent(
                            event_type="THINKING",
                            thinking=f"통합 검색 서버 응답 지연으로 인해, 학교 공지사항 검색({notice_tool.name})을 통해 '{clean_q}' 관련 정보를 대체 조회합니다.",
                        )
                        from uuid import uuid4
                        tool_calls = [{
                            "id": f"call_auto_notice_{uuid4().hex[:6]}",
                            "type": "function",
                            "function": {
                                "name": notice_tool.name,
                                "arguments": {"query": clean_q},
                            }
                        }]
                    else:
                        break
                elif ("DIRECTORY" in executed_categories or "api_directory" in executed_tool_names or any("directory" in n.lower() or "contact" in n.lower() for n in executed_tool_names)) and empty_directory_queries and "SEARCH" not in executed_categories:
                    search_tools = tool_registry.get_tools_by_category("SEARCH")
                    search_tool = search_tools[0] if search_tools else (
                        tool_registry.get_tool("api_unified_search") or tool_registry.get_tool("unifiedSearch") or tool_registry.get_tool("unified_search")
                    )
                    if search_tool:
                        fallback_q = list(empty_directory_queries)[0] or request.message
                        clean_q = AgentRouter._clean_entity_query(fallback_q) or fallback_q
                        logger.info(f"Deterministic Multi-Hop: Chaining from empty DIRECTORY to {search_tool.name} for query '{clean_q}'")
                        yield AgentStreamEvent(
                            event_type="THINKING",
                            thinking=f"교내 전화번호부 DB에 '{clean_q}' 관련 연락처가 확인되지 않아, 인천대학교 전 도메인 통합 검색({search_tool.name})을 통해 학교 공식 웹페이지 및 부서 안내에서 연락처를 추가 탐색합니다.",
                        )
                        from uuid import uuid4
                        tool_calls = [{
                            "id": f"call_auto_search_{uuid4().hex[:6]}",
                            "type": "function",
                            "function": {
                                "name": search_tool.name,
                                "arguments": {"q": clean_q, "query": clean_q},
                            }
                        }]
                    else:
                        break
                elif any(cat in failed_tool_categories or cat in empty_tool_categories for cat in ["NOTICE", "SCHEDULE", "CLUB", "LOST_PROPERTY", "TIMETABLE_GAP"]) and "SEARCH" not in executed_categories:
                    search_tools = tool_registry.get_tools_by_category("SEARCH")
                    search_tool = search_tools[0] if search_tools else (
                        tool_registry.get_tool("api_unified_search") or tool_registry.get_tool("unifiedSearch") or tool_registry.get_tool("unified_search")
                    )
                    if search_tool:
                        hit_cat = next(cat for cat in ["NOTICE", "SCHEDULE", "CLUB", "LOST_PROPERTY", "TIMETABLE_GAP"] if cat in failed_tool_categories or cat in empty_tool_categories)
                        fallback_q = failed_tool_categories.get(hit_cat) or empty_tool_categories.get(hit_cat) or request.message
                        clean_q = AgentRouter._clean_entity_query(fallback_q) or fallback_q
                        logger.info(f"Deterministic Multi-Hop: Chaining from empty/failed {hit_cat} to {search_tool.name} for query '{clean_q}'")
                        yield AgentStreamEvent(
                            event_type="THINKING",
                            thinking=f"해당 분야({hit_cat})에서 결과를 찾지 못해, 인천대학교 전 도메인 고도화 통합 검색({search_tool.name})을 통해 '{clean_q}' 관련 정보를 추가 탐색합니다.",
                        )
                        from uuid import uuid4
                        tool_calls = [{
                            "id": f"call_auto_search_{uuid4().hex[:6]}",
                            "type": "function",
                            "function": {
                                "name": search_tool.name,
                                "arguments": {"q": clean_q, "query": clean_q},
                            }
                        }]
                    else:
                        break
                else:
                    # No more tools needed, proceed to final response
                    break

            # Deduplication & Loop Prevention Guardrail:
            # 1. Check if all proposed tool calls in this hop are identical repeats of already executed calls
            all_repeated = True
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", {})
                sig = f"{fn_name}:{json.dumps(fn_args, sort_keys=True)}"
                if sig not in executed_tool_signatures:
                    all_repeated = False
                    break

            if all_repeated:
                logger.info(f"ReAct Loop break: All tool calls in hop {hop + 1} were identical repeats. Proceeding to synthesis.")
                break

            # 2. Filter out redundant search calls if a search has already succeeded with results
            filtered_calls = []
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", {})
                sig = f"{fn_name}:{json.dumps(fn_args, sort_keys=True)}"
                target_tool_check = tool_registry.get_tool(fn_name)
                is_search = target_tool_check and (target_tool_check.category == "SEARCH" or "unifiedsearch" in target_tool_check.name.lower())
                q_arg = fn_args.get("q", "")
                if is_search and q_arg in successful_search_queries:
                    logger.info(f"Skipping redundant search call for already satisfied query '{q_arg}'")
                    continue
                filtered_calls.append(tc)
                executed_tool_signatures.add(sig)

            if not filtered_calls:
                logger.info("ReAct Loop break: No new non-redundant tool calls remaining. Proceeding to synthesis.")
                break

            tool_calls = filtered_calls

            # Append assistant's tool-calling message to history
            raw_msg = response.get("raw_message") or {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            }
            messages.append(raw_msg)

            # Execute all requested tool calls in this hop
            for tc in tool_calls:
                tc_id = tc.get("id")
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", {})

                # Match tool from registry (Direct, Case-Insensitive, Category, or Alias)
                target_tool = tool_registry.get_tool(fn_name)
                if not target_tool:
                    fn_lower = fn_name.lower().replace("-", "_")
                    for t in tool_registry.list_tools():
                        t_name_lower = t.name.lower()
                        t_cat_lower = t.category.lower()
                        if t_name_lower == fn_lower or t_cat_lower == fn_lower:
                            target_tool = t
                            break
                        # Domain alias matching
                        if "cafeteria" in fn_lower or "menu" in fn_lower:
                            if "menu" in t_name_lower or t_cat_lower == "cafeteria":
                                target_tool = t
                                break
                        elif "library" in fn_lower or "seat" in fn_lower or "reading" in fn_lower:
                            if t_name_lower == "library_reading_rooms_status":
                                target_tool = t
                                break
                        elif "academic" in fn_lower or "schreg" in fn_lower or "portal" in fn_lower:
                            if "portal_get_academic" in t_name_lower:
                                target_tool = t
                                break
                        elif "contact" in fn_lower or "directory" in fn_lower:
                            if "searchdirectory" in t_name_lower or "searchcontacts" in t_name_lower or "directory" in t_name_lower:
                                target_tool = t
                                break
                        elif "knowledge" in fn_lower or "inuchat" in fn_lower or "regulation" in fn_lower:
                            if "knowledge" in t_name_lower or t_cat_lower == "inu_ai_knowledge":
                                target_tool = t
                                break

                if not target_tool:
                    logger.warning(f"Tool {fn_name} not found in registry")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": fn_name,
                        "content": json.dumps({"error": f"Tool {fn_name} not found"}, ensure_ascii=False),
                    })
                    continue

                tool_obs = ""
                async for event, summary, ac_data, rag_data in self._execute_tool(
                    tool=target_tool,
                    tool_args=fn_args,
                    request=request,
                    exec_context=exec_context,
                    emitted_cards=emitted_cards,
                    emitted_actions=emitted_actions,
                    academic_context=academic_data_dict,
                ):
                    if event:
                        yield event
                        if getattr(event, "event_type", None) == "STATUS":
                            st_state = getattr(event, "status_state", None)
                            if st_state == "failed":
                                q_val = fn_args.get("query") or fn_args.get("q") or request.message
                                failed_tool_categories[target_tool.category.upper()] = str(q_val)
                            elif st_state == "empty":
                                q_val = fn_args.get("query") or fn_args.get("q") or request.message
                                empty_tool_categories[target_tool.category.upper()] = str(q_val)
                                if target_tool.category.upper() == "DIRECTORY":
                                    empty_directory_queries.add(str(q_val))
                            elif st_state == "completed" and target_tool.category.upper() == "DIRECTORY":
                                directory_found_results = True
                    if summary:
                        tool_summary_text += summary
                        tool_obs += summary
                    if ac_data:
                        academic_data_dict.update(ac_data)
                    if rag_data:
                        inuchat_rag_data = rag_data

                executed_categories.add(target_tool.category.upper())
                executed_tool_names.add(target_tool.name)

                if (target_tool.category == "SEARCH" or "unifiedsearch" in target_tool.name.lower()) and "총 0건" not in tool_obs:
                    q_val = fn_args.get("q", "")
                    if q_val:
                        successful_search_queries.add(q_val)

                if not tool_obs:
                    tool_obs = json.dumps(
                        academic_data_dict if target_tool.category == "PORTAL" else {"status": "success"},
                        ensure_ascii=False
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": fn_name,
                    "content": tool_obs,
                })

            # If a contact query has now executed api_directory / DIRECTORY and actually found results, multi-hop info is gathered.
            is_contact_query = any(kw in request.message for kw in ["연락처", "전화번호", "전화", "연구실", "번호", "이메일", "메일", "교수님", "교수", "담임교수", "지도교수", "과사", "사무실", "찾아줘"])
            if is_contact_query and directory_found_results:
                logger.info("Multi-hop contact lookup successfully found results in Directory. Proceeding directly to synthesis.")
                break

        # 4. Response Streaming Strategy
        # Case 1: Pure Academic Regulation / Graduation query -> INUChat Direct Pass-through
        other_campus_domains = {
            "BUS", "CAFETERIA", "TIMETABLE", "WEATHER", "LIBRARY", "DIRECTORY", "CAMPUS_WATCH",
            "REMINDER", "DAILY_BRIEF", "KEYWORD", "SETTINGS"
        }
        has_other_campus_tools = bool(executed_categories.intersection(other_campus_domains))
        has_inuchat = "INU_AI_KNOWLEDGE" in executed_categories or inuchat_rag_data is not None

        if has_inuchat and not has_other_campus_tools:
            logger.info("Executing INUChat Direct Pass-through (Pure Academic/Regulation query)")
            yield AgentStreamEvent(
                event_type="THINKING",
                thinking="인천대학교 공식 학칙 및 졸업 규정 지식베이스의 정확한 내용을 전달합니다.",
            )

            # Build enriched question with anonymized academic context (Strict Zero-PII)
            clean_message = sanitize_text(request.message)
            inu_question = clean_message
            if academic_data_dict:
                ac_parts = []
                if academic_data_dict.get("entryYear"):
                    ac_parts.append(f"입학연도={academic_data_dict['entryYear']}학번")
                if academic_data_dict.get("departmentName"):
                    ac_parts.append(f"학과={academic_data_dict['departmentName']}")
                if academic_data_dict.get("enrollmentStatus"):
                    ac_parts.append(f"학적상태={academic_data_dict['enrollmentStatus']}")
                if academic_data_dict.get("acquiredCredits"):
                    ac_parts.append(f"취득학점={academic_data_dict['acquiredCredits']}학점")
                if academic_data_dict.get("gradeAverage"):
                    ac_parts.append(f"평점평균={academic_data_dict['gradeAverage']}")

                if ac_parts:
                    inu_question = f"{clean_message}\n\n[비식별 학적 참고정보: {', '.join(ac_parts)}]"

            # Find inuchat tool
            inuchat_tool = None
            for t in tool_registry.list_tools():
                if t.category.upper() == "INU_AI_KNOWLEDGE":
                    inuchat_tool = t
                    break

            tokens_emitted = 0
            if inuchat_tool and hasattr(inuchat_tool, "stream_execute"):
                async for token, is_done, full_acc, citations in inuchat_tool.stream_execute({"question": inu_question}):
                    if token:
                        tokens_emitted += 1
                        yield AgentStreamEvent(event_type="TOKEN", content=token)
                    if is_done and citations and "INU_AI_KNOWLEDGE" not in emitted_cards:
                        card = CardSynthesizer.synthesize_for_domain(
                            domain="INU_AI_KNOWLEDGE",
                            data={"citations": citations, "rag_answer": full_acc},
                            query=request.message,
                        )
                        if card:
                            yield AgentStreamEvent(event_type="CARD", card=card)
                            emitted_cards.add("INU_AI_KNOWLEDGE")

                if tokens_emitted == 0:
                    yield AgentStreamEvent(
                        event_type="TOKEN",
                        content="인천대학교 공식 학사 규정 지식베이스 조회를 완료하였습니다.",
                    )

                yield AgentStreamEvent(event_type="DONE")
                return

        # Case 2: On-Demand P2P Action Dispatched -> Complete first stream and await Client Callback (Native App only)
        # Note: TIMETABLE is excluded from pausing the stream so that fallback card + alternative guidance text is immediately synthesized!
        action_wait_cats = [c for c in emitted_actions if c != "TIMETABLE"]
        if is_app and action_wait_cats:
            first_action_cat = action_wait_cats[0]
            logger.info(
                f"[WAITING_FOR_ACTION_CALLBACK] query='{request.message[:40]}' "
                f"actions={action_wait_cats}. Pausing stream for client callback."
            )
            yield AgentStreamEvent(
                event_type="STATUS",
                status_id="p2p_action_waiting",
                status_title="단말기 보안 영역에서 학교 공식 시스템 데이터를 안전하게 확인 중입니다...",
                status_category=first_action_cat,
                status_state="running",
            )
            # Zero-Silence Guarantee: Always stream informative guidance tokens so the chat UI never stays blank while awaiting callback
            yield AgentStreamEvent(
                event_type="TOKEN",
                content="🔒 단말기 보안 영역(P2P)에서 학교 공식 시스템과 안전하게 통신하여 최신 학적 데이터를 확인하고 있습니다. 잠시만 기다려 주세요...",
            )
            yield AgentStreamEvent(event_type="DONE")
            return

        # Case 3: Composite Query or General Campus Tools -> Synthesis LLM
        additional_guideline = ""
        if has_inuchat:
            additional_guideline = (
                "\n\n[INUChat 학사 규정 답변 우선 및 온전한 보존 지침]:\n"
                "학사 규정, 졸업 요건, 학칙에 관한 내용은 [인천대학교 공식 학칙/규정 지식베이스]의 내용을 주 축(Core Grounding)으로 삼아 "
                "절대로 임의 축소하거나 왜곡하지 말고 최대한 원문 내용을 사용자에게 손상 없이 전달하세요. "
                "함께 조회된 다른 도구(학식, 버스 등)의 추가적인 팩트는 자연스럽게 곁들여서 하나의 친절하고 완성된 답변으로 작성하세요."
            )

        final_system_prompt = get_system_prompt_for_client(request.client, tool_summary=tool_summary_text) + additional_guideline

        synthesis_messages = [
            {"role": "system", "content": final_system_prompt}
        ]

        if request.history:
            for h in request.history[-6:]:
                role = getattr(h, "role", None) or (h.get("role") if isinstance(h, dict) else "user")
                content = getattr(h, "content", None) or (h.get("content") if isinstance(h, dict) else "")
                if content:
                    clean_role = "assistant" if role == "assistant" else "user"
                    synthesis_messages.append({"role": clean_role, "content": content})

        synthesis_messages.append({"role": "user", "content": request.message})

        logger.info(
            f"[GROUNDING_STATUS] query='{request.message[:40]}' "
            f"executed_tools={list(executed_tool_names)} "
            f"categories={list(executed_categories)} "
            f"has_grounding={bool(tool_summary_text.strip())} "
            f"summary_len={len(tool_summary_text)}"
        )

        try:
            logger.info(
                f"Starting synthesis stream for '{request.message[:30]}...' (Categories: {executed_categories})"
            )

            tokens_emitted = 0
            async for token in llm_client.stream_chat(messages=synthesis_messages):
                tokens_emitted += 1
                yield AgentStreamEvent(event_type="TOKEN", content=token)

            if tokens_emitted == 0:
                fallback_msg = "조회된 정보를 확인하였으나 세부 안내 문장을 생성하지 못했습니다. 질문을 다시 한번 입력해 주세요."
                yield AgentStreamEvent(event_type="TOKEN", content=fallback_msg)

            # Debug metadata event for troubleshooting and diagnostic trace
            if client_ctx.get("debug") or client_ctx.get("show_debug"):
                from app.orchestrator.prompts import get_current_semester
                yield AgentStreamEvent(
                    event_type="DEBUG",
                    debug={
                        "query": request.message,
                        "client": request.client,
                        "semester": get_current_semester(datetime.now()),
                        "candidate_tools": [t.name for t in candidate_tools],
                        "executed_tools": list(executed_tool_names),
                        "executed_categories": list(executed_categories),
                        "dispatched_actions": list(emitted_actions),
                        "emitted_cards": list(emitted_cards),
                        "grounding_data_length": len(tool_summary_text),
                        "has_grounded_data": bool(tool_summary_text.strip()),
                    }
                )

            # Optional inline debug footer in response text if requested
            if client_ctx.get("show_debug_footer"):
                debug_footer = (
                    f"\n\n---\n`[DEBUG TRACE]`\n"
                    f"- **도구 후보군**: {', '.join(t.name for t in candidate_tools)}\n"
                    f"- **실행된 도구**: {', '.join(executed_tool_names) or '없음'}\n"
                    f"- **조회 데이터 수신**: {'성공 (데이터 있음)' if tool_summary_text.strip() else '없음 (0건 또는 미조회)'}\n"
                )
                yield AgentStreamEvent(event_type="TOKEN", content=debug_footer)

            yield AgentStreamEvent(event_type="DONE")
        except Exception as e:
            logger.error(f"[AGENT_ERROR] Error in orchestrator stream: {e}", exc_info=True)
            yield AgentStreamEvent(event_type="ERROR", error=str(e))

    async def _execute_tool(
        self,
        tool: BaseTool,
        tool_args: Dict[str, Any],
        request: ChatRequest,
        exec_context: Dict[str, Any],
        emitted_cards: set,
        emitted_actions: set,
        academic_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Helper generator to execute a single tool, emitting status events and returning (event, summary, academic_data, rag_data).
        """
        tool_display_name = resolve_tool_display_name(tool.category, tool.name)
        if tool.category in ["LMS", "PORTAL"]:
            tool_id = f"tool_{tool.category.lower()}"
        else:
            tool_id = f"tool_{tool.category.lower()}_{abs(hash(tool.name)) % 10000}"

        summary_out = ""
        ac_data_out = {}
        rag_data_out = None

        logger.info(f"[TOOL_EXEC_START] tool={tool.name} category={tool.category} args={tool_args}")

        # Case A: Client Action (LMS, Portal ERP)
        if tool.category in ["LMS", "PORTAL"]:
            yield (
                AgentStreamEvent(
                    event_type="STATUS",
                    status_id=tool_id,
                    status_title=f"{tool_display_name} 조회 중...",
                    status_category=tool.category,
                    status_state="running",
                ),
                "",
                None,
                None,
            )

            client_ctx = request.client_context or {}
            is_app = bool(client_ctx.get("isApp", False))
            portal_meta = client_ctx.get("portal") if isinstance(client_ctx.get("portal"), dict) else {}
            is_portal_linked = portal_meta.get("linked") is True

            domain_data = None
            has_dispatched_action = False
            is_academic_tool = "academic" in tool.name.lower() or "grade" in tool.name.lower()
            is_timetable_tool = "timetable" in tool.name.lower() or "sukang" in tool.name.lower() or "tlsn" in tool.name.lower()

            if tool.category == "PORTAL" and is_academic_tool:
                domain_data = (
                    client_ctx.get("academic")
                    or client_ctx.get("academicDisplay")
                    or client_ctx.get("studentInfo")
                    or client_ctx.get("student")
                )
                if not domain_data and ("departmentName" in client_ctx or "advisorProfessorName" in client_ctx):
                    domain_data = client_ctx
            elif is_timetable_tool:
                domain_data = (
                    client_ctx.get("studentTimetable")
                    or client_ctx.get("timetable")
                    or client_ctx.get("courses")
                    or client_ctx.get("tlsnTimetable")
                )
            elif tool.category == "LMS":
                domain_data = client_ctx.get("lms")

            if domain_data and isinstance(domain_data, (dict, list)):
                if is_timetable_tool:
                    ac_data_out = domain_data if isinstance(domain_data, dict) else {"items": domain_data}
                    classes = domain_data if isinstance(domain_data, list) else (
                        domain_data.get("items") or domain_data.get("courses") or domain_data.get("todayClasses") or []
                    )
                    tt_lines = ["\n[학교 포털 수강신청 시간표 연동 데이터]:"]
                    if classes:
                        tt_lines.append(f"- 수강신청 완료된 교과목 (총 {len(classes)}과목):")
                        for c in classes[:10]:
                            if isinstance(c, dict):
                                c_name = c.get("title") or c.get("courseName") or c.get("subject") or c.get("name") or "과목"
                                c_room = c.get("room") or c.get("classroom") or c.get("location") or ""
                                c_time = c.get("time") or c.get("classTime") or ""
                                tt_lines.append(f"  • {c_name} | {c_time} | {c_room}".rstrip(" |"))
                    else:
                        tt_lines.append("- 현재 학기에 등록된 수강신청 내역이 없습니다.")
                    summary_out = "\n".join(tt_lines) + "\n"

                    if "TIMETABLE" not in emitted_cards:
                        card = CardSynthesizer.synthesize_for_domain(
                            domain="TIMETABLE",
                            tool_name=tool.name,
                            data=domain_data,
                            query=request.message,
                        )
                        if card:
                            yield (AgentStreamEvent(event_type="CARD", card=card), "", None, None)
                            emitted_cards.add("TIMETABLE")
                elif tool.category == "PORTAL" and isinstance(domain_data, dict):
                    display_data = client_ctx.get("academicDisplay")
                    merged_portal = {**domain_data, **(display_data if isinstance(display_data, dict) else {})}

                    # 100% Zero-PII: Sanitize record (no names, no full student IDs, no personal phone numbers)
                    clean_academic = sanitize_academic_record(merged_portal)
                    ac_data_out = clean_academic
                    summary_out = build_anonymized_academic_summary(clean_academic)

                    if tool.category not in emitted_cards:
                        card_source = (client_ctx.get("academicDisplay") or domain_data)
                        card = CardSynthesizer.synthesize_for_domain(
                            domain=tool.category,
                            tool_name=tool.name,
                            data=card_source,
                            query=request.message,
                        )
                        if card:
                            yield (AgentStreamEvent(event_type="CARD", card=card), "", None, None)
                            emitted_cards.add(tool.category)
                elif tool.category == "LMS" and isinstance(domain_data, (dict, list)):
                    events = []
                    courses = []
                    if isinstance(domain_data, dict):
                        events = domain_data.get("events") or domain_data.get("assignments") or []
                        courses = domain_data.get("courses") or []
                    elif isinstance(domain_data, list):
                        events = domain_data

                    lms_lines = ["\n[이러닝(LMS) 실제 학생 연동 데이터]:"]
                    if events:
                        lms_lines.append(f"- 다가오는 과제/시험/일정 (총 {len(events)}건):")
                        for ev in events[:8]:
                            if isinstance(ev, dict):
                                ev_name = ev.get("name") or ev.get("title") or "과제"
                                c_info = ev.get("course")
                                c_name = c_info.get("fullname") if isinstance(c_info, dict) else str(c_info or "")
                                due_str = ev.get("formattedtime") or "마감일시 미정"
                                c_prefix = f"[{c_name}] " if c_name else ""
                                lms_lines.append(f"  • {c_prefix}{ev_name} (마감: {due_str})")
                    else:
                        lms_lines.append("- 다가오는 마감 예정 과제 및 일정: 없음 (모두 완료 또는 등록된 과제 없음)")

                    if courses:
                        lms_lines.append(f"- 현재 수강 중인 강좌 (총 {len(courses)}과목):")
                        for c in courses[:10]:
                            if isinstance(c, dict):
                                c_name = c.get("fullname") or c.get("name") or ""
                                if c_name:
                                    lms_lines.append(f"  • {c_name}")

                    summary_out = "\n".join(lms_lines) + "\n"

                    if tool.category not in emitted_cards:
                        card = CardSynthesizer.synthesize_for_domain(
                            domain=tool.category,
                            tool_name=tool.name,
                            data=domain_data,
                            query=request.message,
                        )
                        if card:
                            yield (AgentStreamEvent(event_type="CARD", card=card), "", None, None)
                            emitted_cards.add(tool.category)
            else:
                has_dispatched_action = False
                action_category = "TIMETABLE" if is_timetable_tool else tool.category

                # Case A-1: Mobile App Environment (ReactNativeWebView) -> Dispatch On-Demand P2P Scraping Action (Academic only)
                if is_app and not is_timetable_tool:
                    if action_category not in emitted_actions:
                        try:
                            action_instruction = await tool.execute({}, exec_context)
                            if hasattr(action_instruction, "action_id"):
                                if getattr(request, "session_id", None):
                                    action_instruction.session_id = request.session_id
                                logger.info(
                                    f"[ACTION_DISPATCHED] action_id={action_instruction.action_id} domain={action_instruction.auth_domain} "
                                    f"target={action_instruction.request.url if action_instruction.request else 'none'} "
                                    f"session_id={action_instruction.session_id}"
                                )
                                yield (AgentStreamEvent(event_type="ACTION_REQUIRED", action=action_instruction), "", None, None)
                                emitted_actions.add(action_category)
                                has_dispatched_action = True
                        except Exception as ex:
                            logger.warning(f"Failed to generate client action for {tool.name}: {ex}")

                    if has_dispatched_action:
                        if is_timetable_tool:
                            summary_out = (
                                f"\n[TIMETABLE_STATUS]: 학생의 모바일 단말기(앱)에 최신 학교 시스템 조회 지침(Action: {tool.name})을 성공적으로 전달했습니다. "
                                "현재 세션에 사전 동기화된 이번 학기 수강신청/시간표 데이터가 없습니다. "
                                "(⚠️ 최우선 핵심 지침: 임의의 과목명이나 강의실을 절대로 지어내지 마세요! "
                                "또한 단일 대화 턴이므로 '잠시만 기다려 주세요' 또는 '불러오는 중입니다'라는 말로만 답변을 끝내지 마십시오. "
                                "'현재 인팁에 동기화된 이번 학기 수강신청 시간표가 등록되어 있지 않습니다. 인팁 앱의 [시간표] 탭에서 시간표를 추가하거나 관리할 수 있습니다'라고 대안과 함께 친절하고 명확하게 안내하세요.)\n"
                            )
                            if "TIMETABLE" not in emitted_cards:
                                fallback_card = CardSynthesizer.synthesize_for_domain(
                                    domain="TIMETABLE",
                                    tool_name=tool.name,
                                    data=None,
                                    query=request.message,
                                )
                                if fallback_card:
                                    yield (AgentStreamEvent(event_type="CARD", card=fallback_card), "", None, None)
                                    emitted_cards.add("TIMETABLE")
                        else:
                            summary_out = (
                                f"\n[{tool.category}_STATUS]: 학생의 모바일 단말기(앱)에 최신 학교 시스템 조회 지침(Action: {tool.name})을 성공적으로 하달했습니다. "
                                "현재 단말기가 학교 종합정보시스템(ERP)과 직접 통신하여 최신 학적 내역을 확인하고 있습니다. "
                                "(⚠️ 최우선 핵심 지침: 단말기 조회가 완료되기 전까지 시스템 조회 데이터에 정보가 없으므로, 임의의 정보를 절대로 지어내거나 추측하지 마세요! "
                                "계정 연동 카드를 누르라고 안내하지 말고, '기기에서 학교 포털 종합정보시스템(ERP)에 접속하여 최신 학적 정보를 안전하게 불러오는 중입니다'라고 정직하게 안내하세요.)\n"
                            )
                    elif tool.category == "PORTAL" and is_portal_linked:
                        err_msg = portal_meta.get("academicErrorMessage") or "포털 또는 ERP 응답을 확인하지 못했습니다."
                        target_domain = "TIMETABLE" if is_timetable_tool else "PORTAL"
                        if target_domain not in emitted_cards:
                            if is_timetable_tool:
                                fetch_fail_card = CardSynthesizer.synthesize_for_domain(
                                    domain="TIMETABLE",
                                    tool_name=tool.name,
                                    data=None,
                                    query=request.message,
                                )
                            else:
                                fetch_fail_card = CardSynthesizer.synthesize_for_domain(
                                    domain="PORTAL",
                                    tool_name=tool.name,
                                    data={"status": "FETCH_FAILED", "message": err_msg},
                                    query=request.message,
                                )
                            if fetch_fail_card:
                                yield (AgentStreamEvent(event_type="CARD", card=fetch_fail_card), "", None, None)
                                emitted_cards.add(target_domain)
                                emitted_cards.add(tool.category)

                        summary_out = (
                            f"\n[PORTAL_STATUS]: 학생의 포털 계정은 정상 연동되어 있으나, 학교 ERP(종합정보시스템) 응답 지연으로 학적 정보를 일시적으로 불러오지 못했습니다. ({err_msg}) "
                            "학생에게 계정 연동은 잘 유지되어 있으니 잠시 후 같은 질문을 다시 보내달라고 친절하게 안내하세요. (⚠️ 중요 지침: 계정 연동 카드를 다시 누르라고 절대 안내하지 마세요!)\n"
                        )
                    else:
                        if tool.category not in emitted_cards:
                            auth_card = CardSynthesizer.synthesize_for_domain(
                                domain=tool.category,
                                tool_name=tool.name,
                                data="AUTH_REQUIRED",
                                query=request.message,
                            )
                            if auth_card:
                                yield (AgentStreamEvent(event_type="CARD", card=auth_card), "", None, None)
                                emitted_cards.add(tool.category)

                        if tool.category == "PORTAL":
                            summary_out = (
                                "\n[PORTAL_STATUS]: 현재 세션에는 연동된 학생의 실제 학적 데이터가 없습니다. "
                                "대화창에 표시된 [포털 계정 연동하기] 카드를 눌러 1회 연동을 완료하면 즉시 조회가 가능함을 학생에게 친절히 안내하세요. "
                                "(⚠️ 중요 지침: 앱의 '설정'이나 '마이페이지' 등 다른 메뉴로 이동하라고 안내하지 마세요! "
                                "오직 '화면에 표시된 [포털 계정 연동하기] 카드를 눌러 연동을 진행해 주세요'라고만 정확히 안내해야 합니다.)\n"
                            )
                        else:
                            summary_out = (
                                "\n[LMS_STATUS]: 현재 세션에는 연동된 학생의 실제 이러닝(LMS) 데이터가 없습니다. "
                                "대화창에 표시된 [포털 계정 연동하기] 카드를 눌러 1회 연동을 완료하면 이러닝 과제와 학적 조회가 즉시 가능함을 학생에게 친절히 안내하세요. "
                                "(⚠️ 중요 지침: 인천대학교 포털, 이러닝(LMS), 도서관은 모두 동일한 포털 계정(학번/비밀번호)을 사용하므로 1회 등록 시 모두 함께 연동됩니다. "
                                "앱의 '설정'이나 '마이페이지' 등 다른 메뉴를 안내하지 말고, 오직 '화면에 표시된 [포털 계정 연동하기] 카드를 눌러 연동을 진행해 주세요'라고만 정확히 안내해야 합니다.)\n"
                            )

                # Case A-2: Web Environment (is_app is False) -> No Native Bridge, Seamless Card & Synthesis
                else:
                    if is_timetable_tool:
                        summary_out = (
                            f"\n[TIMETABLE_STATUS]: 현재 세션에는 동기화된 이번 학기 수강신청/시간표 데이터가 없습니다. (등록된 수업 없음) "
                            "(⚠️ 최우선 핵심 지침: 임의의 과목명이나 강의실을 절대로 지어내지 마세요! "
                            "사용자에게 '현재 인팁에 동기화된 이번 학기 수강신청 시간표가 등록되어 있지 않습니다. 인팁 앱의 [시간표] 탭에서 시간표를 추가하거나 관리할 수 있으며, 또는 포털 종합정보시스템의 수강신청 내역에서 확인하실 수 있습니다'라고 대안과 함께 친절하고 명확하게 안내하세요.)\n"
                        )
                        if "TIMETABLE" not in emitted_cards:
                            fallback_card = CardSynthesizer.synthesize_for_domain(
                                domain="TIMETABLE",
                                tool_name=tool.name,
                                data=None,
                                query=request.message,
                            )
                            if fallback_card:
                                yield (AgentStreamEvent(event_type="CARD", card=fallback_card), "", None, None)
                                emitted_cards.add("TIMETABLE")
                    elif tool.category == "PORTAL" and is_portal_linked:
                        err_msg = portal_meta.get("academicErrorMessage") or "현재 세션에 동기화된 학적 정보가 없습니다."
                        target_domain = "PORTAL"
                        if target_domain not in emitted_cards:
                            fetch_fail_card = CardSynthesizer.synthesize_for_domain(
                                domain="PORTAL",
                                tool_name=tool.name,
                                data={"status": "FETCH_FAILED", "message": err_msg},
                                query=request.message,
                            )
                            if fetch_fail_card:
                                yield (AgentStreamEvent(event_type="CARD", card=fetch_fail_card), "", None, None)
                                emitted_cards.add(target_domain)
                                emitted_cards.add(tool.category)

                        summary_out = (
                            f"\n[PORTAL_STATUS]: 학생의 포털 계정은 정상 연동되어 있으나, 현재 세션에 동기화된 학적 정보를 불러오지 못했습니다. ({err_msg}) "
                            "학생에게 계정 연동은 잘 유지되어 있으니 잠시 후 같은 질문을 다시 시도해 주시거나 인팁 모바일 앱에서 확인해 달라고 친절하게 안내하세요. (⚠️ 중요 지침: 계정 연동 카드를 다시 누르라고 절대 안내하지 마세요!)\n"
                        )
                    else:
                        if tool.category not in emitted_cards:
                            auth_card = CardSynthesizer.synthesize_for_domain(
                                domain=tool.category,
                                tool_name=tool.name,
                                data="AUTH_REQUIRED",
                                query=request.message,
                            )
                            if auth_card:
                                yield (AgentStreamEvent(event_type="CARD", card=auth_card), "", None, None)
                                emitted_cards.add(tool.category)

                        if tool.category == "PORTAL":
                            summary_out = (
                                "\n[PORTAL_STATUS]: 현재 세션에는 연동된 학생의 실제 학적 데이터가 없습니다. "
                                "대화창에 표시된 [포털 계정 연동하기] 카드를 눌러 1회 연동을 완료하면 즉시 조회가 가능함을 학생에게 친절히 안내하세요. "
                                "(⚠️ 중요 지침: 앱의 '설정'이나 '마이페이지' 등 다른 메뉴로 이동하라고 안내하지 마세요! "
                                "오직 '화면에 표시된 [포털 계정 연동하기] 카드를 눌러 연동을 진행해 주세요'라고만 정확히 안내해야 합니다.)\n"
                            )
                        else:
                            summary_out = (
                                "\n[LMS_STATUS]: 현재 세션에는 연동된 학생의 실제 이러닝(LMS) 데이터가 없습니다. "
                                "대화창에 표시된 [포털 계정 연동하기] 카드를 눌러 1회 연동을 완료하면 이러닝 과제와 학적 조회가 즉시 가능함을 학생에게 친절히 안내하세요. "
                                "(⚠️ 중요 지침: 인천대학교 포털, 이러닝(LMS), 도서관은 모두 동일한 포털 계정(학번/비밀번호)을 사용하므로 1회 등록 시 모두 함께 연동됩니다. "
                                "앱의 '설정'이나 '마이페이지' 등 다른 메뉴를 안내하지 말고, 오직 '화면에 표시된 [포털 계정 연동하기] 카드를 눌러 연동을 진행해 주세요'라고만 정확히 안내해야 합니다.)\n"
                            )

            if has_dispatched_action:
                status_state = "running"
                status_title = f"{tool_display_name} 조회 중..."
            elif domain_data or (not is_app and is_timetable_tool):
                status_state = "completed"
                status_title = f"{tool_display_name} 확인 완료"
            elif not is_app and is_portal_linked:
                status_state = "completed"
                status_title = f"{tool_display_name} 확인 완료"
            elif not is_app and not is_portal_linked:
                status_state = "completed"
                status_title = f"{tool_display_name} 연동 안내"
            else:
                status_state = "failed"
                status_title = f"{tool_display_name} 조회 실패"

            yield (
                AgentStreamEvent(
                    event_type="STATUS",
                    status_id=tool_id,
                    status_title=status_title,
                    status_category=tool.category,
                    status_state=status_state,
                ),
                summary_out,
                ac_data_out,
                None,
            )

        # Case B: Server OpenAPI / Direct Tools / MCP Tools
        elif tool.category != "INU_AI_KNOWLEDGE":
            yield (
                AgentStreamEvent(
                    event_type="STATUS",
                    status_id=tool_id,
                    status_title=f"{tool_display_name} 조회 중...",
                    status_category=tool.category,
                    status_state="running",
                ),
                "",
                None,
                None,
            )

            tool_data = None
            final_args = dict(tool_args or {})

            # Middleware: Entity Resolution for DIRECTORY
            if tool.category == "DIRECTORY":
                curr_q = str(final_args.get("query") or "").strip()
                if not curr_q or AgentRouter._is_generic_contact_term(curr_q):
                    resolved_name = None
                    if academic_context and isinstance(academic_context, dict):
                        resolved_name = academic_context.get("advisor") or academic_context.get("advisorProfessorName") or academic_context.get("departmentName")
                    if not resolved_name and request.client_context:
                        acad_disp = request.client_context.get("academicDisplay")
                        if isinstance(acad_disp, dict):
                            resolved_name = acad_disp.get("advisorProfessorName") or acad_disp.get("profNm") or acad_disp.get("departmentName")
                    if not resolved_name and request.history:
                        name_pattern = re.compile(r"([가-힣]{2,4})\s*(?:교수님|교수|선생님)")
                        for h in reversed(request.history[-6:]):
                            content = getattr(h, "content", None) or (h.get("content") if isinstance(h, dict) else "")
                            if content:
                                m = name_pattern.findall(content)
                                if m:
                                    resolved_name = m[-1]
                                    break
                    if resolved_name:
                        final_args["query"] = resolved_name
                if final_args.get("query"):
                    final_args["query"] = AgentRouter._clean_entity_query(str(final_args["query"]))

            # Middleware: Bus stop & route matching
            bus_meta = None
            if tool.category == "BUS":
                from app.orchestrator.router import DynamicBusMatcher
                user_stop = str(final_args.get("bstopId") or final_args.get("stop_name") or final_args.get("stopName") or "").strip()
                bstop_id, stop_name, tab_name, valid_routes = await DynamicBusMatcher.resolve_stop_and_routes(
                    request.message, user_stop if user_stop and not user_stop.isdigit() else None
                )
                if bstop_id:
                    final_args["bstopId"] = bstop_id
                bus_meta = {
                    "validRoutes": valid_routes,
                    "stopName": stop_name,
                    "tabName": tab_name,
                }

            # Schema-driven dynamic parameter coercion (Generic MCP standard)
            schema_params = getattr(tool, "input_schema", None) or {}
            final_args = SchemaCoercer.coerce(schema_params, final_args)

            status_state = "completed"
            status_title = f"{tool_display_name} 확인 완료"

            try:
                res = await tool.execute(final_args, exec_context)
                if isinstance(res, dict) and "error" in res:
                    err_msg = res.get("error", "알 수 없는 통신 오류")
                    is_val_err = res.get("is_validation_error") or ("올바르지 않습니다" in err_msg or "파라미터" in err_msg)
                    if is_val_err:
                        status_state = "failed"
                        status_title = f"{tool_display_name} 입력값 오류"
                        summary_out = (
                            f"\n[도구 입력 형식 오류 (Tool Parameter Error)]:\n"
                            f"서버 반환 메시지: {err_msg}\n"
                            f"💡 [자가 수정(Self-Correction) 지침]: 도구 파라미터가 유효하지 않아 호출에 실패했습니다. "
                            f"위 오류 메시지와 도구 스키마(inputSchema)의 허용 값/형식을 확인하고, 올바른 값으로 즉시 수정하여 도구를 다시 호출하세요.\n"
                        )
                    else:
                        status_state = "failed"
                        status_title = f"{tool_display_name} 조회 실패"
                        summary_out = (
                            f"\n[시스템 오류 고지]: {tool_display_name} 조회 중 일시적인 교내 서버 응답 지연 또는 오류({err_msg})가 발생하여 실시간 정보를 가져오지 못했습니다.\n"
                            f"⚠️ 핵심 응답 지침: 절대로 임의의 가상 정보(식단 메뉴, 버스 도착 시간, 전화번호, 시간표 등)를 지어내지 말고, "
                            f"'현재 교내 시스템 일시 오류로 실시간 정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요'라고 사실대로 사용자에게 안내하세요.\n"
                        )
                elif res is not None:
                    if tool.category == "BUS":
                        mcp_summary = (res.get("summary") or "").strip() if isinstance(res, dict) else ""
                        bus_raw = res.get("rawData") if (isinstance(res, dict) and isinstance(res.get("rawData"), dict)) else {}

                        stop_name = (bus_meta.get("stopName") if bus_meta else None) or bus_raw.get("stopName") or "정류소"
                        tab_name = (bus_meta.get("tabName") if bus_meta else None) or bus_raw.get("tabName") or "정류장"
                        valid_routes = (bus_meta.get("validRoutes") if bus_meta else None) or bus_raw.get("validRoutes") or []

                        raw_arrivals = []
                        if isinstance(res, list):
                            raw_arrivals = res
                        elif isinstance(res, dict):
                            raw_arrivals = bus_raw.get("arrivals") or res.get("arrivals") or res.get("items") or []

                        filtered_arrivals = [
                            item for item in raw_arrivals
                            if isinstance(item, dict) and (not valid_routes or item.get("routeNo") in valid_routes)
                        ] if valid_routes else raw_arrivals

                        tool_data = {
                            "arrivals": filtered_arrivals,
                            "validRoutes": valid_routes,
                            "stopName": stop_name,
                            "tabName": tab_name,
                        }
                        if filtered_arrivals:
                            status_state = "completed"
                            status_title = f"{tool_display_name} 확인 완료"
                            summary_out = (
                                f"\n[BUS 인천시 실시간 시내버스 도착 정보 ({stop_name})]:\n"
                                f"- 모니터링 노선: {', '.join(valid_routes)}\n"
                                f"- 실시간 도착 정보:\n{json.dumps(filtered_arrivals, ensure_ascii=False)}\n"
                            )
                        elif mcp_summary:
                            if "없습니다" in mcp_summary or "운행 대기" in mcp_summary:
                                status_state = "empty"
                                status_title = f"{tool_display_name} 결과 없음"
                            else:
                                status_state = "completed"
                                status_title = f"{tool_display_name} 확인 완료"
                            summary_out = f"\n[BUS 인천시 실시간 시내버스 도착 정보 ({stop_name})]:\n{mcp_summary}\n"
                        else:
                            status_state = "empty"
                            status_title = f"{tool_display_name} 결과 없음"
                            summary_out = (
                                f"\n[BUS 인천시 실시간 시내버스 도착 정보 ({stop_name})]:\n"
                                f"- 모니터링 노선: {', '.join(valid_routes)}\n"
                                f"- 현재 해당 정류장에 운행 대기 중이거나 도착 예정인 인팁 서비스 버스가 없습니다.\n"
                                f"💡 지침: 학생에게 현재 운행 중이거나 도착 예정인 버스가 없음을 사실대로 안내하세요. 절대로 임의의 도착 시간을 지어내지 마세요.\n"
                            )
                    elif tool.category == "CAFETERIA":
                        tool_data = res
                        caf_name = final_args.get("cafeteria", "학생식당")
                        raw_menus = []
                        mcp_summary = (res.get("summary") or "").strip() if isinstance(res, dict) else ""
                        if isinstance(res, list):
                            raw_menus = res
                        elif isinstance(res, dict):
                            raw_menus = (
                                res.get("rawData")
                                or (res.get("uiComponent", {}).get("data") if isinstance(res.get("uiComponent"), dict) else [])
                                or res.get("data")
                                or res.get("items")
                                or res.get("menus")
                                or res.get("cafeterias")
                                or []
                            )
                            if isinstance(raw_menus, dict):
                                raw_menus = [raw_menus]

                        lines = [f"\n[CAFETERIA {caf_name} 식단 메뉴 조회 결과]:"]
                        has_valid_menu = False

                        # Case 1: List of strings [조식, 중식, 석식] (Spring Boot INU Portal standard format)
                        if isinstance(raw_menus, list) and raw_menus and isinstance(raw_menus[0], str):
                            meal_names = ["조식(아침)", "중식(점심)", "석식(저녁)"]
                            for idx, menu_text in enumerate(raw_menus):
                                label = meal_names[idx] if idx < len(meal_names) else f"식단 {idx+1}"
                                clean_text = str(menu_text).strip()
                                if clean_text and clean_text != "-" and clean_text != "운영없음" and clean_text != "식단 없음":
                                    has_valid_menu = True
                                    lines.append(f"### [{label}]\n{clean_text}\n")
                                else:
                                    lines.append(f"- {label}: 운영 없음 또는 식단 미등록")

                        # Case 2: List of dicts
                        elif isinstance(raw_menus, list):
                            for m in raw_menus:
                                if isinstance(m, dict):
                                    if "breakfast" in m or "lunch" in m or "dinner" in m:
                                        b = m.get("breakfast")
                                        l = m.get("lunch")
                                        d = m.get("dinner")
                                        for meal_label, meal_val in [("조식(아침)", b), ("중식(점심)", l), ("석식(저녁)", d)]:
                                            clean_val = str(meal_val).strip() if meal_val else ""
                                            if clean_val and clean_val not in ["-", "운영없음", "식단 없음"]:
                                                has_valid_menu = True
                                                lines.append(f"### [{meal_label}]\n{clean_val}\n")
                                            elif clean_val:
                                                lines.append(f"- {meal_label}: {clean_val}")
                                    else:
                                        corner = m.get("name") or m.get("cornerName") or m.get("corner") or "코너"
                                        menu_str = m.get("menu") or m.get("menuName") or ""
                                        meal_label = m.get("mealLabel") or m.get("mealType") or ""
                                        if menu_str and menu_str != "-":
                                            has_valid_menu = True
                                            prefix = f"[{meal_label}] " if meal_label else ""
                                            lines.append(f"- {prefix}{corner}: {menu_str}")

                        if not has_valid_menu and mcp_summary:
                            if "올바르지 않습니다" in mcp_summary or "오류가 발생했습니다" in mcp_summary or "에러" in mcp_summary:
                                status_state = "failed"
                                status_title = f"{tool_display_name} 조회 실패"
                                summary_out = (
                                    f"\n[시스템 오류 고지]: {tool_display_name} 조회 중 일시적인 서버 파라미터 오류가 발생했습니다.\n"
                                    f"⚠️ 핵심 지침: 학생에게 현재 식단 정보를 불러오는 중 일시적 오류가 발생했음을 사실대로 안내하세요.\n"
                                )
                            elif any(k in mcp_summary for k in ["쉽니다", "운영하지 않습니다", "정보가 없습니다", "등록된 식단이 없습니다", "식단 없음", "운영 없음"]):
                                status_state = "empty"
                                status_title = f"{tool_display_name} 결과 없음"
                                summary_out = f"\n[CAFETERIA {caf_name} 식단 안내]:\n{mcp_summary}\n"
                            else:
                                has_valid_menu = True
                                lines.append(f"\n{mcp_summary}")

                        if has_valid_menu:
                            status_state = "completed"
                            status_title = f"{tool_display_name} 확인 완료"
                            lines.append("\n⚠️ 핵심 지침: 반드시 위 실제 조회된 메뉴 목록에 근거하여 안내하세요. 위 목록에 없는 가상의 메뉴를 절대로 추가하거나 지어내지 마세요.")
                            summary_out = "\n".join(lines) + "\n"
                        elif status_state != "failed":
                            status_state = "empty"
                            status_title = f"{tool_display_name} 결과 없음"
                            summary_out = (
                                f"\n[CAFETERIA {caf_name} 식단 메뉴 조회 결과]:\n"
                                f"- 현재 {caf_name}에 등록된 식단 메뉴가 없습니다 (식당 운영 시간 외 또는 식단 미등록 상태).\n"
                                f"⚠️ 핵심 지침: 절대로 가상의 식단 메뉴를 지어내어 답변하지 마세요. 학생에게 '현재 {caf_name}의 식단 정보가 등록되어 있지 않거나 식당 미운영 상태입니다'라고 사실대로 안내하세요.\n"
                            )
                    elif tool.category == "LIBRARY" and isinstance(res, dict):
                        tool_data = res
                        lib_raw = res.get("rawData") if isinstance(res.get("rawData"), dict) else (res.get("uiComponent", {}).get("data") if isinstance(res.get("uiComponent"), dict) else res)
                        mcp_summary = (res.get("summary") or "").strip()
                        mode = lib_raw.get("mode") or res.get("mode")
                        if mode == "STUDY_ROOMS":
                            study_rooms = lib_raw.get("rooms") or lib_raw.get("data", {}).get("rooms", []) or res.get("rooms", [])
                            rooms_info = "\n".join([
                                f"- {r.get('name')}: 위치 {r.get('location')}, 수용정원 {r.get('quota')}, 구비시설 ({', '.join(r.get('tags', []))})"
                                for r in study_rooms if isinstance(r, dict)
                            ])
                            notice = lib_raw.get("notice") or res.get("notice", "")
                            summary_out = (
                                f"\n[학산도서관 스터디룸 목록 및 예약 안내]:\n"
                                f"{rooms_info}\n"
                                f"- 이용 규정: {notice}\n"
                                f"💡 지침: 학생에게 위 스터디룸 목록을 친절히 안내하고, 대화창의 [학산도서관 스터디룸] 카드에서 원하는 방을 터치하면 바로 예약 화면으로 연결된다고 설명하세요.\n"
                            )
                        elif mode in ["RESERVE_SEAT", "RESERVE_STUDY_ROOM"]:
                            summary_out = (
                                f"\n[학산도서관 대화형 신청 카드 발급 완료]:\n"
                                f"{lib_raw.get('instruction') or res.get('instruction', '')}\n"
                            )
                        else:
                            rooms = lib_raw.get("rooms") or res.get("rooms", [])
                            if rooms:
                                rooms_info = "\n".join([
                                    f"- {r.get('name', '')}: 잔여 {r.get('available_seats', r.get('seats', {}).get('available', 0))}석 / 전체 {r.get('total_seats', r.get('seats', {}).get('total', 0))}석"
                                    for r in rooms if isinstance(r, dict)
                                ])
                                summary_out = (
                                    f"\n[학산도서관 열람실 실시간 잔여 좌석 현황 (공식 pyxis 시스템 실시간 관측 데이터)]:\n"
                                    f"{rooms_info}\n"
                                    f"⚠️ 지침: 반드시 위 실제 실시간 잔여 좌석 수치 그대로 학생에게 안내하세요 (임의의 숫자를 지어내지 마세요). "
                                    f"대화창 아래 제공된 실시간 열람실 카드에서 원하는 열람실을 터치하면 좌석 배정 화면으로 이동할 수 있음을 덧붙이세요.\n"
                                )
                            elif mcp_summary:
                                summary_out = f"\n[학산도서관 실시간 좌석 현황]:\n{mcp_summary}\n"
                            else:
                                status_state = "empty"
                                status_title = f"{tool_display_name} 결과 없음"
                                summary_out = "\n[학산도서관]: 현재 조회 가능한 좌석 정보가 없습니다.\n"
                    elif tool.category == "CAMPUS_WATCH" and isinstance(res, dict):
                        tool_data = res
                        summary_out = f"\n[학산도서관 실시간 빈자리 알림/스나이퍼 감시 결과]:\n{res.get('summary', '')}\n"
                    elif tool.category == "TIMETABLE":
                        tool_data = res if res is not None else {}
                        classes = []
                        if isinstance(res, list):
                            classes = res
                        elif isinstance(res, dict):
                            classes = (
                                res.get("rawData")
                                or (res.get("uiComponent", {}).get("data") if isinstance(res.get("uiComponent"), dict) else [])
                                or res.get("items")
                                or res.get("todayClasses")
                                or res.get("courses")
                                or []
                            )
                        if isinstance(classes, dict):
                            classes = [classes]

                        if not classes:
                            status_state = "empty"
                            status_title = f"{tool_display_name} 결과 없음"
                            summary_out = (
                                "\n[INTIP 인팁 시간표 조회 결과]:\n"
                                "- 오늘 등록된 수업/강의 일정이 없습니다 (또는 인팁 앱에 시간표가 등록되어 있지 않습니다).\n"
                                "💡 지침: 학생에게 오늘 예정된 수업이 없거나 시간표가 등록되지 않았음을 친절히 안내하고, '인팁 앱의 [시간표] 탭에서 이번 학기 시간표를 추가하거나 확인할 수 있어요'라고 안내하세요.\n"
                                "⚠️ 중요: 이 기능은 인팁(INTIP) 앱 자체의 시간표 기능이므로, 포털/LMS 계정 연동을 절대 요구하지 마세요.\n"
                            )
                        else:
                            status_state = "completed"
                            status_title = f"{tool_display_name} 확인 완료"
                            summary_out = (
                                f"\n[INTIP 인팁 오늘의 수업 시간표]:\n"
                                f"{json.dumps(classes, ensure_ascii=False)[:1000]}\n"
                                f"💡 지침: 학생에게 오늘 수업 시간과 강의실을 명확히 안내하고, 하단의 [나의 수업 시간표] 카드에서 전체 시간표를 확인할 수 있다고 덧붙이세요.\n"
                            )
                    elif tool.category == "REMINDER":
                        tool_data = res
                        summary_out = f"\n[AI 맞춤 알림(리마인더) 처리 결과]:\n{json.dumps(res, ensure_ascii=False)}\n💡 지침: 알림 등록/수정/삭제/조회 결과를 사용자에게 친절하고 명확하게 안내하세요.\n"
                    elif tool.category == "DAILY_BRIEF":
                        tool_data = res
                        summary_out = f"\n[데일리 브리프 아침 일정 설정 결과]:\n{json.dumps(res, ensure_ascii=False)}\n💡 지침: 데일리 브리프 시간 및 활성화 설정 상태를 친절히 안내하세요.\n"
                    elif tool.category == "KEYWORD":
                        tool_data = res
                        summary_out = f"\n[공지사항 키워드 알림 구독 결과]:\n{json.dumps(res, ensure_ascii=False)}\n💡 지침: 공지 키워드 등록/조회/삭제 결과를 사용자에게 명확히 안내하세요.\n"
                    elif tool.category == "SETTINGS":
                        tool_data = res
                        summary_out = f"\n[내 맞춤 알림 및 브리프 종합 설정]:\n{json.dumps(res, ensure_ascii=False)}\n💡 지침: 등록된 키워드, 맞춤 알림, 데일리 브리프 설정을 일목요연하게 정리해 안내하세요.\n"
                    elif tool.category == "DIRECTORY":
                        tool_data = res
                        q_param = str(final_args.get("query") or "").strip()
                        raw_list = []
                        if isinstance(res, list):
                            raw_list = res
                        elif isinstance(res, dict):
                            raw_list = (
                                res.get("rawData")
                                or (res.get("uiComponent", {}).get("data") if isinstance(res.get("uiComponent"), dict) else [])
                                or res.get("contents")
                                or res.get("items")
                                or (res.get("data", {}).get("contents") if isinstance(res.get("data"), dict) else [])
                                or (res.get("data") if isinstance(res.get("data"), list) else [])
                                or []
                            )
                        if isinstance(raw_list, dict):
                            raw_list = [raw_list]

                        mcp_summary = (res.get("summary") or "").strip() if isinstance(res, dict) else ""
                        is_empty = False
                        if not raw_list and not mcp_summary:
                            is_empty = True
                        elif mcp_summary and ("찾지 못했습니다" in mcp_summary or "없습니다" in mcp_summary) and not raw_list:
                            is_empty = True

                        if is_empty:
                            status_state = "empty"
                            status_title = f"{q_param} 연락처 결과 없음" if q_param else f"{tool_display_name} 결과 없음"
                            summary_out = (
                                f"\n[교내 부서/학과/교수 연락처 검색 결과 (검색어: '{q_param}')]:\n"
                                f"- 검색어 '{q_param}' 관련 연락처가 교내 전화번호부 DB에 등록되어 있지 않습니다 (0건).\n"
                                f"💡 [자가 회복 및 교차 탐색 지침 (Self-Correction & Fallback)]:\n"
                                f"전화번호부 DB에 일치하는 결과가 없습니다. 절대로 첫 시도 실패만으로 조회를 포기하지 마십시오!\n"
                                f"1) 인천대학교 전 도메인 고도화 통합 검색 도구(api_unified_search 또는 unifiedSearch)를 즉시 호출하여 학교 공식 홈페이지, 부서 조직도, 공지사항 등에서 '{q_param}' 연락처 및 위치를 포괄적으로 검색하세요.\n"
                                f"2) 또는 검색어를 단과대/핵심 명칭(예: '정보기술대학' 또는 '정보기술')으로 단순화하여 재검색을 시도하세요.\n"
                            )
                        else:
                            status_state = "completed"
                            status_title = f"{q_param} 연락처 확인 완료" if q_param else f"{tool_display_name} 확인 완료"
                            if mcp_summary and "찾지 못했습니다" not in mcp_summary:
                                summary_out = (
                                    f"\n[교내 부서/학과/교수 연락처 검색 결과 (검색어: '{q_param}')]:\n"
                                    f"{mcp_summary}\n"
                                    f"💡 지침: 위 조회된 실제 연락처(전화번호, 이메일, 연구실/사무실 위치 등)를 학생에게 사실대로 정확히 안내하세요. 가상의 정보를 임의로 추가하거나 지어내지 마세요.\n"
                                )
                            else:
                                lines = [f"\n[교내 부서/학과/교수 연락처 검색 결과 (검색어: '{q_param}')]:"]
                                for e in raw_list[:6]:
                                    if isinstance(e, dict):
                                        dept_name = e.get("departmentName") or e.get("deptName") or ""
                                        name = e.get("name") or ""
                                        if dept_name and not name:
                                            colg = e.get("collegeName") or ""
                                            phone = e.get("officePhoneNumber") or e.get("phoneNumber") or ""
                                            loc = e.get("officeLocation") or e.get("location") or ""
                                            hp = e.get("homepageUrl") or ""
                                            label = f"{dept_name} ({colg})" if colg else dept_name
                                            lines.append(f"- [학과 사무실] {label}: 📞 {phone}" + (f" (위치: {loc})" if loc else "") + (f", 홈페이지: {hp}" if hp else ""))
                                        else:
                                            pos = e.get("position") or ""
                                            affil = e.get("detailAffiliation") or e.get("affiliation") or ""
                                            phone = e.get("phoneNumber") or ""
                                            email = e.get("email") or ""
                                            lines.append(f"- [교수/교직원] {name} ({pos}, {affil}): 📞 {phone}" + (f", ✉️ {email}" if email else ""))
                                lines.append("💡 지침: 위 조회된 실제 연락처(전화번호, 이메일, 연구실/사무실 위치 등)를 학생에게 사실대로 정확히 안내하세요.")
                                summary_out = "\n".join(lines) + "\n"
                    elif tool.category == "SCHEDULE":
                        tool_data = res
                        sched_list = []
                        if isinstance(res, list):
                            sched_list = res
                        elif isinstance(res, dict):
                            sched_list = (
                                res.get("rawData")
                                or (res.get("uiComponent", {}).get("data") if isinstance(res.get("uiComponent"), dict) else [])
                                or res.get("items")
                                or res.get("contents")
                                or res.get("schedules")
                                or (res.get("data") if isinstance(res.get("data"), list) else [])
                                or []
                            )
                        if isinstance(sched_list, dict):
                            sched_list = [sched_list]

                        mcp_summary = (res.get("summary") or "").strip() if isinstance(res, dict) else ""
                        lines = [f"\n[인천대학교 학사일정 조회 결과 ({tool_display_name})]:"]
                        is_empty = False
                        if not sched_list and not mcp_summary:
                            is_empty = True
                        elif mcp_summary and any(kw in mcp_summary for kw in ["없습니다", "찾지 못했습니다", "0건", "없음", "일정이 없습니다"]) and not sched_list:
                            is_empty = True

                        if not is_empty:
                            status_state = "completed"
                            status_title = f"{tool_display_name} 확인 완료"
                            if sched_list:
                                for s in sched_list[:8]:
                                    if isinstance(s, dict):
                                        title = s.get("title") or s.get("content") or "학사일정"
                                        start = s.get("start") or s.get("startDate") or ""
                                        end = s.get("end") or s.get("endDate") or ""
                                        date_str = f"{start} ~ {end}" if start and end and start != end else (start or end or "일정 미정")
                                        cal_link = f"/home/calendar?date={start}" if start else "/home/calendar"
                                        lines.append(f"- [{title}]({cal_link}) (기간: {date_str})")
                            elif mcp_summary:
                                lines.append(mcp_summary)
                            lines.append("\n💡 [학사일정 링크 작성 지침]: 학사일정에는 절대 공지사항 링크(/home/notice/{id})를 붙이지 마세요! 위 제공된 인팁 앱 캘린더 상대 경로인 `[일정명](/home/calendar?date=YYYY-MM-DD)`(시작일 기준) 또는 `[학사일정 전체보기](/home/calendar)`로만 마크다운 링크를 표(|---|---|)나 목록에 그대로 작성하세요.")
                        else:
                            status_state = "empty"
                            status_title = f"{tool_display_name} 결과 없음"
                            lines.append("- 조회된 학사일정이 없습니다.")
                            lines.append("💡 [학사일정 링크 작성 지침]: 학생에게 등록된 일정이 없음을 안내하고, 전체 학사일정을 확인할 수 있도록 `[학사일정 전체보기](/home/calendar)` 링크를 제공하세요.")
                        summary_out = "\n".join(lines) + "\n"
                    elif tool.category == "NOTICE":
                        tool_data = res
                        notices_list = []
                        if isinstance(res, list):
                            notices_list = res
                        elif isinstance(res, dict):
                            notices_list = (
                                res.get("rawData")
                                or (res.get("uiComponent", {}).get("data") if isinstance(res.get("uiComponent"), dict) else [])
                                or res.get("contents")
                                or res.get("items")
                                or res.get("notices")
                                or []
                            )
                        if isinstance(notices_list, dict):
                            notices_list = [notices_list]

                        mcp_summary = (res.get("summary") or "").strip() if isinstance(res, dict) else ""
                        lines = [f"\n[인천대학교 공지사항 조회 결과 ({tool_display_name})]:"]
                        is_empty_notice = False
                        if not notices_list and not mcp_summary:
                            is_empty_notice = True
                        elif mcp_summary and any(kw in mcp_summary for kw in ["없습니다", "찾지 못했습니다", "0건", "없음"]) and not notices_list:
                            is_empty_notice = True

                        if not is_empty_notice:
                            status_state = "completed"
                            status_title = f"{tool_display_name} 확인 완료"
                            if notices_list:
                                for n in notices_list[:5]:
                                    if isinstance(n, dict):
                                        t = n.get("title") or "공지사항"
                                        n_id = n.get("id")
                                        # INTIP 내부 상세 페이지 경로 (/home/notice/{id}) 연결로 외부 브라우저 이탈 방지
                                        u = f"/home/notice/{n_id}" if n_id else (n.get("url") or "")
                                        d = n.get("createDate") or n.get("date") or ""
                                        w = n.get("writer") or n.get("category") or ""
                                        link_str = f"[{t}]({u})" if u else t
                                        sub_str = f" (게시일: {d}, 작성: {w})" if d or w else ""
                                        lines.append(f"- {link_str}{sub_str}")
                            elif mcp_summary:
                                lines.append(mcp_summary)
                            lines.append("💡 [중요 지침]: 위 공지 제목과 내부 상세 링크(/home/notice/{공지ID})를 마크다운 링크 형식([공지제목](/home/notice/{id}))으로 답변에 그대로 포함하여 안내하세요. 외부 브라우저로 나가지 않고 INTIP 앱 내부 상세 페이지로 열립니다.")
                        else:
                            status_state = "empty"
                            status_title = f"{tool_display_name} 결과 없음"
                            lines.append("- 조회된 공지사항이 없습니다.")
                        summary_out = "\n".join(lines) + "\n"
                    elif tool.category == "SEARCH" or "unifiedsearch" in tool.name.lower():
                        tool_data = res
                        if isinstance(res, dict):
                            s_data = res.get("rawData") if isinstance(res.get("rawData"), dict) else res
                            if isinstance(s_data, dict) and "items" in s_data and isinstance(s_data["items"], list) and s_data["items"]:
                                first_item = s_data["items"][0]
                                if isinstance(first_item, dict) and any(k in first_item for k in ["notices", "schedules", "directory", "departmentNotices"]):
                                    s_data = first_item
                            total_cnt = s_data.get("totalCount", 0)
                            q_val = s_data.get("query") or s_data.get("q") or final_args.get("q") or final_args.get("query", "")
                            mcp_summary = (res.get("summary") or "").strip() if isinstance(res, dict) else ""
                            lines = [f"\n[인천대학교 통합 검색 결과 (검색어: '{q_val}', 총 {total_cnt}건)]:"]

                            notices = s_data.get("notices", {}).get("items", []) if isinstance(s_data.get("notices"), dict) else []
                            if notices:
                                lines.append("- 📢 학교 공지사항:")
                                for n in notices[:5]:
                                    t = n.get("title") or "공지사항"
                                    n_id = n.get("id")
                                    # INTIP 내부 상세 페이지 경로 (/home/notice/{id}) 연결로 외부 브라우저 이탈 방지
                                    u = f"/home/notice/{n_id}" if n_id else (n.get("url") or "")
                                    d = n.get("createDate") or ""
                                    w = n.get("writer") or n.get("category") or ""
                                    link_str = f"[{t}]({u})" if u else t
                                    sub_str = f" (게시일: {d}, 작성: {w})" if d or w else ""
                                    lines.append(f"  • {link_str}{sub_str}")

                            dept_notices = s_data.get("departmentNotices", {}).get("items", []) if isinstance(s_data.get("departmentNotices"), dict) else []
                            if dept_notices:
                                lines.append("- 📢 학과 공지사항:")
                                for dn in dept_notices[:5]:
                                    t = dn.get("title") or "학과공지"
                                    u = dn.get("url") or ""
                                    dept = dn.get("departmentName") or dn.get("department") or ""
                                    d = dn.get("createDate") or ""
                                    link_str = f"[{t}]({u})" if u else t
                                    sub_str = f" (학과: {dept}, 게시일: {d})" if dept or d else ""
                                    lines.append(f"  • {link_str}{sub_str}")

                            dirs = s_data.get("directory", {}).get("items", []) if isinstance(s_data.get("directory"), dict) else []
                            if dirs:
                                lines.append("- 📞 교직원 및 학과 연락처:")
                                for di in dirs[:3]:
                                    name = di.get("name") or "교직원/학과"
                                    aff = di.get("detailAffiliation") or di.get("affiliation") or ""
                                    phone = di.get("phoneNumber") or ""
                                    pos = di.get("position") or di.get("duties") or ""
                                    lines.append(f"  • {name} ({aff}): 📞 {phone} {pos}".strip())

                            scheds = s_data.get("schedules", {}).get("items", []) if isinstance(s_data.get("schedules"), dict) else []
                            if scheds:
                                lines.append("- 📅 학사일정:")
                                for s in scheds[:5]:
                                    content = s.get("content") or s.get("title") or "학사일정"
                                    start = s.get("startDate") or s.get("start") or ""
                                    end = s.get("endDate") or s.get("end") or ""
                                    date_str = f" (기간: {start} ~ {end})" if start and end and start != end else (f" (날짜: {start})" if start else "")
                                    cal_link = f"/home/calendar?date={start}" if start else "/home/calendar"
                                    lines.append(f"  • [{content}]({cal_link}){date_str}")

                            courses = s_data.get("courses", {}).get("items", []) if isinstance(s_data.get("courses"), dict) else []
                            if courses:
                                lines.append("- 📚 개설강의:")
                                for c in courses[:3]:
                                    c_name = c.get("courseName") or "강의"
                                    prof = c.get("professor") or ""
                                    time_room = c.get("timeRoom") or ""
                                    credit = c.get("credit") or ""
                                    lines.append(f"  • {c_name} ({prof} 교수, {time_room}, {credit}학점)".strip())

                            clubs = s_data.get("clubs", {}).get("items", []) if isinstance(s_data.get("clubs"), dict) else []
                            if clubs:
                                lines.append("- 🎯 동아리:")
                                for cl in clubs[:3]:
                                    lines.append(f"  • {cl.get('name')} ({cl.get('category')}, {cl.get('room')})")

                            posts = s_data.get("posts", {}).get("items", []) if isinstance(s_data.get("posts"), dict) else []
                            if posts:
                                lines.append("- 💬 커뮤니티:")
                                for p in posts[:3]:
                                    lines.append(f"  • {p.get('title')} ({p.get('board')}, 추천: {p.get('likeCount')})")

                            if "오류가 발생했습니다" in mcp_summary:
                                status_state = "failed"
                                status_title = f"{tool_display_name} 조회 실패"
                                summary_out = (
                                    f"\n[시스템 오류 고지]: {tool_display_name} 조회 중 일시적인 교내 서버 응답 오류가 발생했습니다.\n"
                                    f"💡 [자율 대체 검색 지침]: 통합 검색 엔진(Elasticsearch)이 일시 장애 상태입니다. "
                                    f"사용자가 찾고자 하는 정보('{q_val}')를 제공하기 위해, 즉시 학교 공지사항 검색 도구(`api_notice`)를 호출하여 공지사항에서 정보를 찾아 안내하세요.\n"
                                )
                            elif total_cnt == 0:
                                status_state = "empty"
                                status_title = f"{tool_display_name} 결과 없음"
                                lines.append("- 검색 결과가 0건입니다.")
                                lines.append("💡 [자율 재검색 지침]: 결과가 0건이므로, 질문에서 불필요한 수식어를 덜어내거나 상위어/동의어로 검색어를 완화하여 1회 재검색(Query Expansion)할 수 있습니다. 이미 재검색했거나 마땅한 키워드가 없으면 검색 결과가 없음을 친절히 안내하세요.")
                                summary_out = "\n".join(lines) + "\n"
                            else:
                                status_state = "completed"
                                status_title = f"{tool_display_name} 확인 완료"
                                lines.append("💡 [중요 지침]: 검색 결과가 충분히 확보되었습니다. 추가 도구 호출을 즉시 중단하고, 위 공지 제목과 제공된 링크(학교공지는 INTIP 내부 상세 경로인 /home/notice/{공지ID})를 표나 목록에 마크다운 링크([공지제목](링크)) 형태로 그대로 포함하여 최종 답변을 작성하세요. 단순 텍스트로만 제목을 적지 마십시오.")
                                summary_out = "\n".join(lines) + "\n"
                        else:
                            summary_out = f"\n[{tool.category} 실시간 조회 데이터 ({tool.name})]:\n{json.dumps(res, ensure_ascii=False)[:1000]}\n"
                    elif tool.category == "WEATHER":
                        tool_data = res
                        w_raw = res.get("rawData") if (isinstance(res, dict) and isinstance(res.get("rawData"), dict)) else (res if isinstance(res, dict) else {})
                        mcp_summary = (res.get("summary") or "").strip() if isinstance(res, dict) else ""
                        if "오류가 발생했습니다" in mcp_summary or (not w_raw and not mcp_summary):
                            status_state = "failed"
                            status_title = f"{tool_display_name} 조회 실패"
                            summary_out = f"\n[날씨 실시간 조회 실패]: 현재 기상청/교내 날씨 서버와 통신할 수 없습니다.\n"
                        else:
                            status_state = "completed"
                            status_title = f"{tool_display_name} 확인 완료"
                            temp = w_raw.get("temperature") or w_raw.get("temp")
                            sky = w_raw.get("sky") or w_raw.get("weather") or ""
                            pm10 = w_raw.get("pm10Grade") or w_raw.get("pm10") or ""
                            pm25 = w_raw.get("pm25Grade") or w_raw.get("pm25") or ""
                            lines = ["\n[송도 캠퍼스 실시간 날씨 및 대기 정보]:"]
                            if mcp_summary:
                                lines.append(mcp_summary)
                            else:
                                if temp is not None:
                                    lines.append(f"- 현재 기온: {temp}°C")
                                if sky:
                                    lines.append(f"- 하늘 상태: {sky}")
                                if pm10:
                                    lines.append(f"- 미세먼지: {pm10}")
                                if pm25:
                                    lines.append(f"- 초미세먼지: {pm25}")
                            lines.append("💡 지침: 송도 캠퍼스 실시간 기온과 날씨, 미세먼지 정보를 학생에게 친절히 안내하세요.")
                            summary_out = "\n".join(lines) + "\n"
                    elif tool.category == "CLUB":
                        tool_data = res
                        c_raw = res.get("rawData") if (isinstance(res, dict) and isinstance(res.get("rawData"), dict)) else (res if isinstance(res, dict) else {})
                        mcp_summary = (res.get("summary") or "").strip() if isinstance(res, dict) else ""
                        items_list = c_raw.get("items") or res.get("items") or []
                        total_cnt = c_raw.get("totalCount", len(items_list)) if isinstance(c_raw, dict) else len(items_list)
                        if "오류" in mcp_summary:
                            status_state = "failed"
                            status_title = f"{tool_display_name} 조회 실패"
                            summary_out = f"\n[동아리 목록 조회 실패]: 동아리 정보를 불러오는 중 오류가 발생했습니다.\n"
                        elif total_cnt == 0 or (not items_list and "조회된 정보가 없습니다" in mcp_summary):
                            status_state = "empty"
                            status_title = f"{tool_display_name} 결과 없음"
                            summary_out = f"\n[교내 동아리 목록]: 현재 등록된 동아리 정보가 없습니다.\n"
                        else:
                            status_state = "completed"
                            status_title = f"{tool_display_name} 확인 완료"
                            summary_out = f"\n[교내 동아리 목록 조회 결과 ({total_cnt}건)]:\n{mcp_summary or json.dumps(items_list[:5], ensure_ascii=False)}\n"
                    elif tool.category == "LOST_PROPERTY":
                        tool_data = res
                        lp_raw = res.get("rawData") if (isinstance(res, dict) and isinstance(res.get("rawData"), dict)) else (res if isinstance(res, dict) else {})
                        mcp_summary = (res.get("summary") or "").strip() if isinstance(res, dict) else ""
                        items_list = lp_raw.get("items") or res.get("items") or []
                        if "오류" in mcp_summary:
                            status_state = "failed"
                            status_title = f"{tool_display_name} 조회 실패"
                            summary_out = f"\n[분실물 조회 실패]: 학내 분실물 목록 조회 중 일시적인 시스템 오류가 발생했습니다.\n"
                        elif not items_list and ("없습니다" in mcp_summary or "0건" in mcp_summary or not lp_raw):
                            status_state = "empty"
                            status_title = f"{tool_display_name} 결과 없음"
                            summary_out = f"\n[학내 분실물 목록]: 현재 등록된 분실물/습득물 내역이 없습니다.\n"
                        else:
                            status_state = "completed"
                            status_title = f"{tool_display_name} 확인 완료"
                            summary_out = f"\n[학내 분실물/습득물 조회 결과]:\n{mcp_summary or json.dumps(items_list[:5], ensure_ascii=False)}\n"
                    elif tool.category == "TIMETABLE_GAP":
                        tool_data = res
                        mcp_summary = (res.get("summary") or "").strip() if isinstance(res, dict) else ""
                        ui_comp = res.get("uiComponent") if isinstance(res, dict) else None
                        ui_type = ui_comp.get("type") if isinstance(ui_comp, dict) else None
                        if "오류" in mcp_summary:
                            status_state = "failed"
                            status_title = f"{tool_display_name} 조회 실패"
                            summary_out = f"\n[공강 시간 분석 오류]: {mcp_summary}\n"
                        elif ui_type == "AUTH_REQUIRED" or "로그인" in mcp_summary:
                            status_state = "empty"
                            status_title = f"{tool_display_name} 결과 없음"
                            summary_out = (
                                f"\n[공강 시간 분석 안내]:\n{mcp_summary}\n"
                                f"💡 지침: 공강 시간을 분석하려면 인팁(INTIP) 시간표가 필요함을 안내하세요.\n"
                            )
                        else:
                            status_state = "completed"
                            status_title = f"{tool_display_name} 확인 완료"
                            summary_out = f"\n[공강 시간 분석 결과]:\n{mcp_summary}\n"
                    else:
                        tool_data = res
                        mcp_summary = (res.get("summary") or "").strip() if isinstance(res, dict) else ""
                        if mcp_summary:
                            if "오류" in mcp_summary or "실패" in mcp_summary:
                                status_state = "failed"
                                status_title = f"{tool_display_name} 조회 실패"
                            elif any(k in mcp_summary for k in ["결과가 없습니다", "조회 결과가 없습니다", "등록된 정보가 없습니다", "내역이 없습니다", "0건"]):
                                status_state = "empty"
                                status_title = f"{tool_display_name} 결과 없음"
                            else:
                                status_state = "completed"
                                status_title = f"{tool_display_name} 확인 완료"
                            summary_out = f"\n[{tool_display_name} 조회 결과 ({tool.name})]:\n{mcp_summary}\n"
                        else:
                            summary_out = f"\n[{tool.category} 실시간 조회 데이터 ({tool.name})]:\n{json.dumps(res, ensure_ascii=False)[:1000]}\n"
            except Exception as ex:
                logger.warning(f"Error executing tool {tool.name}: {ex}")
                status_state = "failed"
                status_title = f"{tool_display_name} 조회 실패"
                summary_out = (
                    f"\n[시스템 오류 고지]: {tool_display_name} 조회 중 일시적인 교내 통신 오류({str(ex)})가 발생하여 실시간 정보를 가져오지 못했습니다.\n"
                    f"⚠️ 핵심 응답 지침: 절대로 임의의 가상 정보를 지어내지 말고, '현재 교내 시스템 일시 오류로 실시간 정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요'라고 사실대로 사용자에게 안내하세요.\n"
                )

            if tool.category not in emitted_cards and tool_data:
                card = CardSynthesizer.synthesize_for_domain(
                    domain=tool.category,
                    tool_name=tool.name,
                    data=tool_data,
                    query=request.message,
                )
                if card:
                    yield (AgentStreamEvent(event_type="CARD", card=card), "", None, None)
                    emitted_cards.add(tool.category)

            yield (
                AgentStreamEvent(
                    event_type="STATUS",
                    status_id=tool_id,
                    status_title=status_title,
                    status_category=tool.category,
                    status_state=status_state,
                ),
                summary_out,
                None,
                None,
            )

        # Case C: INUChat Official Knowledge RAG Tool
        elif tool.category == "INU_AI_KNOWLEDGE":
            yield (
                AgentStreamEvent(
                    event_type="STATUS",
                    status_id=tool_id,
                    status_title=f"{tool_display_name} 검색 중...",
                    status_category="INU_AI_KNOWLEDGE",
                    status_state="running",
                ),
                "",
                None,
                None,
            )

            clean_msg = sanitize_text(request.message)
            inu_q = clean_msg
            if academic_context:
                ac_parts = []
                if academic_context.get("entryYear"):
                    ac_parts.append(f"입학연도={academic_context['entryYear']}학번")
                if academic_context.get("departmentName"):
                    ac_parts.append(f"학과={academic_context['departmentName']}")
                if academic_context.get("enrollmentStatus"):
                    ac_parts.append(f"학적상태={academic_context['enrollmentStatus']}")
                if academic_context.get("acquiredCredits"):
                    ac_parts.append(f"취득학점={academic_context['acquiredCredits']}학점")
                if academic_context.get("gradeAverage"):
                    ac_parts.append(f"평점평균={academic_context['gradeAverage']}")
                if ac_parts:
                    inu_q = f"{clean_msg}\n\n[비식별 학적 참고정보: {', '.join(ac_parts)}]"

            status_state = "completed"
            status_title = f"{tool_display_name} 확인 완료"

            try:
                res = await tool.execute({"question": inu_q}, exec_context)
                if isinstance(res, dict) and res.get("success") and res.get("data"):
                    rag_data = res.get("data", {})
                    rag_answer = rag_data.get("rag_answer", "")
                    rag_data_out = rag_data
                    summary_out = f"\n[인천대학교 공식 학칙/규정 지식베이스 (INUChat RAG 검색 결과)]:\n{rag_answer}\n"

                    if "INU_AI_KNOWLEDGE" not in emitted_cards:
                        card = CardSynthesizer.synthesize_for_domain(
                            domain="INU_AI_KNOWLEDGE",
                            tool_name=tool.name,
                            data=rag_data,
                            query=request.message,
                        )
                        if card:
                            yield (AgentStreamEvent(event_type="CARD", card=card), "", None, None)
                            emitted_cards.add("INU_AI_KNOWLEDGE")
                else:
                    status_state = "completed"
                    status_title = f"{tool_display_name} 결과 없음"
                    summary_out = (
                        f"\n[시스템 학칙 지식베이스 검색 결과 없음]: 관련 학칙/규정 정보를 찾지 못했습니다.\n"
                        f"⚠️ 핵심 지침: 가상의 학칙 규정을 지어내지 말고, 지식베이스에서 해당 내용을 찾지 못했음을 사실대로 안내하세요.\n"
                    )
            except Exception as ex:
                logger.warning(f"Error executing INUChat tool {tool.name}: {ex}")
                status_state = "failed"
                status_title = f"{tool_display_name} 조회 실패"
                summary_out = (
                    f"\n[시스템 오류 고지]: 학칙/규정 지식베이스 검색 중 일시적인 오류({str(ex)})가 발생했습니다.\n"
                    f"⚠️ 핵심 지침: 가상의 학칙 조항을 지어내지 말고, 일시적인 지식베이스 통신 지연 사실을 안내하세요.\n"
                )

            yield (
                AgentStreamEvent(
                    event_type="STATUS",
                    status_id=tool_id,
                    status_title=status_title,
                    status_category="INU_AI_KNOWLEDGE",
                    status_state=status_state,
                ),
                summary_out,
                None,
                rag_data_out,
            )

    async def resume_stream_with_action_result(
        self,
        callback: ChatActionCallbackRequest,
    ) -> AsyncGenerator[AgentStreamEvent, None]:
        """
        Resumes the conversation stream after the client executes an on-demand P2P action.
        Synthesizes the actual SDUI card, chains secondary tools if needed (e.g., professor contact search),
        and streams the final synthesized answer.
        """
        logger.info(
            f"[ACTION_RESUME_START] action_id={callback.action_id} "
            f"success={callback.success} session_id={callback.session_id} "
            f"query='{(callback.original_message or '')[:40]}'"
        )

        act_id_lower = callback.action_id.lower()
        if any(kw in act_id_lower for kw in ["timetable", "sreg", "tlsn"]):
            domain = "TIMETABLE"
        elif any(kw in act_id_lower for kw in ["lms", "assign", "course"]):
            domain = "LMS"
        elif any(kw in act_id_lower for kw in ["library", "lib", "seat"]):
            domain = "LIBRARY"
        else:
            domain = "PORTAL"

        domain_display_name = resolve_tool_display_name(domain, callback.action_id)

        # 1. Action execution failed on client/device
        if not callback.success:
            err_msg = callback.error_message or "학교 공식 시스템 연동 중 일시적인 오류가 발생했습니다."
            logger.warning(f"[ACTION_RESUME_FAILED] action_id={callback.action_id} error={err_msg}")

            yield AgentStreamEvent(
                event_type="STATUS",
                status_id=f"tool_{domain.lower()}",
                status_title=f"{domain_display_name} 조회 실패",
                status_category=domain,
                status_state="failed",
            )

            fail_card = CardSynthesizer.synthesize_for_domain(
                domain=domain,
                data={"status": "FETCH_FAILED", "message": err_msg},
                query=callback.original_message or "",
            )
            if fail_card:
                yield AgentStreamEvent(event_type="CARD", card=fail_card)

            yield AgentStreamEvent(
                event_type="TOKEN",
                content=f"학교 시스템(ERP/포털) 연동 중 오류가 발생했습니다: {err_msg}\n잠시 후 다시 시도해 주세요.",
            )
            yield AgentStreamEvent(event_type="DONE")
            return

        # 2. Action execution succeeded -> Synthesize Card & Grounding
        data = callback.data or {}
        is_empty_data = False
        if domain == "TIMETABLE":
            raw_courses = data if isinstance(data, list) else (
                data.get("courses") or data.get("items") or data.get("timetable") or []
            ) if isinstance(data, dict) else []
            if not raw_courses:
                is_empty_data = True
        elif domain == "LMS":
            events = data.get("events", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            if not events:
                is_empty_data = True
        elif not data:
            is_empty_data = True

        status_title = f"{domain_display_name} 결과 없음" if is_empty_data else f"{domain_display_name} 확인 완료"

        yield AgentStreamEvent(
            event_type="STATUS",
            status_id=f"tool_{domain.lower()}",
            status_title=status_title,
            status_category=domain,
            status_state="completed",
        )

        data = callback.data or {}
        card = CardSynthesizer.synthesize_for_domain(
            domain=domain,
            data=data,
            query=callback.original_message or "",
        )
        if card:
            yield AgentStreamEvent(event_type="CARD", card=card)

        # Build domain summary for LLM grounding
        tool_summary_lines: List[str] = []
        advisor_name: Optional[str] = None

        if domain == "TIMETABLE":
            tool_summary_lines.append("\n[학교 포털 수강신청 시간표 공식 데이터]:")
            raw_courses = data if isinstance(data, list) else (
                data.get("courses") or data.get("items") or data.get("timetable") or []
            ) if isinstance(data, dict) else []

            if raw_courses:
                for idx, c in enumerate(raw_courses):
                    if isinstance(c, dict):
                        c_name = c.get("courseName") or c.get("title") or c.get("subject") or "과목"
                        prof = c.get("professor") or c.get("profNm") or "미지정"
                        time_str = c.get("time") or c.get("classTime") or c.get("timeStr") or "시간미정"
                        room = c.get("classroom") or c.get("room") or c.get("timeRoom") or "강의실미정"
                        tool_summary_lines.append(f"- {c_name}: 담당교수={prof}, 시간={time_str}, 강의실={room}")
            else:
                tool_summary_lines.append("- 이번 학기 등록된 수강신청 내역이 없습니다.")

        elif domain == "PORTAL":
            tool_summary_lines.append("\n[학교 종합정보시스템(ERP) 학적 기본 데이터]:")
            if isinstance(data, dict):
                advisor_name = data.get("advisorProfessorName") or data.get("advisor") or data.get("profNm")
                std_nm = data.get("studentName") or data.get("name") or "학생"
                std_no = data.get("studentId") or data.get("stdNo") or ""
                dept = data.get("department") or data.get("dept") or ""
                status = data.get("academicStatus") or data.get("status") or "재학"
                credits = data.get("earnedCredits") or data.get("credits") or ""
                gpa = data.get("gradeAverage") or data.get("gpa") or ""

                if std_nm: tool_summary_lines.append(f"- 성명: {std_nm}")
                if std_no: tool_summary_lines.append(f"- 학번: {std_no}")
                if dept: tool_summary_lines.append(f"- 소속: {dept}")
                if status: tool_summary_lines.append(f"- 학적상태: {status}")
                if credits: tool_summary_lines.append(f"- 취득학점: {credits}학점")
                if gpa: tool_summary_lines.append(f"- 평점평균: {gpa}")
                if advisor_name: tool_summary_lines.append(f"- 지도교수: {advisor_name} 교수님")
            else:
                tool_summary_lines.append(f"- 데이터: {str(data)}")

        elif domain == "LMS":
            tool_summary_lines.append("\n[이러닝(LMS) 과제 및 강의 데이터]:")
            events = data.get("events", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for ev in events[:5]:
                if isinstance(ev, dict):
                    ev_name = ev.get("name") or ev.get("fullname") or "과제"
                    tool_summary_lines.append(f"- 과제/일정: {ev_name}")

        orig_query = callback.original_message or ""
        is_contact_query = any(kw in orig_query for kw in ["연락처", "전화번호", "전화", "연구실", "번호", "이메일", "메일", "교수님", "교수", "담임교수", "지도교수", "과사", "사무실", "찾아줘"])

        # 3. Multi-Hop Chaining: Contact query + Advisor Professor found
        if is_contact_query and advisor_name:
            logger.info(f"[MULTI_HOP_CHAIN] Chaining to api_directory for advisor '{advisor_name}'")
            yield AgentStreamEvent(
                event_type="STATUS",
                status_id="contact_chain_search",
                status_title=f"{advisor_name} 교수님 교내 연락처 및 연구실 조회 중...",
                status_category="DIRECTORY",
                status_state="running",
            )

            dir_tool = tool_registry.get_tool("api_directory")
            if dir_tool:
                try:
                    dir_res = await dir_tool.execute(
                        {"query": advisor_name},
                        {"client": callback.client, "auth": "", "authorization": "", "query": advisor_name},
                    )
                    dir_card = CardSynthesizer.synthesize_for_domain(
                        domain="DIRECTORY",
                        tool_name="api_directory",
                        data=dir_res,
                        query=orig_query,
                    )
                    if dir_card:
                        yield AgentStreamEvent(event_type="CARD", card=dir_card)

                    contacts = dir_res if isinstance(dir_res, list) else (
                        dir_res.get("contacts") or dir_res.get("data") or []
                    ) if isinstance(dir_res, dict) else []

                    tool_summary_lines.append(f"\n[교내 전화번호부 {advisor_name} 교수님 연락처 검색 결과]:")
                    if contacts:
                        for c in contacts[:3]:
                            if isinstance(c, dict):
                                c_name = c.get("name") or advisor_name
                                c_dept = c.get("dept") or c.get("department") or ""
                                c_tel = c.get("telephone") or c.get("phone") or c.get("tel") or "전화번호 없음"
                                c_email = c.get("email") or ""
                                c_loc = c.get("location") or c.get("room") or ""
                                tool_summary_lines.append(
                                    f"- {c_name} ({c_dept}): 전화={c_tel}, 이메일={c_email}, 위치={c_loc}"
                                )
                        chain_status_title = f"{advisor_name} 교수님 연락처 확인 완료"
                    else:
                        tool_summary_lines.append(f"- {advisor_name} 교수님의 공식 등록 연락처를 찾지 못했습니다.")
                        chain_status_title = f"{advisor_name} 교수님 연락처 결과 없음"

                    yield AgentStreamEvent(
                        event_type="STATUS",
                        status_id="contact_chain_search",
                        status_title=chain_status_title,
                        status_category="DIRECTORY",
                        status_state="completed",
                    )
                except Exception as ex:
                    logger.warning(f"Error running chained api_directory: {ex}")
                    yield AgentStreamEvent(
                        event_type="STATUS",
                        status_id="contact_chain_search",
                        status_title=f"{advisor_name} 교수님 연락처 조회 실패",
                        status_category="DIRECTORY",
                        status_state="failed",
                    )

        # 4. Final LLM Response Synthesis Stream
        tool_summary_text = "\n".join(tool_summary_lines)
        system_prompt = get_system_prompt_for_client(callback.client, tool_summary=tool_summary_text)

        synthesis_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

        if callback.history:
            for h in callback.history[-6:]:
                role = getattr(h, "role", None) or (h.get("role") if isinstance(h, dict) else "user")
                content = getattr(h, "content", None) or (h.get("content") if isinstance(h, dict) else "")
                if content:
                    clean_role = "assistant" if role == "assistant" else "user"
                    synthesis_messages.append({"role": clean_role, "content": content})

        synthesis_messages.append({
            "role": "user",
            "content": orig_query or "학교 시스템에서 조회된 내역을 친절하게 안내해줘."
        })

        try:
            tokens_emitted = 0
            async for token in llm_client.stream_chat(messages=synthesis_messages):
                tokens_emitted += 1
                yield AgentStreamEvent(event_type="TOKEN", content=token)
            if tokens_emitted == 0:
                yield AgentStreamEvent(
                    event_type="TOKEN",
                    content="학교 시스템 데이터 조회를 완료하였습니다.",
                )
            yield AgentStreamEvent(event_type="DONE")
        except Exception as e:
            logger.error(f"[ACTION_RESUME_ERROR] Synthesis error: {e}", exc_info=True)
            yield AgentStreamEvent(event_type="ERROR", error=str(e))


orchestrator = AgentOrchestrator()

