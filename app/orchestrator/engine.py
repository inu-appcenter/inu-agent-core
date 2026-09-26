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
from app.llm.schemas import ChatRequest, AgentStreamEvent
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
        raw_token = client_ctx.get("auth") or client_ctx.get("authorization", "")
        clean_token = raw_token.replace("Bearer ", "").strip() if raw_token else ""

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

        # 3. Native Tool Calling ReAct Loop (Autonomous Multi-Hop with Streaming Thought)
        MAX_HOPS = 4
        for hop in range(MAX_HOPS):
            response = None
            streamed_any_thought = False
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

            # If thought was not streamed token-by-token, yield it now
            if thought and not streamed_any_thought:
                yield AgentStreamEvent(
                    event_type="THINKING",
                    thinking=thought,
                )

            if not tool_calls:
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

            if inuchat_tool and hasattr(inuchat_tool, "stream_execute"):
                async for token, is_done, full_acc, citations in inuchat_tool.stream_execute({"question": inu_question}):
                    if token:
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

                yield AgentStreamEvent(event_type="DONE")
                return

        # Case 2: Composite Query or General Campus Tools -> Synthesis LLM
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

        try:
            logger.info(
                f"Starting synthesis stream for '{request.message[:30]}...' (Categories: {executed_categories})"
            )

            async for token in llm_client.stream_chat(messages=synthesis_messages):
                yield AgentStreamEvent(event_type="TOKEN", content=token)

            yield AgentStreamEvent(event_type="DONE")
        except Exception as e:
            logger.error(f"Error in orchestrator stream: {e}", exc_info=True)
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
        tool_id = f"tool_{tool.category.lower()}_{abs(hash(tool.name)) % 10000}"

        summary_out = ""
        ac_data_out = {}
        rag_data_out = None

        # Case A: Client Action (LMS, Portal ERP)
        if tool.category in ["LMS", "PORTAL"]:
            yield (
                AgentStreamEvent(
                    event_type="STATUS",
                    status_id=tool_id,
                    status_title=f"{tool_display_name} 연동 중...",
                    status_category=tool.category,
                    status_state="running",
                ),
                "",
                None,
                None,
            )

            client_ctx = request.client_context or {}
            portal_meta = client_ctx.get("portal") if isinstance(client_ctx.get("portal"), dict) else {}
            is_portal_linked = portal_meta.get("linked") is True

            domain_data = None
            if tool.category == "PORTAL":
                domain_data = (
                    client_ctx.get("academic")
                    or client_ctx.get("academicDisplay")
                    or client_ctx.get("studentInfo")
                    or client_ctx.get("student")
                )
                if not domain_data and ("departmentName" in client_ctx or "advisorProfessorName" in client_ctx):
                    domain_data = client_ctx
            else:
                domain_data = client_ctx.get("lms")

            if domain_data and isinstance(domain_data, (dict, list)):
                if tool.category == "PORTAL" and isinstance(domain_data, dict):
                    display_data = client_ctx.get("academicDisplay")
                    merged_portal = {**domain_data, **(display_data if isinstance(display_data, dict) else {})}

                    # 100% Zero-PII: Sanitize record (no names, no full student IDs, no personal phone numbers)
                    clean_academic = sanitize_academic_record(merged_portal)
                    ac_data_out = clean_academic
                    summary_out = build_anonymized_academic_summary(clean_academic)
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
                    card_source = (client_ctx.get("academicDisplay") or domain_data) if tool.category == "PORTAL" else domain_data
                    card = CardSynthesizer.synthesize_for_domain(
                        domain=tool.category,
                        tool_name=tool.name,
                        data=card_source,
                        query=request.message,
                    )
                    if card:
                        yield (AgentStreamEvent(event_type="CARD", card=card), "", None, None)
                        emitted_cards.add(tool.category)
            else:
                if tool.category not in emitted_actions:
                    try:
                        action_instruction = await tool.execute({}, exec_context)
                        if hasattr(action_instruction, "action_id"):
                            yield (AgentStreamEvent(event_type="ACTION_REQUIRED", action=action_instruction), "", None, None)
                            emitted_actions.add(tool.category)
                    except Exception as ex:
                        logger.warning(f"Failed to generate client action for {tool.name}: {ex}")

                if tool.category == "PORTAL" and is_portal_linked:
                    # 계정 연동은 되어 있으나 히든 웹뷰/ERP 응답 지연인 경우
                    err_msg = portal_meta.get("academicErrorMessage") or "포털 또는 ERP 응답을 확인하지 못했습니다."
                    if tool.category not in emitted_cards:
                        fetch_fail_card = CardSynthesizer.synthesize_for_domain(
                            domain="PORTAL",
                            tool_name=tool.name,
                            data={"status": "FETCH_FAILED", "message": err_msg},
                            query=request.message,
                        )
                        if fetch_fail_card:
                            yield (AgentStreamEvent(event_type="CARD", card=fetch_fail_card), "", None, None)
                            emitted_cards.add(tool.category)

                    summary_out = (
                        f"\n[PORTAL_STATUS]: 학생의 포털 계정은 정상 연동되어 있으나, 학교 ERP(종합정보시스템) 응답 지연으로 학적 정보를 일시적으로 불러오지 못했습니다. ({err_msg}) "
                        "학생에게 계정 연동은 잘 유지되어 있으니 잠시 후 같은 질문을 다시 보내달라고 친절하게 안내하세요. (⚠️ 중요 지침: 계정 연동 카드를 다시 누르라고 절대 안내하지 마세요!)\n"
                    )
                else:
                    # 연동 데이터가 없을 때 미연동 안내 카드(PORTAL_AUTH_REQUIRED / LMS_AUTH_REQUIRED)를 즉시 합성하여 전달
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

            yield (
                AgentStreamEvent(
                    event_type="STATUS",
                    status_id=tool_id,
                    status_title=f"{tool_display_name} 연동 확인",
                    status_category=tool.category,
                    status_state="completed",
                ),
                summary_out,
                ac_data_out,
                None,
            )

        # Case B: Server OpenAPI / Direct Tools
        elif tool.category in ["CAFETERIA", "BUS", "TIMETABLE", "NOTICE", "SCHEDULE", "DIRECTORY", "WEATHER", "LIBRARY", "CAMPUS_WATCH", "REMINDER", "DAILY_BRIEF", "KEYWORD", "SETTINGS"]:
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
                user_stop = str(final_args.get("bstopId") or final_args.get("stop_name") or "").strip()
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

            status_state = "completed"
            status_title = f"{tool_display_name} 확인 완료"

            try:
                res = await tool.execute(final_args, exec_context)
                if isinstance(res, dict) and "error" in res:
                    err_msg = res.get("error", "알 수 없는 통신 오류")
                    status_state = "failed"
                    status_title = f"{tool_display_name} 조회 실패"
                    summary_out = (
                        f"\n[시스템 오류 고지]: {tool_display_name} 조회 중 일시적인 교내 서버 응답 지연 또는 오류({err_msg})가 발생하여 실시간 정보를 가져오지 못했습니다.\n"
                        f"⚠️ 핵심 응답 지침: 절대로 임의의 가상 정보(식단 메뉴, 버스 도착 시간, 전화번호, 시간표 등)를 지어내지 말고, "
                        f"'현재 교내 시스템 일시 오류로 실시간 정보를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요'라고 사실대로 사용자에게 안내하세요.\n"
                    )
                elif res is not None:
                    if tool.category == "BUS" and bus_meta:
                        valid_routes = bus_meta.get("validRoutes", [])
                        stop_name = bus_meta.get("stopName", "정류소")
                        tab_name = bus_meta.get("tabName", "정류장")
                        raw_list = res if isinstance(res, list) else res.get("items", [])
                        filtered_arrivals = [
                            item for item in raw_list
                            if isinstance(item, dict) and item.get("routeNo") in valid_routes
                        ] if valid_routes else raw_list

                        tool_data = {
                            "arrivals": filtered_arrivals,
                            "validRoutes": valid_routes,
                            "stopName": stop_name,
                            "tabName": tab_name,
                        }
                        if filtered_arrivals:
                            summary_out = (
                                f"\n[BUS 인천시 실시간 시내버스 도착 정보 ({stop_name})]:\n"
                                f"- 모니터링 노선: {', '.join(valid_routes)}\n"
                                f"- 실시간 도착 정보:\n{json.dumps(filtered_arrivals, ensure_ascii=False)}\n"
                            )
                        else:
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
                        if isinstance(res, list):
                            raw_menus = res
                        elif isinstance(res, dict):
                            raw_menus = res.get("data") or res.get("items") or res.get("menus") or res.get("cafeterias") or []
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
                                    corner = m.get("name") or m.get("cornerName") or m.get("corner") or "코너"
                                    menu_str = m.get("menu") or m.get("menuName") or ""
                                    meal_label = m.get("mealLabel") or m.get("mealType") or ""
                                    if menu_str and menu_str != "-":
                                        has_valid_menu = True
                                        prefix = f"[{meal_label}] " if meal_label else ""
                                        lines.append(f"- {prefix}{corner}: {menu_str}")

                        if has_valid_menu:
                            lines.append("\n⚠️ 핵심 지침: 반드시 위 실제 조회된 메뉴 목록에 근거하여 안내하세요. 위 목록에 없는 가상의 메뉴를 절대로 추가하거나 지어내지 마세요.")
                            summary_out = "\n".join(lines) + "\n"
                        else:
                            summary_out = (
                                f"\n[CAFETERIA {caf_name} 식단 메뉴 조회 결과]:\n"
                                f"- 현재 {caf_name}에 등록된 식단 메뉴가 없습니다 (식당 운영 시간 외 또는 식단 미등록 상태).\n"
                                f"⚠️ 핵심 지침: 절대로 가상의 식단 메뉴를 지어내어 답변하지 마세요. 학생에게 '현재 {caf_name}의 식단 정보가 등록되어 있지 않거나 식당 미운영 상태입니다'라고 사실대로 안내하세요.\n"
                            )
                    elif tool.category == "LIBRARY" and isinstance(res, dict):
                        tool_data = res
                        mode = res.get("mode")
                        if mode == "STUDY_ROOMS":
                            study_rooms = res.get("rooms") or res.get("data", {}).get("rooms", [])
                            rooms_info = "\n".join([
                                f"- {r.get('name')}: 위치 {r.get('location')}, 수용정원 {r.get('quota')}, 구비시설 ({', '.join(r.get('tags', []))})"
                                for r in study_rooms
                            ])
                            notice = res.get("data", {}).get("notice") or res.get("notice", "")
                            summary_out = (
                                f"\n[학산도서관 스터디룸 목록 및 예약 안내]:\n"
                                f"{rooms_info}\n"
                                f"- 이용 규정: {notice}\n"
                                f"💡 지침: 학생에게 위 스터디룸 목록을 친절히 안내하고, 대화창의 [학산도서관 스터디룸] 카드에서 원하는 방을 터치하면 바로 예약 화면으로 연결된다고 설명하세요.\n"
                            )
                        elif mode in ["RESERVE_SEAT", "RESERVE_STUDY_ROOM"]:
                            summary_out = (
                                f"\n[학산도서관 대화형 신청 카드 발급 완료]:\n"
                                f"{res.get('instruction', '')}\n"
                            )
                        else:
                            rooms = res.get("rooms", [])
                            rooms_info = "\n".join([
                                f"- {r.get('name', '')}: 잔여 {r.get('available_seats', r.get('seats', {}).get('available', 0))}석 / 전체 {r.get('total_seats', r.get('seats', {}).get('total', 0))}석"
                                for r in rooms
                            ])
                            summary_out = (
                                f"\n[학산도서관 열람실 실시간 잔여 좌석 현황 (공식 pyxis 시스템 실시간 관측 데이터)]:\n"
                                f"{rooms_info}\n"
                                f"⚠️ 지침: 반드시 위 실제 실시간 잔여 좌석 수치 그대로 학생에게 안내하세요 (임의의 숫자를 지어내지 마세요). "
                                f"대화창 아래 제공된 실시간 열람실 카드에서 원하는 열람실을 터치하면 좌석 배정 화면으로 이동할 수 있음을 덧붙이세요.\n"
                            )
                    elif tool.category == "CAMPUS_WATCH" and isinstance(res, dict):
                        tool_data = res
                        summary_out = f"\n[학산도서관 실시간 빈자리 알림/스나이퍼 감시 결과]:\n{res.get('summary', '')}\n"
                    elif tool.category == "TIMETABLE":
                        tool_data = res if res is not None else {}
                        classes = []
                        if isinstance(res, list):
                            classes = res
                        elif isinstance(res, dict):
                            classes = res.get("items") or res.get("todayClasses") or res.get("courses") or []

                        if not classes:
                            summary_out = (
                                "\n[INTIP 인팁 시간표 조회 결과]:\n"
                                "- 오늘 등록된 수업/강의 일정이 없습니다 (또는 인팁 앱에 시간표가 등록되어 있지 않습니다).\n"
                                "💡 지침: 학생에게 오늘 예정된 수업이 없거나 시간표가 등록되지 않았음을 친절히 안내하고, '인팁 앱의 [시간표] 탭에서 이번 학기 시간표를 추가하거나 확인할 수 있어요'라고 안내하세요.\n"
                                "⚠️ 중요: 이 기능은 인팁(INTIP) 앱 자체의 시간표 기능이므로, 포털/LMS 계정 연동을 절대 요구하지 마세요.\n"
                            )
                        else:
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
                            raw_list = res.get("contents") or res.get("items") or (res.get("data", {}).get("contents") if isinstance(res.get("data"), dict) else []) or []

                        if "college" in tool.name.lower() or "office" in tool.name.lower():
                            contacts = raw_list
                            lines = [f"\n[교내 학과/단과대 사무실(과사) 연락처 조회 결과 (검색어: '{q_param}')]:"]
                            if contacts:
                                for c in contacts[:5]:
                                    if isinstance(c, dict):
                                        dept = c.get("departmentName") or c.get("deptName") or ""
                                        colg = c.get("collegeName") or ""
                                        phone = c.get("officePhoneNumber") or c.get("phoneNumber") or ""
                                        loc = c.get("officeLocation") or c.get("location") or ""
                                        dept_label = f"{dept} ({colg})" if colg else dept
                                        lines.append(f"- {dept_label} 학과 사무실: 📞 {phone}" + (f" (위치: {loc})" if loc else ""))
                            else:
                                lines.append(f"- '{q_param}' 관련 학과 사무실 연락처가 조회되지 않았습니다.")
                            lines.append("💡 지침: 위 학과 사무실 번호는 학과 사무실(과사) 번호이며 교수님 개인 연구실 번호가 아닙니다. 학과 사무실 번호임을 명확히 구분하여 안내하세요.")
                            summary_out = "\n".join(lines) + "\n"
                        else:
                            entries = raw_list
                            lines = [f"\n[교내 교수/교직원/부서 연락처 검색 결과 (검색어: '{q_param}')]:"]
                            if entries:
                                for e in entries[:5]:
                                    if isinstance(e, dict):
                                        name = e.get("name") or ""
                                        pos = e.get("position") or ""
                                        affil = e.get("detailAffiliation") or e.get("affiliation") or ""
                                        phone = e.get("phoneNumber") or ""
                                        email = e.get("email") or ""
                                        lines.append(f"- {name} ({pos}, {affil}): 📞 {phone}" + (f", ✉️ {email}" if email else ""))
                            else:
                                lines.append(f"- 검색어 '{q_param}' 관련 교수/교직원 개인 연락처가 교내 전화번호부 DB에 등록되어 있지 않습니다.")
                                lines.append("💡 지침: 교수님 개인 연락처가 조회되지 않고 학과 사무실 번호만 있는 경우, '{교수명} 교수님의 개인 연락처는 등록되어 있지 않으나, 소속 학과인 {학과명} 학과 사무실({번호})로 문의하실 수 있습니다'라고 맥락을 밝혀 친절히 안내하세요.")
                            summary_out = "\n".join(lines) + "\n"
                    elif tool.category == "NOTICE":
                        tool_data = res
                        notices_list = []
                        if isinstance(res, list):
                            notices_list = res
                        elif isinstance(res, dict):
                            notices_list = res.get("contents") or res.get("items") or res.get("notices") or []

                        lines = [f"\n[인천대학교 공지사항 조회 결과 ({tool_display_name})]:"]
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
                            lines.append("💡 [중요 지침]: 위 공지 제목과 내부 상세 링크(/home/notice/{공지ID})를 마크다운 링크 형식([공지제목](/home/notice/{id}))으로 답변에 그대로 포함하여 안내하세요. 외부 브라우저로 나가지 않고 INTIP 앱 내부 상세 페이지로 열립니다.")
                        else:
                            lines.append("- 조회된 공지사항이 없습니다.")
                        summary_out = "\n".join(lines) + "\n"
                    elif tool.category == "SEARCH" or "unifiedsearch" in tool.name.lower():
                        tool_data = res
                        if isinstance(res, dict):
                            total_cnt = res.get("totalCount", 0)
                            q_val = res.get("query", "")
                            lines = [f"\n[인천대학교 통합 검색 결과 (검색어: '{q_val}', 총 {total_cnt}건)]:"]

                            notices = res.get("notices", {}).get("items", []) if isinstance(res.get("notices"), dict) else []
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

                            dept_notices = res.get("departmentNotices", {}).get("items", []) if isinstance(res.get("departmentNotices"), dict) else []
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

                            dirs = res.get("directory", {}).get("items", []) if isinstance(res.get("directory"), dict) else []
                            if dirs:
                                lines.append("- 📞 교직원 및 학과 연락처:")
                                for di in dirs[:3]:
                                    name = di.get("name") or "교직원/학과"
                                    aff = di.get("detailAffiliation") or di.get("affiliation") or ""
                                    phone = di.get("phoneNumber") or ""
                                    pos = di.get("position") or di.get("duties") or ""
                                    lines.append(f"  • {name} ({aff}): 📞 {phone} {pos}".strip())

                            scheds = res.get("schedules", {}).get("items", []) if isinstance(res.get("schedules"), dict) else []
                            if scheds:
                                lines.append("- 📅 학사일정:")
                                for s in scheds[:3]:
                                    content = s.get("content") or "학사일정"
                                    start = s.get("startDate") or ""
                                    end = s.get("endDate") or ""
                                    date_str = f" (기간: {start} ~ {end})" if start and end else ""
                                    lines.append(f"  • {content}{date_str}")

                            courses = res.get("courses", {}).get("items", []) if isinstance(res.get("courses"), dict) else []
                            if courses:
                                lines.append("- 📚 개설강의:")
                                for c in courses[:3]:
                                    c_name = c.get("courseName") or "강의"
                                    prof = c.get("professor") or ""
                                    time_room = c.get("timeRoom") or ""
                                    credit = c.get("credit") or ""
                                    lines.append(f"  • {c_name} ({prof} 교수, {time_room}, {credit}학점)".strip())

                            clubs = res.get("clubs", {}).get("items", []) if isinstance(res.get("clubs"), dict) else []
                            if clubs:
                                lines.append("- 🎯 동아리:")
                                for cl in clubs[:3]:
                                    lines.append(f"  • {cl.get('name')} ({cl.get('category')}, {cl.get('room')})")

                            posts = res.get("posts", {}).get("items", []) if isinstance(res.get("posts"), dict) else []
                            if posts:
                                lines.append("- 💬 커뮤니티:")
                                for p in posts[:3]:
                                    lines.append(f"  • {p.get('title')} ({p.get('board')}, 추천: {p.get('likeCount')})")

                            if total_cnt == 0:
                                lines.append("- 검색 결과가 0건입니다.")
                                lines.append("💡 [자율 재검색 지침]: 결과가 0건이므로, 질문에서 불필요한 수식어를 덜어내거나 상위어/동의어로 검색어를 완화하여 1회 재검색(Query Expansion)할 수 있습니다. 이미 재검색했거나 마땅한 키워드가 없으면 검색 결과가 없음을 친절히 안내하세요.")
                            else:
                                lines.append("💡 [중요 지침]: 검색 결과가 충분히 확보되었습니다. 추가 도구 호출을 즉시 중단하고, 위 공지 제목과 제공된 링크(학교공지는 INTIP 내부 상세 경로인 /home/notice/{공지ID})를 표나 목록에 마크다운 링크([공지제목](링크)) 형태로 그대로 포함하여 최종 답변을 작성하세요. 단순 텍스트로만 제목을 적지 마십시오.")

                            summary_out = "\n".join(lines) + "\n"
                        else:
                            summary_out = f"\n[{tool.category} 실시간 조회 데이터 ({tool.name})]:\n{json.dumps(res, ensure_ascii=False)[:1000]}\n"
                    else:
                        tool_data = res
                        summary_out = f"\n[{tool.category} 실시간 조회 데이터 ({tool.name})]:\n{json.dumps(res, ensure_ascii=False)[:1000]}\n"
            except Exception as ex:
                logger.warning(f"Error executing tool {tool.name}: {ex}")
                status_state = "failed"
                status_title = f"{tool_display_name} 일시 조회 지연"
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
            status_title = f"{tool_display_name} 검색 완료"

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
                    status_state = "failed"
                    status_title = f"{tool_display_name} 검색 결과 없음"
                    summary_out = (
                        f"\n[시스템 학칙 지식베이스 검색 결과 없음]: 관련 학칙/규정 정보를 찾지 못했습니다.\n"
                        f"⚠️ 핵심 지침: 가상의 학칙 규정을 지어내지 말고, 지식베이스에서 해당 내용을 찾지 못했음을 사실대로 안내하세요.\n"
                    )
            except Exception as ex:
                logger.warning(f"Error executing INUChat tool {tool.name}: {ex}")
                status_state = "failed"
                status_title = f"{tool_display_name} 검색 일시 지연"
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


orchestrator = AgentOrchestrator()

