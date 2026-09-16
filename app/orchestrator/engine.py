"""
Core Agent Orchestration Engine
Integrates Tool Registry, Tool Pruner (Gemma 27B optimization), and ReAct execution with SSE streaming.
"""
from typing import AsyncGenerator, List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import json

from app.core.logging import logger
from app.llm.client import llm_client
from app.llm.schemas import ChatRequest, AgentStreamEvent
from app.orchestrator.prompts import (
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
    }
    if category in category_map:
        return category_map[category]

    name_lower = name.lower()
    if "watch" in name_lower or "sniper" in name_lower:
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
        Execute streaming chat with intent handling, autonomous ReAct chaining,
        real LLM reasoning thoughts, and INUChat direct pass-through.
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
            top_k=6,
        )

        client_ctx = request.client_context or {}
        raw_token = client_ctx.get("auth") or client_ctx.get("authorization", "")
        clean_token = raw_token.replace("Bearer ", "").strip() if raw_token else ""

        exec_context = {
            "auth": clean_token,
            "authorization": f"Bearer {clean_token}" if clean_token else "",
            "client": request.client,
            "query": request.message,
        }

        # 2. Step 1: Real LLM Reasoning & 1st-hop Plan
        initial_plan = await AgentRouter.decide_initial_plan(
            query=request.message,
            history=request.history,
            candidate_tools=candidate_tools,
        )

        initial_thought = initial_plan.get("thought") or f"'{request.message[:20]}...' 질문을 해결하기 위해 필요한 캠퍼스 시스템을 확인하고 있습니다."
        yield AgentStreamEvent(
            event_type="THINKING",
            thinking=initial_thought,
        )

        raw_initial_tools = initial_plan.get("tools", [])
        first_hop_categories = []
        for item in raw_initial_tools:
            if isinstance(item, str):
                first_hop_categories.append(item.upper())
            elif isinstance(item, dict):
                val = item.get("name") or item.get("category") or item.get("tool") or ""
                if val:
                    first_hop_categories.append(str(val).upper())

        # Filter 1st-hop tools to execute
        first_tools = []
        for cat in first_hop_categories:
            matching = [t for t in candidate_tools if t.category.upper() == cat]
            if matching:
                first_tools.extend(matching)
            else:
                cat_tools = tool_registry.get_tools_by_category(cat)
                if cat_tools:
                    first_tools.extend(cat_tools)
                else:
                    reg_tool = tool_registry.get_tool(cat.lower())
                    if reg_tool:
                        first_tools.append(reg_tool)

        # Fallback if no matching tools found
        if not first_tools and candidate_tools:
            first_tools = candidate_tools[:2]

        executed_categories = set()
        executed_tool_names = set()
        emitted_cards = set()
        emitted_actions = set()
        tool_summary_text = ""
        academic_data_dict = {}
        inuchat_rag_data = None

        # Execute 1st-hop tools
        for tool in first_tools:
            async for event, summary, ac_data, rag_data in self._execute_tool(
                tool=tool,
                request=request,
                exec_context=exec_context,
                emitted_cards=emitted_cards,
                emitted_actions=emitted_actions,
            ):
                if event:
                    yield event
                if summary:
                    tool_summary_text += summary
                if ac_data:
                    academic_data_dict.update(ac_data)
                if rag_data:
                    inuchat_rag_data = rag_data

            executed_categories.add(tool.category.upper())
            executed_tool_names.add(tool.name)

        # 3. Step 2: ReAct Autonomous Chaining (Sufficiency Check)
        sec_plan = await AgentRouter.decide_secondary_plan(
            query=request.message,
            history=request.history,
            current_observation=tool_summary_text,
            executed_categories=list(executed_categories),
            available_tools=candidate_tools,
        )

        sec_thought = sec_plan.get("thought", "").strip()
        raw_sec_tools = sec_plan.get("tools", [])
        sec_categories = []
        for item in raw_sec_tools:
            name_val = item if isinstance(item, str) else (item.get("name") or item.get("category") or item.get("tool") or "")
            if name_val and str(name_val).upper() not in executed_categories:
                sec_categories.append(str(name_val).upper())

        if sec_categories:
            if sec_thought:
                yield AgentStreamEvent(
                    event_type="THINKING",
                    thinking=sec_thought,
                )

            # Execute 2nd-hop tools
            second_tools = []
            for cat in sec_categories:
                matching = [t for t in candidate_tools if t.category.upper() == cat]
                if matching:
                    second_tools.extend(matching)
                else:
                    for t in tool_registry.list_tools():
                        if t.category.upper() == cat and t.name not in executed_tool_names:
                            second_tools.append(t)
                            break

            for tool in second_tools:
                async for event, summary, ac_data, rag_data in self._execute_tool(
                    tool=tool,
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
                    if ac_data:
                        academic_data_dict.update(ac_data)
                    if rag_data:
                        inuchat_rag_data = rag_data

                executed_categories.add(tool.category.upper())
                executed_tool_names.add(tool.name)
        elif sec_thought:
            yield AgentStreamEvent(
                event_type="THINKING",
                thinking=sec_thought,
            )

        # 4. Step 3: Response Streaming Strategy
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

            # Build enriched question with anonymized academic context
            inu_question = request.message
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
                    inu_question = f"{request.message}\n\n[비식별 학적 참고정보: {', '.join(ac_parts)}]"

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

        system_prompt = get_system_prompt_for_client(request.client, tool_summary=tool_summary_text) + additional_guideline

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": request.message})

        try:
            logger.info(
                f"Starting synthesis stream for '{request.message[:30]}...' (Categories: {executed_categories})"
            )

            async for token in llm_client.stream_chat(messages=messages):
                yield AgentStreamEvent(event_type="TOKEN", content=token)

            yield AgentStreamEvent(event_type="DONE")
        except Exception as e:
            logger.error(f"Error in orchestrator stream: {e}", exc_info=True)
            yield AgentStreamEvent(event_type="ERROR", error=str(e))

    async def _execute_tool(
        self,
        tool: BaseTool,
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
                domain_data = client_ctx.get("academic") or client_ctx.get("academicDisplay")
            else:
                domain_data = client_ctx.get("lms")

            if domain_data and isinstance(domain_data, (dict, list)):
                if tool.category == "PORTAL" and isinstance(domain_data, dict):
                    display_data = client_ctx.get("academicDisplay")
                    merged_portal = {**domain_data, **(display_data if isinstance(display_data, dict) else {})}

                    dept = merged_portal.get("departmentName") or merged_portal.get("deptName") or ""
                    colg = merged_portal.get("collegeName") or ""
                    status = merged_portal.get("enrollmentStatus") or ""
                    change = merged_portal.get("latestEnrollmentChange") or ""
                    sem = merged_portal.get("completedSemesterCount") or ""
                    credits = merged_portal.get("acquiredCredits") or ""
                    gpa = merged_portal.get("gradeAverage") or ""
                    entry = merged_portal.get("entryYear") or (merged_portal.get("studentId", "")[:4] if merged_portal.get("studentId") else "")
                    name = merged_portal.get("koreanName") or "학우"
                    advisor = merged_portal.get("advisorProfessorName") or merged_portal.get("profNm") or ""

                    status_display = status
                    if change and change != status:
                        status_display = f"{status} ({change})"

                    ac_data_out = {
                        "name": name,
                        "departmentName": dept,
                        "collegeName": colg,
                        "enrollmentStatus": status_display,
                        "completedSemesterCount": sem,
                        "acquiredCredits": credits,
                        "gradeAverage": gpa,
                        "entryYear": entry,
                        "advisor": advisor,
                    }

                    summary_out = (
                        f"\n[포털 종합정보(ERP) 학생 실제 학적 연동 데이터]:\n"
                        f"- 성명/소속: {name}님 ({colg} {dept})\n"
                        f"- 학적 상태: {status_display}" + (f" (이수 학기: {sem})" if sem else "") + "\n"
                        f"- 취득 학점: {credits}학점 (평점 평균: {gpa})\n"
                        f"- 입학 정보: {entry}학번\n"
                        + (f"- 지도교수: {advisor} 교수님\n" if advisor else "")
                        + f"- [응답 지침]: 위 연동된 실제 학적 및 학점 데이터를 바탕으로 학생의 질문에 정확하고 친절하게 답변하세요.\n"
                    )
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
            if tool.category == "DIRECTORY":
                query_intent_keywords = ["전화", "연락처", "번호", "과사", "사무실", "교수님", "교수", "위치", "호실", "문의", "연구실"]
                has_contact_intent = any(k in request.message.lower() for k in query_intent_keywords)
                if not has_contact_intent:
                    return
            elif tool.category == "NOTICE":
                notice_intent_keywords = ["공지", "소식", "모집", "안내문", "선발", "대회", "신청"]
                has_notice_intent = any(k in request.message.lower() for k in notice_intent_keywords)
                if not has_notice_intent:
                    return

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
            tool_args = await AgentRouter.extract_tool_arguments(
                tool=tool,
                query=request.message,
                history=request.history,
            )

            bus_meta = tool_args.pop("_meta", None) if tool.category == "BUS" else None

            try:
                res = await tool.execute(tool_args, exec_context)
                if res is not None and not (isinstance(res, dict) and "error" in res):
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
                        summary_out = (
                            f"\n[BUS 인천시 실시간 시내버스 도착 정보 ({stop_name})]:\n"
                            f"- 모니터링 노선: {', '.join(valid_routes)}\n"
                            f"- 실시간 도착 정보:\n{json.dumps(filtered_arrivals, ensure_ascii=False)}\n"
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
                    else:
                        tool_data = res
                        summary_out = f"\n[{tool.category} 실시간 조회 데이터 ({tool.name})]:\n{json.dumps(res, ensure_ascii=False)[:1000]}\n"
            except Exception as ex:
                logger.warning(f"Error executing tool {tool.name}: {ex}")

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
                    status_title=f"{tool_display_name} 확인 완료",
                    status_category=tool.category,
                    status_state="completed",
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

            inu_q = request.message
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
                    inu_q = f"{request.message}\n\n[비식별 학적 참고정보: {', '.join(ac_parts)}]"

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
            except Exception as ex:
                logger.warning(f"Error executing INUChat tool {tool.name}: {ex}")

            yield (
                AgentStreamEvent(
                    event_type="STATUS",
                    status_id=tool_id,
                    status_title=f"{tool_display_name} 검색 완료",
                    status_category="INU_AI_KNOWLEDGE",
                    status_state="completed",
                ),
                summary_out,
                None,
                rag_data_out,
            )


orchestrator = AgentOrchestrator()

