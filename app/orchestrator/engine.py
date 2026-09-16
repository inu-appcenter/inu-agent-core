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
from app.orchestrator.pruner import ToolPruner
from app.orchestrator.card_synthesizer import CardSynthesizer
from app.orchestrator.router import AgentRouter
from app.tools.registry import tool_registry


class AgentOrchestrator:
    async def run_stream(self, request: ChatRequest) -> AsyncGenerator[AgentStreamEvent, None]:
        """
        Execute streaming chat with intent handling, tool pruning, and SSE events.
        """
        # 0. Out-of-scope hard guardrail (Coding, pure math, unrelated universities)
        if is_out_of_scope_question(request.message):
            logger.info(f"Out-of-scope question detected: '{request.message[:30]}...' -> streaming refusal.")
            yield AgentStreamEvent(event_type="TOKEN", content=OUT_OF_SCOPE_REFUSAL_MESSAGE)
            yield AgentStreamEvent(event_type="DONE")
            return

        # 1. Tool Pruning: select top 3-4 most relevant tools from registry with multi-turn context
        all_tools = tool_registry.list_tools()
        pruned_tools = ToolPruner.prune(
            query=request.message,
            tools=all_tools,
            history=request.history,
            client=request.client,
            max_tools=4,
        )

        # Context for tool execution (Token Relay)
        exec_context = {
            "authorization": (request.client_context or {}).get("authorization", ""),
            "client": request.client,
        }

        tool_summary_text = ""
        emitted_cards = set()
        emitted_actions = set()

        # 2. Execute tools if matched and synthesize SDUI cards
        for tool in pruned_tools:
            # Case A: Client Action (LMS, Portal ERP) -> Check client_context or emit instruction
            if tool.category in ["LMS", "PORTAL"]:
                client_ctx = request.client_context or {}
                if tool.category == "PORTAL":
                    # Check academic and academicDisplay first, then portal
                    domain_data = client_ctx.get("academic") or client_ctx.get("academicDisplay") or client_ctx.get("portal")
                    if isinstance(domain_data, dict) and list(domain_data.keys()) == ["linked"]:
                        domain_data = client_ctx.get("academic") or client_ctx.get("academicDisplay")
                else:
                    domain_data = client_ctx.get("lms")

                if domain_data and isinstance(domain_data, (dict, list)):
                    # Actual client context provided (e.g. from mobile app SSO)
                    logger.info(f"Using provided client_context for domain {tool.category}")
                    if tool.category == "PORTAL" and isinstance(domain_data, dict):
                        dept = domain_data.get("departmentName") or domain_data.get("deptName") or ""
                        colg = domain_data.get("collegeName") or ""
                        status = domain_data.get("enrollmentStatus") or ""
                        sem = domain_data.get("completedSemesterCount") or ""
                        credits = domain_data.get("acquiredCredits") or ""
                        gpa = domain_data.get("gradeAverage") or ""
                        entry = domain_data.get("entryYear") or (domain_data.get("studentId", "")[:4] if domain_data.get("studentId") else "")
                        name = domain_data.get("koreanName") or "학우"

                        tool_summary_text += (
                            f"\n[포털 종합정보(ERP) 학생 실제 학적 연동 데이터]:\n"
                            f"- 성명/소속: {name}님 ({colg} {dept})\n"
                            f"- 학적 상태: {status}" + (f" (이수 학기: {sem})" if sem else "") + "\n"
                            f"- 취득 학점: {credits}학점 (평점 평균: {gpa})\n"
                            f"- 입학 정보: {entry}학번\n"
                            f"- [응답 지침]: 위 연동된 실제 학적 및 학점 데이터를 바탕으로 학생의 질문에 정확하고 친절하게 답변하세요.\n"
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

                                    due_str = ev.get("formattedtime")
                                    if not due_str:
                                        ts = ev.get("timesort") or ev.get("timestart")
                                        if isinstance(ts, (int, float)) and ts > 0:
                                            try:
                                                kst = timezone(timedelta(hours=9))
                                                due_str = datetime.fromtimestamp(ts, tz=kst).strftime("%m월 %d일(%a) %H:%M")
                                            except Exception:
                                                due_str = str(ts)
                                    due_str = due_str or "마감일시 미정"
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

                        lms_lines.append("- [응답 지침]: 위 연동된 실제 LMS 과제 및 수강 강좌 데이터를 바탕으로 학생에게 친절하고 명확하게 답변하세요.")
                        tool_summary_text += "\n".join(lms_lines) + "\n"
                    else:
                        tool_summary_text += f"\n[{tool.category} 실제 학생 연동 데이터]:\n{json.dumps(domain_data, ensure_ascii=False)[:1000]}\n"

                    if tool.category not in emitted_cards:
                        card = CardSynthesizer.synthesize_for_domain(
                            domain=tool.category,
                            tool_name=tool.name,
                            data=domain_data,
                            query=request.message,
                        )
                        if card:
                            logger.info(f"Emitting SDUI Card for {tool.category} from client context")
                            yield AgentStreamEvent(event_type="CARD", card=card)
                            emitted_cards.add(tool.category)
                else:
                    # No client context data provided -> emit action required (only once per domain) and add grounding instruction
                    if tool.category not in emitted_actions:
                        try:
                            action_instruction = await tool.execute({}, exec_context)
                            if hasattr(action_instruction, "action_id"):
                                logger.info(f"Emitting ClientActionInstruction: {action_instruction.action_id} ({action_instruction.auth_domain})")
                                yield AgentStreamEvent(event_type="ACTION_REQUIRED", action=action_instruction)
                                emitted_actions.add(tool.category)
                        except Exception as ex:
                            logger.warning(f"Failed to generate client action for tool {tool.name}: {ex}")

                    if f"[{tool.category}_STATUS]" not in tool_summary_text:
                        if tool.category == "PORTAL":
                            portal_meta = client_ctx.get("portal") or {}
                            err_code = portal_meta.get("academicErrorCode")
                            err_msg = portal_meta.get("academicErrorMessage")
                            if err_code == "NO_CREDENTIALS":
                                reason_guide = "현재 INTIP 앱의 [설정] > [포털 계정 연동]에 포털 계정이 등록되어 있지 않습니다. 앱 설정에서 1회 계정을 연동해 주셔야 실시간 학적 조회가 가능합니다."
                            elif err_code == "LOGIN_FAILED":
                                reason_guide = "포털 로그인 인증에 실패했습니다. 포털 비밀번호가 최근 변경되었는지 확인하거나, INTIP 앱 설정에서 포털 계정 정보를 재등록해 주세요."
                            elif err_code in ["ERP_ERROR", "NETWORK_ERROR"]:
                                reason_guide = f"학교 포털(ERP) 시스템 응답 지연 또는 세션 오류가 발생했습니다. ({err_msg or '잠시 후 다시 시도'}). 학교 포털(portal.inu.ac.kr)에 접속하여 비밀번호 변경 팝업이 뜨는지 확인해 주세요."
                            else:
                                reason_guide = "개인정보 보호(Zero-Knowledge) 원칙에 따라 학생의 포털 학적은 모바일 INTIP 앱의 보안 영역(KeyStore) 연동을 통해서만 안전하게 실시간 조회됩니다."

                            tool_summary_text += (
                                f"\n[PORTAL_STATUS] (포털 종합정보 데이터 연동 상태):\n"
                                f"- {reason_guide}\n"
                                f"- 현재 세션에는 연동된 실제 학생의 학적/학점 데이터가 없습니다.\n"
                                f"- [절대 금지]: 가상의 학과, 가상의 학점이나 성적을 절대로 지어내지 마십시오.\n"
                                f"- [응답 지침]: 위 원인과 연동 방법을 친절하고 정확하게 학생에게 안내하십시오.\n"
                            )
                        else:
                            domain_name_kr = "이러닝(LMS)"
                            tool_summary_text += (
                                f"\n[LMS_STATUS] (이러닝 데이터 연동 상태):\n"
                                f"- 개인정보 보호 및 보안(Zero-Knowledge) 원칙에 따라, 학생의 이러닝(LMS) 과제 및 수강 강좌는 사용자 기기(INTIP 앱) 연동을 통해서만 안전하게 실시간 조회됩니다.\n"
                                f"- 현재 세션에는 연동된 실제 학생의 이러닝 데이터가 없습니다.\n"
                                f"- [연동 방법 안내]: INTIP 앱의 [설정] > [이러닝(LMS) 계정 연동] 또는 [포털 계정 연동]을 통해 계정을 등록하시면 실시간 과제와 강의 일정을 즉시 조회할 수 있습니다.\n"
                                f"- [절대 금지]: 가상의 과목명이나 가상의 과제를 절대로 지어내지 마십시오.\n"
                                f"- [응답 지침]: 현재 연동된 이러닝 정보가 없으므로, 위 연동 경로를 안내하며 앱 내 연동이 필요함을 친절히 안내하십시오.\n"
                            )

            # Case B: Server OpenAPI / Direct Tools (Cafeteria, Bus, Timetable, Notices, Schedule, Directory, Weather, Library)
            elif tool.category in ["CAFETERIA", "BUS", "TIMETABLE", "NOTICE", "SCHEDULE", "DIRECTORY", "WEATHER", "LIBRARY"]:
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
                            tab_name = bus_meta.get("tabName", "인입런")
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
                            logger.info(f"Bus arrivals filtered by INTIP routes ({valid_routes}): {len(filtered_arrivals)}/{len(raw_list)} items")
                            if filtered_arrivals:
                                tool_summary_text += (
                                    f"\n[BUS 실시간 도착 정보 ({tab_name} - {stop_name})]:\n"
                                    f"- 인팁(INTIP) 서비스 대상 모니터링 노선: {', '.join(valid_routes)}\n"
                                    f"- 실시간 도착 예정 버스:\n{json.dumps(filtered_arrivals, ensure_ascii=False)}\n"
                                )
                            else:
                                tool_summary_text += (
                                    f"\n[BUS 실시간 도착 정보 ({tab_name} - {stop_name})]:\n"
                                    f"- 인팁(INTIP) 서비스 대상 모니터링 노선: {', '.join(valid_routes)}\n"
                                    f"- 현재 해당 정류소에 운행 대기 중이거나 도착 예정인 인팁 서비스 대상 버스({', '.join(valid_routes)})가 없습니다.\n"
                                    f"- [엄격 지침]: 인팁 프론트엔드에서 공식 서비스하지 않는 일반 시내/광역 버스(예: M6464 등)는 절대로 답변에 언급하지 마십시오. 서비스 대상 노선({', '.join(valid_routes)}) 중 현재 도착 정보가 없음을 사실대로 친절히 안내하세요.\n"
                                )
                        elif tool.category == "LIBRARY" and isinstance(res, dict):
                            tool_data = res
                            mode = res.get("mode", "SEATS")
                            if mode == "RESERVE_SEAT":
                                seat_info = res.get("data", {})
                                r_name = seat_info.get("roomName", "열람실")
                                s_no = f"{seat_info.get('seatNo')}번 " if seat_info.get("seatNo") else ""
                                tool_summary_text += (
                                    f"\n[학산도서관 열람실 좌석 배정 신청]:\n"
                                    f"- 대상: {r_name} {s_no}좌석\n"
                                    f"- 대화창에 [학산도서관 좌석 배정 확인 카드]가 준비되었습니다.\n"
                                    f"- [핵심 응답 지침]: 사용자에게 대화창 아래 카드에서 좌석 상태를 확인하신 후 [확인 및 배정 신청하기] 버튼을 누르면 기기(INTIP 앱)에서 도서관 시스템에 즉시 배정을 완료한다고 친절하게 안내하세요.\n"
                                    f"- [절대 금지]: '예약은 학생이 웹사이트나 키오스크에서 직접 하라'고 안내하지 마십시오. 대화창 카드의 버튼 클릭으로 앱이 즉시 배정을 완료합니다.\n"
                                )
                            elif mode == "RESERVE_STUDY_ROOM":
                                study_info = res.get("data", {})
                                tool_summary_text += (
                                    f"\n[학산도서관 스터디룸 예약 신청]:\n"
                                    f"- 대상: {study_info.get('roomName', '스터디룸')}\n"
                                    f"- 대화창에 [학산도서관 스터디룸 예약 확인 카드]가 준비되었습니다.\n"
                                    f"- [핵심 응답 지침]: 사용자에게 아래 카드에서 예약 정보를 확인하고 [확인 및 예약 신청하기] 버튼을 누르면 기기에서 즉시 예약이 완료된다고 안내하세요.\n"
                                )
                            elif mode == "STUDY_ROOMS":
                                tool_summary_text += (
                                    f"\n[학산도서관 스터디룸 목록]:\n"
                                    f"- 대화창에 스터디룸 목록 카드가 준비되었습니다.\n"
                                    f"- [응답 지침]: 원하는 스터디룸을 터치하면 바로 예약 신청이 가능함을 안내하세요.\n"
                                )
                            else:
                                rooms_info = "\n".join([
                                    f"- {r['name']}: 잔여 {r.get('available_seats', r.get('seats', {}).get('available', 0))}석 / 전체 {r.get('total_seats', r.get('seats', {}).get('total', 0))}석"
                                    for r in res.get("rooms", [])
                                ])
                                tool_summary_text += (
                                    f"\n[학산도서관 열람실 실시간 잔여 좌석 현황]:\n"
                                    f"{rooms_info}\n"
                                    f"- [핵심 응답 지침]: 위 실시간 잔여 좌석을 친절하게 브리핑하고, 대화창 아래 카드에서 원하는 열람실을 터치하면 바로 좌석 선택 및 배정 신청 화면으로 연결된다고 안내하세요.\n"
                                    f"- [절대 금지]: '예약은 웹사이트나 키오스크에서 직접 하라'고 하지 마십시오. 대화창 카드를 터치하여 바로 좌석을 배정받을 수 있습니다.\n"
                                )
                        else:
                            tool_data = res
                            logger.info(f"Successfully executed tool: {tool.name} with dynamic args {tool_args}")
                            tool_summary_text += f"\n[{tool.category} 실시간 조회 데이터 ({tool.name})]:\n{json.dumps(res, ensure_ascii=False)[:1000]}\n"
                    else:
                        logger.warning(f"OpenApiTool {tool.name} returned error or empty: {res}")
                except Exception as ex:
                    logger.warning(f"Error executing OpenAPI tool {tool.name}: {ex}")

                # Synthesize and emit SDUI card for domain
                if tool.category not in emitted_cards:
                    card = CardSynthesizer.synthesize_for_domain(
                        domain=tool.category,
                        tool_name=tool.name,
                        data=tool_data,
                        query=request.message,
                    )
                    if card:
                        logger.info(f"Emitting SDUI Card for domain: {tool.category}")
                        yield AgentStreamEvent(event_type="CARD", card=card)
                        emitted_cards.add(tool.category)

            # Case C: INUChat Official Knowledge RAG Tool (Academic Regulations, Graduation, Policies)
            elif tool.category == "INU_AI_KNOWLEDGE":
                try:
                    res = await tool.execute({"question": request.message}, exec_context)
                    if isinstance(res, dict) and res.get("success") and res.get("data"):
                        rag_data = res.get("data")
                        rag_answer = rag_data.get("rag_answer", "")
                        logger.info(f"Successfully retrieved INUChat RAG Knowledge (len: {len(rag_answer)})")
                        tool_summary_text += f"\n[인천대학교 공식 학칙/규정 지식베이스 (INUChat RAG 검색 결과)]:\n{rag_answer}\n"

                        if "INU_AI_KNOWLEDGE" not in emitted_cards:
                            card = CardSynthesizer.synthesize_for_domain(
                                domain="INU_AI_KNOWLEDGE",
                                tool_name=tool.name,
                                data=rag_data,
                                query=request.message,
                            )
                            if card:
                                logger.info("Emitting SDUI Card for INUChat Citations")
                                yield AgentStreamEvent(event_type="CARD", card=card)
                                emitted_cards.add("INU_AI_KNOWLEDGE")
                except Exception as ex:
                    logger.warning(f"Error executing INUChat tool {tool.name}: {ex}")

        # 3. Build System Prompt with Grounding Data
        system_prompt = get_system_prompt_for_client(request.client, tool_summary=tool_summary_text)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

        # Append conversation history
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})

        # Append current user prompt
        messages.append({"role": "user", "content": request.message})

        try:
            logger.info(
                f"Starting stream for '{request.message[:30]}...' (Tools executed: {len(pruned_tools)}, HasData: {bool(tool_summary_text)})"
            )

            async for token in llm_client.stream_chat(
                messages=messages,
            ):
                yield AgentStreamEvent(event_type="TOKEN", content=token)

            yield AgentStreamEvent(event_type="DONE")
        except Exception as e:
            logger.error(f"Error in orchestrator stream: {e}", exc_info=True)
            yield AgentStreamEvent(event_type="ERROR", error=str(e))


orchestrator = AgentOrchestrator()
