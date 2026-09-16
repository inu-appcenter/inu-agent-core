"""
Core Agent Orchestration Engine
Integrates Tool Registry, Tool Pruner (Gemma 27B optimization), and ReAct execution with SSE streaming.
"""
from typing import AsyncGenerator, List, Dict, Any, Optional
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
                domain_key = tool.category.lower()  # "lms", "portal"
                domain_data = client_ctx.get(domain_key)

                if domain_data and isinstance(domain_data, (dict, list)):
                    # Actual client context provided (e.g. from mobile app SSO)
                    logger.info(f"Using provided client_context for domain {tool.category}")
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
                        domain_name_kr = "이러닝(LMS)" if tool.category == "LMS" else "포털 종합정보"
                        item_kr = "과제, 수강 강좌, 출석" if tool.category == "LMS" else "성적, 취득 학점, 학적"
                        tool_summary_text += (
                            f"\n[{tool.category}_STATUS] ({domain_name_kr} 데이터 연동 상태):\n"
                            f"- 개인정보 보호 및 보안(Zero-Knowledge) 원칙에 따라, {domain_name_kr}의 개인 {item_kr} 등은 사용자 기기(INTIP 앱) 연동을 통해서만 안전하게 실시간 조회됩니다.\n"
                            f"- 현재 세션에는 연동된 실제 학생의 {domain_name_kr} 데이터가 없습니다.\n"
                            f"- [절대 금지]: 가상의 과목명이나 가상의 {item_kr}를 절대로 지어내거나 임의의 표로 작성하지 마십시오.\n"
                            f"- [응답 지침]: 현재 연동된 {domain_name_kr} 정보가 없으므로, 실시간 확인을 위해 {domain_name_kr} 계정 연동(INTIP 앱 지원)이 필요함을 친절히 안내하십시오.\n"
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
                        elif tool.category == "LIBRARY" and isinstance(res, dict) and "rooms" in res:
                            tool_data = res
                            rooms_info = "\n".join([
                                f"- {r['name']}: 잔여 {r['available_seats']}석 / 전체 {r['total_seats']}석 (사용 중: {r['occupied_seats']}석, 이용률 {r['utilization_rate']})"
                                for r in res.get("rooms", [])
                            ])
                            tool_summary_text += (
                                f"\n[인천대학교 학술정보관(도서관) 열람실 실시간 좌석 현황]:\n"
                                f"{rooms_info}\n"
                                f"- [응답 지침]: 위 실시간 잔여 좌석 데이터를 바탕으로 사용자에게 열람실별 현재 잔여 좌석 수와 여유/혼잡 상태를 친절하고 정확하게 안내하세요.\n"
                                f"- [금지 사항]: INTIP 앱 내에 존재하지 않는 가상의 '도서관 메뉴'나 '예약 버튼'을 누르라고 거짓 안내하지 마십시오. 열람실 좌석 배정 및 스터디룸 예약은 학술정보관 공식 모바일 웹(https://lib.inu.ac.kr) 또는 학술정보관 현장 키오스크를 통해 진행할 수 있음을 안내하세요.\n"
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
