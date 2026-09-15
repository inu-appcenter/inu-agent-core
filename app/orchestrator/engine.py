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
            # Case A: Client Action (LMS, Portal ERP, Library P2P) -> Check client_context or emit instruction
            if tool.category in ["LMS", "PORTAL", "LIBRARY"]:
                client_ctx = request.client_context or {}
                domain_key = tool.category.lower()  # "lms", "portal", "library"
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
                        domain_name_kr = "사이버캠퍼스(LMS)" if tool.category == "LMS" else ("도서관" if tool.category == "LIBRARY" else "포털 종합정보")
                        tool_summary_text += (
                            f"\n[{tool.category}_STATUS] ({domain_name_kr} 데이터 연동 상태):\n"
                            f"- 개인정보 보호 및 보안(Zero-Knowledge) 원칙에 따라, {domain_name_kr}의 개인 과제, 수강 강좌, 출석, 성적 등은 사용자 기기(INTIP 앱) 연동을 통해서만 안전하게 실시간 조회됩니다.\n"
                            f"- 현재 세션에는 연동된 실제 학생의 {domain_name_kr} 데이터가 없습니다.\n"
                            f"- [절대 금지]: 가상의 과목명(경영학원론, 데이터구조 등), 가상의 과제명, 가상의 마감 일자를 절대로 지어내거나 임의의 표로 작성하지 마십시오.\n"
                            f"- [응답 지침]: 현재 연동된 과제/강좌 정보가 없으므로, 실시간 과제 마감과 강의 진도를 확인하려면 사이버캠퍼스(LMS) 계정 연동(INTIP 앱 지원)이 필요함을 친절히 안내하십시오.\n"
                        )

            # Case B: Server OpenAPI Tool (Cafeteria, Bus, Timetable, Notices, Schedule, Directory, Weather)
            elif tool.category in ["CAFETERIA", "BUS", "TIMETABLE", "NOTICE", "SCHEDULE", "DIRECTORY", "WEATHER"]:
                tool_data = None
                tool_args = await AgentRouter.extract_tool_arguments(
                    tool=tool,
                    query=request.message,
                    history=request.history,
                )
                try:
                    res = await tool.execute(tool_args, exec_context)
                    if res is not None and not (isinstance(res, dict) and "error" in res):
                        tool_data = res
                        logger.info(f"Successfully executed OpenApiTool: {tool.name} with dynamic args {tool_args}")
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
