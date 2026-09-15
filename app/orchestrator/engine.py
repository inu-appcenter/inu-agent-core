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

        # 2. Execute tools if matched and synthesize SDUI cards
        for tool in pruned_tools:
            # Case A: Client Action (LMS, Portal ERP, Library P2P) -> Emit instruction
            if tool.category in ["LMS", "PORTAL", "LIBRARY"]:
                try:
                    action_instruction = await tool.execute({}, exec_context)
                    if hasattr(action_instruction, "action_id"):
                        logger.info(f"Emitting ClientActionInstruction: {action_instruction.action_id} ({action_instruction.auth_domain})")
                        yield AgentStreamEvent(event_type="ACTION_REQUIRED", action=action_instruction)
                except Exception as ex:
                    logger.warning(f"Failed to generate client action for tool {tool.name}: {ex}")

            # Case B: Server OpenAPI Tool (Cafeteria, Bus, Timetable, Notices)
            elif tool.category in ["CAFETERIA", "BUS", "TIMETABLE", "NOTICE", "SCHEDULE"]:
                tool_data = None
                try:
                    res = await tool.execute({}, exec_context)
                    if isinstance(res, dict) and res.get("success") and res.get("data"):
                        tool_data = res.get("data")
                        logger.info(f"Successfully executed OpenApiTool: {tool.name}")
                        tool_summary_text += f"\n[{tool.name} 조회 결과]:\n{json.dumps(res.get('data'), ensure_ascii=False)[:1000]}\n"
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
