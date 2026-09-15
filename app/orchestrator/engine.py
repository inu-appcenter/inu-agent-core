"""
Core Agent Orchestration Engine
Handles Tool Pruning (Gemma 27B optimization), ReAct looping, and SSE streaming.
"""
from typing import AsyncGenerator, List, Dict, Any, Optional
import json

from app.core.logging import logger
from app.llm.client import llm_client
from app.llm.schemas import ChatRequest, AgentStreamEvent, GenerativeCard
from app.orchestrator.prompts import get_system_prompt_for_client
from app.tools.base import BaseTool


class AgentOrchestrator:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register_tool(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} (Category: {tool.category})")

    def prune_tools(self, user_query: str, client: str) -> List[Dict[str, Any]]:
        """
        Tool Pruning for Gemma 27B:
        Instead of dumping 50 tools into the prompt, filter to 3-5 most relevant tools
        based on keyword heuristics and client context.
        """
        q = user_query.lower()
        selected_tools: List[BaseTool] = []

        # 1. Intent keyword heuristics
        if any(k in q for k in ["학식", "밥", "메뉴", "식당", "점심", "저녁", "아침"]):
            selected_tools.extend([t for t in self._tools.values() if "cafeteria" in t.name or "menu" in t.name])

        if any(k in q for k in ["버스", "셔틀", "인입런", "정류장", "노선"]):
            selected_tools.extend([t for t in self._tools.values() if "bus" in t.name])

        if any(k in q for k in ["시간표", "강의", "수업", "강좌", "교수"]):
            selected_tools.extend([t for t in self._tools.values() if "timetable" in t.name or "course" in t.name])

        if any(k in q for k in ["과제", "lms", "사캠", "사이버캠퍼스", "제출"]):
            selected_tools.extend([t for t in self._tools.values() if t.category == "LMS"])

        if any(k in q for k in ["도서관", "열람실", "좌석", "자리", "연장"]):
            selected_tools.extend([t for t in self._tools.values() if t.category == "LIBRARY"])

        if any(k in q for k in ["성적", "학점", "학적", "장학", "휴학", "졸업", "포털"]):
            selected_tools.extend([t for t in self._tools.values() if t.category == "PORTAL"])

        if any(k in q for k in ["외박", "기숙사", "생활관", "상벌점", "세탁기", "돔"]):
            selected_tools.extend([t for t in self._tools.values() if t.category == "DORM"])

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for t in selected_tools:
            if t.name not in seen:
                seen.add(t.name)
                deduped.append(t)

        # Fallback: if no specific keywords matched, provide top general tools
        if not deduped:
            deduped = list(self._tools.values())[:4]

        # Limit to max 5 tools for Gemma 27B attention retention
        final_tools = deduped[:5]
        return [t.get_schema() for t in final_tools]

    async def run_stream(self, request: ChatRequest) -> AsyncGenerator[AgentStreamEvent, None]:
        """
        Execute streaming chat with intent handling and SSE events.
        """
        system_prompt = get_system_prompt_for_client(request.client)
        pruned_tools = self.prune_tools(request.message, request.client)

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]

        # Append conversation history
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})

        # Append current user prompt
        messages.append({"role": "user", "content": request.message})

        try:
            async for token in llm_client.stream_chat(messages=messages, tools=pruned_tools if pruned_tools else None):
                yield AgentStreamEvent(event_type="TOKEN", content=token)

            yield AgentStreamEvent(event_type="DONE")
        except Exception as e:
            logger.error(f"Error in orchestrator stream: {e}", exc_info=True)
            yield AgentStreamEvent(event_type="ERROR", error=str(e))


orchestrator = AgentOrchestrator()
