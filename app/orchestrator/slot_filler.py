"""
Interactive Slot-Filling Finite State Machine (FSM)
Detects missing mandatory parameters for action/mutation tools, pre-fetches context,
and orchestrates structured user clarifications with quick-selection chips.
"""
from typing import Dict, Any, List, Optional, Tuple
import json

from app.core.logging import logger
from app.tools.base import BaseTool
from app.tools.registry import tool_registry
from app.llm.schemas import AgentStreamEvent, ComponentCard, CardLink


# Slot definitions and human-friendly prompt metadata for action domains
ACTION_SLOT_METADATA = {
    "LIBRARY_STUDY_ROOM": {
        "required_slots": ["roomId", "startTime"],
        "labels": {
            "roomId": "스터디룸 방 번호",
            "startTime": "이용 시작 시간",
            "duration": "이용 시간",
        },
        "prefetch_tool": "api_getAvailableStudyRooms",
        "question_template": "학산도서관 스터디룸 예약을 위해 이용하실 방 번호와 시작 시간을 알려주세요.",
    },
    "REMINDER": {
        "required_slots": ["time", "topic"],
        "labels": {
            "time": "알림 수신 시각",
            "topic": "알림 주제 (학식, 날씨, 공지 등)",
        },
        "question_template": "알림을 설정할 시간과 내용을 알려주세요.",
    },
    "COUNSELING": {
        "required_slots": ["category", "preferredDate"],
        "labels": {
            "category": "상담 분야 (학업, 진로, 심리 등)",
            "preferredDate": "희망 상담 날짜",
        },
        "question_template": "원하시는 상담 분야와 희망하시는 날짜를 알려주세요.",
    },
}


class SlotFillingValidator:
    """
    Validates tool execution arguments against JSON Schema 'required' fields
    and detects missing conversational slots.
    """

    @classmethod
    def check_missing_slots(
        cls, tool: BaseTool, args: Dict[str, Any]
    ) -> List[str]:
        """
        Inspects the tool's inputSchema or action metadata to find missing mandatory slots.
        """
        schema = getattr(tool, "input_schema", None) or getattr(tool, "parameters_schema", None) or {}
        required_fields = schema.get("required", []) if isinstance(schema, dict) else []

        missing = []
        for field in required_fields:
            val = args.get(field)
            if val is None or str(val).strip() == "" or str(val).strip().upper() in ["NONE", "NULL", "UNDEFINED"]:
                missing.append(field)

        return missing

    @classmethod
    def should_intercept_action(
        cls, tool_name: str, tool_category: str, args: Dict[str, Any]
    ) -> Tuple[bool, List[str], str]:
        """
        Determines whether a tool invocation must be halted for interactive clarification.
        Returns: (needs_interception, missing_slots, clarification_prompt)
        """
        # 1. Check against registered Action metadata (only for reservation/creation actions)
        t_name_lower = tool_name.lower()
        t_cat_upper = tool_category.upper()

        if ("study_room" in t_name_lower or "studyroom" in t_name_lower) and any(act in t_name_lower for act in ["reserve", "book", "apply"]):
            meta = ACTION_SLOT_METADATA["LIBRARY_STUDY_ROOM"]
            missing = [s for s in meta["required_slots"] if not args.get(s)]
            if missing:
                return True, missing, meta["question_template"]

        if t_cat_upper == "REMINDER" or "reminder" in t_name_lower:
            action_type = str(args.get("action") or "").upper()
            if action_type == "CREATE":
                missing = []
                if not args.get("targetTime"):
                    missing.append("targetTime")
                if not args.get("targetTool"):
                    missing.append("targetTool")
                if missing:
                    return True, missing, ACTION_SLOT_METADATA["REMINDER"]["question_template"]

        # 2. Check general action/mutation tools (Client Actions, Reservations, Post/Delete mutations)
        # Read-only query tools (OpenAPI search/list/get) are handled via normal execution / schema coercion
        target_val = str(args.get("target") or "").upper()
        is_library_mutation = (
            tool_category == "LIBRARY" and any(k in target_val for k in ["RESERVE", "CANCEL", "CHECKIN", "RENEW", "RETURN"])
        )
        is_action_mutation = (
            is_library_mutation
            or tool_category in ["REMINDER", "CAMPUS_WATCH", "DAILY_BRIEF", "KEYWORD"]
            or any(kw in t_name_lower for kw in ["reserve", "create", "delete", "toggle", "apply", "book"])
        )
        if is_action_mutation:
            tool_inst = tool_registry.get_tool(tool_name)
            if tool_inst:
                missing = cls.check_missing_slots(tool_inst, args)
                if missing:
                    korean_names = ", ".join(missing)
                    prompt = f"요청을 처리하기 위해 추가 정보({korean_names})가 필요합니다. 확인 후 알려주시겠어요?"
                    return True, missing, prompt

        return False, [], ""

    @classmethod
    async def prefetch_options_for_clarification(
        cls, tool_name: str, exec_context: Dict[str, Any]
    ) -> Tuple[List[str], Optional[ComponentCard]]:
        """
        Pre-fetches live available options (e.g. available library rooms) to provide dynamic suggestion chips.
        """
        chips: List[str] = []
        card = None

        if "library" in tool_name.lower() or "study_room" in tool_name.lower():
            lib_tool = tool_registry.get_tool("api_getAvailableStudyRooms") or tool_registry.get_tool("library_study_rooms")
            if lib_tool:
                try:
                    res = await lib_tool.execute({}, exec_context)
                    if isinstance(res, dict):
                        rooms = res.get("rooms") or res.get("data", {}).get("rooms", [])
                        for r in rooms[:3]:
                            name = r.get("name", "스터디룸")
                            chips.append(f"{name} 14:00 예약")
                except Exception as e:
                    logger.debug(f"SlotFiller prefetch failed: {e}")

        return chips, card
