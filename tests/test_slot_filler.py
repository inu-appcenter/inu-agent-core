"""
Unit tests for Interactive Slot-Filling FSM (SlotFillingValidator)
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.orchestrator.slot_filler import SlotFillingValidator, ACTION_SLOT_METADATA
from app.tools.registry import tool_registry


def test_should_intercept_library_study_room_missing_slots():
    """Verify interception when study room reservation lacks roomId or startTime."""
    args = {"date": "2026-10-05"}  # missing roomId & startTime
    needs_intercept, missing, prompt = SlotFillingValidator.should_intercept_action(
        tool_name="action_reserve_study_room",
        tool_category="LIBRARY",
        args=args
    )
    assert needs_intercept is True
    assert "roomId" in missing
    assert "startTime" in missing
    assert "스터디룸" in prompt


def test_should_not_intercept_when_slots_fully_provided():
    """Verify no interception when all required parameters are given."""
    args = {"roomId": 105, "startTime": "14:00", "duration": 2}
    needs_intercept, missing, _ = SlotFillingValidator.should_intercept_action(
        tool_name="action_reserve_study_room",
        tool_category="LIBRARY",
        args=args
    )
    assert needs_intercept is False
    assert len(missing) == 0


def test_should_intercept_reminder_creation_missing_slots():
    """Verify interception when reminder creation lacks targetTime or targetTool."""
    args = {"action": "CREATE"}  # missing targetTime & targetTool
    needs_intercept, missing, prompt = SlotFillingValidator.should_intercept_action(
        tool_name="action_manage_reminder",
        tool_category="REMINDER",
        args=args
    )
    assert needs_intercept is True
    assert "targetTime" in missing
    assert "targetTool" in missing
    assert "알림" in prompt


def test_should_not_intercept_reminder_listing():
    """Verify that read-only/listing actions on reminders are never intercepted."""
    args = {"action": "LIST"}
    needs_intercept, missing, _ = SlotFillingValidator.should_intercept_action(
        tool_name="action_manage_reminder",
        tool_category="REMINDER",
        args=args
    )
    assert needs_intercept is False
    assert len(missing) == 0


@pytest.mark.asyncio
async def test_prefetch_options_for_clarification():
    """Verify live option prefetching generates helpful candidate chips."""
    mock_lib_tool = MagicMock()
    mock_lib_tool.execute = AsyncMock(return_value={
        "rooms": [
            {"id": 1, "name": "제1스터디룸(4인실)"},
            {"id": 2, "name": "제2스터디룸(6인실)"},
        ]
    })

    with patch.object(tool_registry, "get_tool", return_value=mock_lib_tool):
        chips, card = await SlotFillingValidator.prefetch_options_for_clarification(
            tool_name="action_reserve_study_room",
            exec_context={}
        )
        assert len(chips) == 2
        assert "제1스터디룸(4인실)" in chips[0]
        assert "제2스터디룸(6인실)" in chips[1]
