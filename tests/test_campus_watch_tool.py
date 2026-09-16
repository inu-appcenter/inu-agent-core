import pytest
from app.tools.campus_watch import CampusWatchTool
from app.orchestrator.card_synthesizer import CardSynthesizer

@pytest.mark.asyncio
async def test_campus_watch_tool_schema():
    tool = CampusWatchTool()
    assert tool.name == "campus_seat_sniper_and_watch"
    assert tool.category == "CAMPUS_WATCH"
    schema = tool.get_schema()
    assert "properties" in schema["function"]["parameters"]
    assert "action" in schema["function"]["parameters"]["properties"]

@pytest.mark.asyncio
async def test_campus_watch_local_action_execution():
    tool = CampusWatchTool()
    res = await tool.execute({
        "action": "WATCH",
        "target_name": "힐링존",
        "seat_no": "12",
        "duration_minutes": 90,
    })
    assert res["component_type"] == "LOCAL_WATCH_ACTION"
    assert res["data"]["watchType"] == "SPECIFIC_SEAT_SNIPER"
    assert res["data"]["seatNo"] == "12"

@pytest.mark.asyncio
async def test_campus_watch_room_execution():
    tool = CampusWatchTool()
    res = await tool.execute({
        "action": "WATCH",
        "target_name": "힐링존",
        "duration_minutes": 60,
    })
    assert res["component_type"] == "CAMPUS_WATCH_RESULT"
    assert res["data"]["targetName"] == "힐링존"

def test_auth_required_card_synthesis():
    portal_card = CardSynthesizer.synthesize_for_domain("PORTAL", data="AUTH_REQUIRED")
    assert portal_card is not None
    assert portal_card.type == "PORTAL_AUTH_REQUIRED"

    lms_card = CardSynthesizer.synthesize_for_domain("LMS", data="AUTH_REQUIRED")
    assert lms_card is not None
    assert lms_card.type == "LMS_AUTH_REQUIRED"

def test_campus_watch_card_synthesis():
    card = CardSynthesizer.synthesize_for_domain(
        "CAMPUS_WATCH",
        data={
            "component_type": "CAMPUS_WATCH_RESULT",
            "data": {"targetName": "힐링존", "remainingMinutes": 90}
        }
    )
    assert card is not None
    assert card.type == "CAMPUS_WATCH_RESULT"
    assert card.data["targetName"] == "힐링존"
