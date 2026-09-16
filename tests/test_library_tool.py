import pytest
import asyncio
from app.tools.library import LibrarySeatTool
from app.orchestrator.card_synthesizer import CardSynthesizer


@pytest.mark.asyncio
async def test_library_seat_tool_schema():
    tool = LibrarySeatTool()
    assert tool.name == "library_reading_rooms_status"
    assert tool.category == "LIBRARY"
    schema = tool.get_schema()
    assert schema["type"] == "function"
    assert "room_name" in schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_library_seat_tool_execution():
    tool = LibrarySeatTool()
    res = await tool.execute({}, {})
    assert "rooms" in res or "error" in res
    if "rooms" in res:
        assert isinstance(res["rooms"], list)
        if len(res["rooms"]) > 0:
            first = res["rooms"][0]
            assert "name" in first
            assert "total_seats" in first
            assert "available_seats" in first


def test_library_card_synthesis():
    sample_data = {
        "total_count": 2,
        "rooms": [
            {
                "id": 1,
                "name": "제1열람실",
                "total_seats": 200,
                "occupied_seats": 50,
                "available_seats": 150,
                "utilization_rate": "25.0%",
            },
            {
                "id": 2,
                "name": "제2열람실",
                "total_seats": 100,
                "occupied_seats": 95,
                "available_seats": 5,
                "utilization_rate": "95.0%",
            }
        ]
    }
    card = CardSynthesizer.synthesize_for_domain(
        domain="LIBRARY",
        tool_name="library_reading_rooms_status",
        data=sample_data,
        query="도서관 좌석 남아있어?",
    )
    assert card is not None
    assert card.card_type == "LIST_CARD"
    assert "열람실 좌석 현황" in card.title
    assert len(card.items) == 2
    assert card.items[0].tag == "여유"
    assert card.items[1].tag == "혼잡"
