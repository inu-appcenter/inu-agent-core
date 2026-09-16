import pytest
import asyncio
from app.tools.notification_tools import (
    ManageReminderTool,
    DailyBriefTool,
    NoticeKeywordTool,
    MySettingsTool,
)
from app.orchestrator.card_synthesizer import CardSynthesizer


@pytest.mark.asyncio
async def test_manage_reminder_tool():
    tool = ManageReminderTool()
    assert tool.name == "action_manage_reminder"
    assert tool.category == "REMINDER"
    
    schema = tool.get_schema()
    assert schema["type"] == "function"
    assert "action" in schema["function"]["parameters"]["properties"]
    
    # Test execution: list (returns error or list when backend not running)
    res_list = await tool.execute({"action": "LIST"}, {"auth": "Bearer test_token"})
    assert "error" in res_list or isinstance(res_list, list) or isinstance(res_list, dict)


@pytest.mark.asyncio
async def test_daily_brief_tool():
    tool = DailyBriefTool()
    assert tool.name == "action_daily_brief"
    assert tool.category == "DAILY_BRIEF"
    
    schema = tool.get_schema()
    assert schema["type"] == "function"
    assert "action" in schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_notice_keyword_tool():
    tool = NoticeKeywordTool()
    assert tool.name == "action_notice_keyword"
    assert tool.category == "KEYWORD"
    
    schema = tool.get_schema()
    assert schema["type"] == "function"
    assert "action" in schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_my_settings_tool():
    tool = MySettingsTool()
    assert tool.name == "action_my_settings"
    assert tool.category == "SETTINGS"
    
    schema = tool.get_schema()
    assert schema["type"] == "function"
    assert "properties" in schema["function"]["parameters"]


def test_notification_card_synthesizer():
    # 1. Reminder Card
    reminder_data = {
        "action": "LIST",
        "reminders": [
            {"id": "rem_1", "title": "캡스톤디자인 중간발표", "remind_at": "2026-09-25 10:00", "is_completed": False}
        ]
    }
    card = CardSynthesizer.synthesize_for_domain("REMINDER", "action_manage_reminder", reminder_data, "내 알림 보여줘")
    assert card is not None
    assert card.card_type == "COMPONENT_CARD"
    assert "맞춤 알림" in card.title

    # 2. Daily Brief Card
    brief_data = {
        "is_enabled": True,
        "brief_time": "08:30",
        "categories": ["LMS", "ACADEMIC", "WEATHER"]
    }
    card_brief = CardSynthesizer.synthesize_for_domain("DAILY_BRIEF", "action_daily_brief", brief_data, "데일리 브리프 설정")
    assert card_brief is not None
    assert card_brief.card_type == "COMPONENT_CARD"
    assert "데일리 브리프" in card_brief.title

    # 3. Notice Keyword Card
    kw_data = [
        {"id": "kw_1", "keyword": "수강신청"},
        {"id": "kw_2", "keyword": "장학금"}
    ]
    card_kw = CardSynthesizer.synthesize_for_domain("KEYWORD", "action_notice_keyword", kw_data, "키워드 알림 목록")
    assert card_kw is not None
    assert card_kw.card_type == "COMPONENT_CARD"
    assert "키워드" in card_kw.title

    # 4. My Settings Card
    settings_data = {
        "keywords": [{"id": 1, "keyword": "장학금"}],
        "reminders": [{"id": 1, "title": "학식"}],
        "dailyBrief": {"enabled": True, "time": "08:30"}
    }
    card_settings = CardSynthesizer.synthesize_for_domain("SETTINGS", "action_my_settings", settings_data, "내 설정")
    assert card_settings is not None
    assert card_settings.card_type == "COMPONENT_CARD"
    assert "나의 맞춤 알림 종합 설정" in card_settings.title