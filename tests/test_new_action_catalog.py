import asyncio
import pytest
from app.rules.registry import action_rule_registry
from app.tools.action_tool import ClientActionTool


def test_new_action_rules_registered():
    rules = action_rule_registry.list_rules()
    action_ids = [r.action_id for r in rules]

    assert "PORTAL_GET_TUITION" in action_ids
    assert "DORM_APPLY_SLEEPOVER" in action_ids
    assert "PORTAL_DOWNLOAD_SYLLABUS" in action_ids


def test_tuition_rule_properties():
    rule = action_rule_registry.get_rule("PORTAL_GET_TUITION")
    assert rule is not None
    assert rule.domain == "PORTAL"
    assert rule.protocol == "NEXACRO_SSV"
    assert rule.action_type == "QUERY"
    assert rule.default_card_type == "METRIC_CARD"

    tool = ClientActionTool(rule)
    assert tool.category == "PORTAL"


@pytest.mark.asyncio
async def test_dorm_sleepover_mutation_requires_confirmation():
    rule = action_rule_registry.get_rule("DORM_APPLY_SLEEPOVER")
    assert rule is not None
    assert rule.domain == "DORM"
    assert rule.protocol == "NEXACRO_SSV"
    assert rule.action_type == "MUTATION"
    assert rule.requires_confirmation is True
    assert rule.confirmation_message is not None
    assert rule.default_card_type == "ACTION_CARD"

    tool = ClientActionTool(rule)
    instruction = await tool.execute(
        arguments={"startDt": "20261003", "endDt": "20261004", "reason": "본가 귀가"},
        context={}
    )
    assert instruction.auth_domain == "PORTAL"
    assert instruction.protocol == "NEXACRO_SSV"
    assert instruction.action_type == "MUTATION"
    assert instruction.requires_confirmation is True
    assert instruction.request.params.get("startDt") == "20261003"


@pytest.mark.asyncio
async def test_syllabus_download_rule():
    rule = action_rule_registry.get_rule("PORTAL_DOWNLOAD_SYLLABUS")
    assert rule is not None
    assert rule.domain == "PORTAL"
    assert rule.protocol == "HTTP_REST"
    assert rule.action_type == "DOWNLOAD"
    assert rule.download_metadata is not None
    assert rule.download_metadata.get("file_name") == "강의계획서.pdf"

    tool = ClientActionTool(rule)
    instruction = await tool.execute(
        arguments={"subjectCode": "CSE101", "yy": "2026", "tmGbn": "10"},
        context={}
    )
    assert instruction.auth_domain == "PORTAL"
    assert instruction.protocol == "HTTP_REST"
    assert instruction.action_type == "DOWNLOAD"
    assert instruction.download_metadata == {"file_name": "강의계획서.pdf", "mime_type": "application/pdf"}
    assert instruction.request.params.get("subjectCode") == "CSE101"
