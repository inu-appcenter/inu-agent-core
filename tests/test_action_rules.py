import asyncio
from fastapi.testclient import TestClient

from app.main import app
from app.rules.registry import action_rule_registry
from app.rules.models import ActionRule, RuleTarget
from app.tools.action_tool import ClientActionTool

client = TestClient(app)


def test_default_rules_registered():
    rules = action_rule_registry.list_rules()
    assert len(rules) >= 7

    action_ids = [r.action_id for r in rules]
    assert "LMS_GET_COURSES" in action_ids
    assert "LMS_GET_UPCOMING_ASSIGNMENTS" in action_ids
    assert "LIB_GET_READING_ROOMS" in action_ids
    assert "LIB_GET_MY_SEAT" in action_ids
    assert "PORTAL_GET_ACADEMIC_RECORD" in action_ids
    assert "PORTAL_GET_SEMESTER_GRADES" in action_ids


def test_client_action_tool_execution():
    async def _test():
        rule = action_rule_registry.get_rule("LMS_GET_UPCOMING_ASSIGNMENTS")
        assert rule is not None

        tool = ClientActionTool(rule)
        assert tool.category == "LMS"
        assert "action_lms_get_upcoming_assignments" == tool.name

        # Execute
        instruction = await tool.execute(arguments={}, context={})
        assert instruction.auth_domain == "LMS"
        assert instruction.protocol == "HTTP_REST"
        assert "lms.inu.ac.kr" in instruction.request.url
        assert instruction.request.params.get("wsfunction") == "core_calendar_get_action_events_by_timesort"

    asyncio.run(_test())


def test_portal_nexacro_action_tool_execution():
    async def _test():
        rule = action_rule_registry.get_rule("PORTAL_GET_SEMESTER_GRADES")
        assert rule is not None

        tool = ClientActionTool(rule)
        assert tool.category == "PORTAL"

        instruction = await tool.execute(arguments={"term": "1"}, context={})
        assert instruction.auth_domain == "PORTAL"
        assert instruction.protocol == "NEXACRO_SSV"
        assert instruction.request.method == "POST"
        assert instruction.request.headers.get("REQFOUNDATAION") == "nexacro"

    asyncio.run(_test())


def test_rules_api_endpoints():
    # 1. List all rules
    resp = client.get("/api/v1/rules")
    assert resp.status_code == 200
    all_rules = resp.json()
    assert len(all_rules) >= 7

    # 2. Filter by domain
    resp_lms = client.get("/api/v1/rules?domain=LMS")
    assert resp_lms.status_code == 200
    lms_rules = resp_lms.json()
    assert all(r["domain"] == "LMS" for r in lms_rules)

    # 3. Get single rule
    resp_single = client.get("/api/v1/rules/LIB_GET_MY_SEAT")
    assert resp_single.status_code == 200
    assert resp_single.json()["action_id"] == "LIB_GET_MY_SEAT"

    # 4. Hotfix single rule
    update_data = resp_single.json()
    update_data["title"] = "현재 이용 중인 열람실 좌석 (핫픽스 갱신)"
    update_data["version"] = "1.0.1"

    put_resp = client.put("/api/v1/rules/LIB_GET_MY_SEAT", json=update_data)
    assert put_resp.status_code == 200
    assert put_resp.json()["title"] == "현재 이용 중인 열람실 좌석 (핫픽스 갱신)"
    assert put_resp.json()["version"] == "1.0.1"


def test_action_report_synthesizes_lms_list_card():
    report_payload = {
        "action_id": "act_lms_upcoming_12345",
        "success": True,
        "status_code": 200,
        "data": {
            "events": [
                {"name": "컴퓨터구조 과제 #3", "course": {"fullname": "컴퓨터구조론"}},
                {"name": "알고리즘 퀴즈", "course": {"fullname": "알고리즘"}},
            ]
        },
    }

    resp = client.post("/api/v1/action/report", json=report_payload)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "success"
    card = res_data["card"]
    assert card["card_type"] == "LIST_CARD"
    assert len(card["items"]) == 2
    assert "컴퓨터구조 과제 #3" in card["items"][0]["title"]


def test_action_report_synthesizes_portal_metric_card():
    report_payload = {
        "action_id": "act_portal_grade_12345",
        "success": True,
        "status_code": 200,
        "data": {
            "전공평점": "4.25",
            "총취득학점": "105",
            "학적상태": "재학",
        },
    }

    resp = client.post("/api/v1/action/report", json=report_payload)
    assert resp.status_code == 200
    res_data = resp.json()
    assert res_data["status"] == "success"
    card = res_data["card"]
    assert card["card_type"] == "METRIC_CARD"
    assert len(card["sub_details"]) == 3
