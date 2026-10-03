import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tools.registry import tool_registry

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_tools():
    tool_registry.register_client_action_rules()


def test_mcp_initialize():
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    }
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["jsonrpc"] == "2.0"
    assert data["id"] == 1
    result = data["result"]
    assert result["protocolVersion"] == "2024-11-05"
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "inu-portal-mcp-server"


def test_mcp_ping():
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "ping",
        "params": {}
    }
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    assert resp.json()["result"] == {}


def test_mcp_tools_list():
    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/list",
        "params": {}
    }
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    tools = data["result"]["tools"]
    assert len(tools) > 0

    tool_names = [t["name"] for t in tools]
    assert any("academic_record" in name.lower() for name in tool_names)

    # Check client action metadata presence
    client_tools = [t for t in tools if t["metadata"].get("executionType") == "CLIENT_ACTION"]
    assert len(client_tools) > 0
    sample = client_tools[0]
    assert "authDomain" in sample["metadata"]
    assert "protocol" in sample["metadata"]
    assert "actionType" in sample["metadata"]


def test_mcp_tools_call_client_action():
    payload = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "action_portal_get_academic_record",
            "arguments": {}
        }
    }
    resp = client.post("/mcp", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["isError"] is False
    assert "clientAction" in data["result"]
    action = data["result"]["clientAction"]
    assert action["auth_domain"] == "PORTAL"
    assert action["protocol"] == "NEXACRO_SSV"


def test_mcp_resources_list_and_read():
    # 1. List
    list_payload = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "resources/list",
        "params": {}
    }
    resp = client.post("/mcp", json=list_payload)
    assert resp.status_code == 200
    resources = resp.json()["result"]["resources"]
    assert len(resources) >= 2
    uris = [r["uri"] for r in resources]
    assert "inu://calendar/academic" in uris

    # 2. Read
    read_payload = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "resources/read",
        "params": {"uri": "inu://calendar/academic"}
    }
    resp_read = client.post("/mcp", json=read_payload)
    assert resp_read.status_code == 200
    contents = resp_read.json()["result"]["contents"]
    assert len(contents) == 1
    assert "course_registration" in contents[0]["text"]


def test_mcp_sse_transport_headers():
    resp = client.get("/mcp/sse?single=true")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "event: endpoint" in resp.text
    assert "/mcp/messages" in resp.text
