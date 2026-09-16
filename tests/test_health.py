from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "inu-agent-core"
    assert data["health"] == "/api/v1/health"


def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app_name" in data


def test_chat_stream_headers():
    # Out of scope message returns immediate refusal and 200 SSE stream
    payload = {"message": "자바스크립트 코드 짜줘"}
    response = client.post(
        "/api/v1/chat/stream",
        json=payload,
        headers={"Auth": "sample_jwt_token_123", "X-AppCenter-Client": "INTIP"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

