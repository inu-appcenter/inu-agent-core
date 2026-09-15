from app.core.config import Settings


def test_default_settings():
    s = Settings(
        LLM_BASE_URL="https://test.domain.inu.ac.kr/v1",
        LLM_API_KEY="test-key-xyz",
        LLM_MODEL_NAME="gemma-27b",
    )
    assert s.APP_NAME == "inu-agent-core"
    assert s.LLM_MODEL_NAME == "gemma-27b"
    assert s.LLM_BASE_URL == "https://test.domain.inu.ac.kr/v1"
    assert s.LLM_API_KEY == "test-key-xyz"
    assert "http://localhost:3000" in s.cors_origins_list
