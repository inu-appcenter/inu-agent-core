import pytest
from app.orchestrator.router import AgentRouter
from app.tools.base import BaseTool


class MockDirectoryTool(BaseTool):
    def __init__(self):
        self.name = "api_directory_getEntries"
        self.description = "교내 전화번호부 및 부서/교수 연락처를 조회합니다."
        self.category = "DIRECTORY"

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "조회할 학과명, 교수/교직원 성함 또는 부서명",
                        }
                    },
                    "required": ["query"],
                },
            },
        }

    async def execute(self, arguments, context):
        return []


def test_clean_entity_query():
    assert AgentRouter._clean_entity_query("홍길동 교수님") == "홍길동"
    assert AgentRouter._clean_entity_query("박문주 교수") == "박문주"
    assert AgentRouter._clean_entity_query("컴퓨터공학부 과사") == "컴퓨터공학부"
    assert AgentRouter._clean_entity_query("학생지원과 사무실") == "학생지원과"
    assert AgentRouter._clean_entity_query("홍길동 전화번호") == "홍길동"
    assert AgentRouter._clean_entity_query("김교수") == "김교수"  # 2글자 미만 잘림 방지


def test_is_generic_contact_term():
    assert AgentRouter._is_generic_contact_term("전화번호") is True
    assert AgentRouter._is_generic_contact_term("전화번호 알려줘") is True
    assert AgentRouter._is_generic_contact_term("전화번호 누구야") is True
    assert AgentRouter._is_generic_contact_term("연락처") is True
    assert AgentRouter._is_generic_contact_term("홍길동") is False
    assert AgentRouter._is_generic_contact_term("컴퓨터공학부") is False


@pytest.mark.asyncio
async def test_extract_tool_arguments_fallback_from_history():
    tool = MockDirectoryTool()
    history = [
        {"role": "user", "content": "내 담임교수님 누구야?"},
        {"role": "assistant", "content": "학우님의 지도교수님은 홍길동 교수님입니다."},
    ]
    # LLM will be guided or fallback
    args = await AgentRouter.extract_tool_arguments(
        tool=tool,
        query="전화번호 누구야?",
        history=history,
        client_context=None,
    )
    # The query should resolve to "홍길동"
    assert args.get("query") == "홍길동"


@pytest.mark.asyncio
async def test_extract_tool_arguments_fallback_from_client_context():
    tool = MockDirectoryTool()
    client_context = {
        "academicDisplay": {
            "advisorProfessorName": "이순신",
            "departmentName": "컴퓨터공학부",
        }
    }
    args = await AgentRouter.extract_tool_arguments(
        tool=tool,
        query="연락처 알려줘",
        history=[],
        client_context=client_context,
    )
    assert args.get("query") == "이순신"
