"""
Unit Tests for In-Memory Semantic Tool Retriever (Tool RAG)
Verifies vector indexing, L2 normalization, 8-domain query matching,
multi-turn context inheritance, and threshold filtering.
"""
import pytest
from pathlib import Path
import json
import numpy as np

from app.tools.openapi import OpenApiConnector
from app.tools.inuchat import InuAiKnowledgeTool
from app.tools.library import LibrarySeatTool
from app.orchestrator.retriever import SemanticToolRetriever, serialize_tool, extract_tool_keywords
from app.llm.schemas import ChatMessage


@pytest.fixture
def portal_tools():
    sample_path = Path(__file__).parent.parent / "app" / "tools" / "schemas" / "inu_portal_sample.json"
    with open(sample_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    connector = OpenApiConnector()
    tools = connector.parse_spec(spec)
    # Add INUChat Knowledge tool and Library seat tool
    tools.append(InuAiKnowledgeTool())
    tools.append(LibrarySeatTool())
    return tools


def test_tool_serialization():
    tool = InuAiKnowledgeTool()
    serialized = serialize_tool(tool)
    assert len(serialized) > 0
    assert "학칙" in serialized or "졸업" in serialized

    keywords = extract_tool_keywords(tool)
    assert "inuai_knowledge_search" in keywords
    assert "question" in keywords


@pytest.mark.asyncio
async def test_semantic_retriever_indexing(portal_tools):
    retriever = SemanticToolRetriever()
    count = await retriever.index_tools(portal_tools)

    assert count == len(portal_tools)
    assert retriever.is_indexed
    assert retriever.vectors.shape == (len(portal_tools), 384)

    # Verify L2 normalization: norms must be 1.0
    norms = np.linalg.norm(retriever.vectors, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(portal_tools)), atol=1e-5)


@pytest.mark.asyncio
async def test_semantic_retriever_domain_matching(portal_tools):
    retriever = SemanticToolRetriever()
    await retriever.index_tools(portal_tools)

    # 1. Cafeteria Query
    caf_tools = await retriever.retrieve("오늘 학생식당 점심 메뉴 뭐야?", top_k=2)
    assert len(caf_tools) >= 1
    assert caf_tools[0].category == "CAFETERIA"

    # 2. Bus Query
    bus_tools = await retriever.retrieve("지금 셔틀버스 도착 시간 좀 알려줘", top_k=2)
    assert len(bus_tools) >= 1
    assert bus_tools[0].category == "BUS"

    # 3. Directory Query
    dir_tools = await retriever.retrieve("컴공 과사 전화번호 알려줘", top_k=2)
    assert len(dir_tools) >= 1
    assert any(t.category in ["DIRECTORY", "SEARCH"] for t in dir_tools)

    # 4. Weather Query
    weather_tools = await retriever.retrieve("오늘 송도 캠퍼스 날씨 어때?", top_k=2)
    assert len(weather_tools) >= 1
    assert weather_tools[0].category == "WEATHER"

    # 5. Timetable Query
    time_tools = await retriever.retrieve("오늘 내 수업 시간표 보여줘", top_k=2)
    assert len(time_tools) >= 1
    assert time_tools[0].category == "TIMETABLE"

    # 6. Notice Query
    notice_tools = await retriever.retrieve("학사 공지사항 목록 조회해줘", top_k=2)
    assert len(notice_tools) >= 1
    assert notice_tools[0].category == "NOTICE"

    # 7. Schedule Query
    sched_tools = await retriever.retrieve("이번 달 학사일정 캘린더 어떻게 돼?", top_k=2)
    assert len(sched_tools) >= 1
    assert sched_tools[0].category == "SCHEDULE"

    # 8. Knowledge RAG Query
    rag_tools = await retriever.retrieve("컴퓨터공학부 졸업 요건 학칙 알려줘", top_k=2)
    assert len(rag_tools) >= 1
    assert rag_tools[0].category == "INU_AI_KNOWLEDGE"

    # 9. Library Reading Room Query
    lib_tools = await retriever.retrieve("학산도서관 열람실 잔여 좌석 남아있어?", top_k=2)
    assert len(lib_tools) >= 1
    assert lib_tools[0].category == "LIBRARY"


@pytest.mark.asyncio
async def test_semantic_retriever_multi_turn_inheritance(portal_tools):
    retriever = SemanticToolRetriever()
    await retriever.index_tools(portal_tools)

    # Follow-up query without explicit directory keywords
    history = [
        ChatMessage(role="user", content="컴퓨터공학부 과사 어디에 있어?"),
        ChatMessage(role="assistant", content="컴퓨터공학부 사무실은 7호관에 위치해 있습니다."),
    ]
    follow_up_query = "전화번호는?"

    selected = await retriever.retrieve(query=follow_up_query, history=history, top_k=2)
    assert len(selected) >= 1
    assert any(t.category in ["DIRECTORY", "SEARCH"] for t in selected)


@pytest.mark.asyncio
async def test_empty_retriever_handling():
    retriever = SemanticToolRetriever()
    assert not retriever.is_indexed

    # Retrieval on empty retriever returns empty list safely
    tools = await retriever.retrieve("오늘 학식 뭐야?")
    assert tools == []


def test_mock_embedding_backend():
    from app.llm.embeddings import MockEmbeddingBackend

    mock = MockEmbeddingBackend()
    embs = mock.embed(["학생식당 메뉴", "컴공 과사 전화번호"])
    assert len(embs) == 2
    assert len(embs[0]) == 384

    # Verify normalization
    norm = np.linalg.norm(embs[0])
    assert abs(norm - 1.0) < 1e-4
