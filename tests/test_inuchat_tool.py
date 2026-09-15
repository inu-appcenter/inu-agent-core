import pytest
from app.tools.inuchat import InuAiKnowledgeTool
from app.orchestrator.pruner import ToolPruner


def test_inuchat_pruning():
    tool = InuAiKnowledgeTool()
    assert tool.name == "inuai_knowledge_search"
    assert tool.category == "INU_AI_KNOWLEDGE"

    pruned = ToolPruner.prune("컴퓨터공학부 졸업 요건 알려줘", [tool], client="INTIP")
    assert len(pruned) == 1
    assert pruned[0].name == "inuai_knowledge_search"


def test_inuchat_citations_extraction():
    tool = InuAiKnowledgeTool()
    sample_text = "제37조 및 제14조에 따라 https://www.inu.ac.kr/inu/666/subview.do 공지 참조"
    citations = tool._extract_citations(sample_text)
    assert len(citations) >= 2
    types = [c["type"] for c in citations]
    assert "URL" in types
    assert "LAW" in types

