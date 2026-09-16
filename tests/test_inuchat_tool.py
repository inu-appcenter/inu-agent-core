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
    sample_text = "제37조 및 제14조에 따라 [2024학년도 동계 계절학기 수강신청 안내](https://www.inu.ac.kr/inu/666/subview.do) 공지 참조"
    citations = tool._extract_citations(sample_text)
    assert len(citations) >= 2
    types = [c["type"] for c in citations]
    assert "URL" in types
    assert "LAW" in types

    url_cit = next(c for c in citations if c["type"] == "URL")
    assert url_cit["title"] == "2024학년도 동계 계절학기 수강신청 안내"
    assert url_cit["url"] == "https://www.inu.ac.kr/inu/666/subview.do"


def test_academic_card_synthesis_normalization():
    from app.orchestrator.card_synthesizer import CardSynthesizer
    sample_academic = {
        "koreanName": "홍길동",
        "studentId": "202101234",
        "departmentName": "컴퓨터공학부",
        "collegeName": "정보기술대학",
        "enrollmentStatus": "재학",
        "latestEnrollmentChange": "졸업유예",
        "completedSemesterCount": "8학기",
        "acquiredCredits": "140",
        "gradeAverage": "4.24",
        "advisorProfessorName": "김교수",
    }
    card = CardSynthesizer.synthesize_for_domain(
        domain="PORTAL",
        tool_name="PORTAL_GET_ACADEMIC_RECORD",
        data=sample_academic,
        query="내 학적 정보",
    )
    assert card is not None
    assert card.card_type == "COMPONENT_CARD"
    assert card.type == "ACADEMIC_INFO"
    data = card.data
    assert data["koreanName"] == "홍길동"
    assert data["advisorProfessorName"] == "김교수"
    assert data["departmentName"] == "컴퓨터공학부"
    assert data["gradeAverage"] == "4.24"
    assert data["latestEnrollmentChange"] == "졸업유예"


