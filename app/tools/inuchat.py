"""
INUChat RAG Tool for inu-agent-core
Connects to https://ai-server.inuappcenter.kr/inuchat/chat to retrieve official INU regulations,
academic policies, graduation requirements, double-major/transfer criteria, and campus citations.
"""
from typing import Dict, Any, Optional, List
import re
import httpx

from app.tools.base import BaseTool
from app.core.config import settings
from app.core.logging import logger


class InuAiKnowledgeTool(BaseTool):
    def __init__(self, base_url: Optional[str] = None):
        self.name = "inuai_knowledge_search"
        self.description = "인천대학교 공식 학칙, 학사 규정, 졸업 요건, 복수전공/부전공/전과 기준, 휴학/복학/학사경고, 장학금 규정 및 상세 공지사항 RAG 검색"
        self.category = "INU_AI_KNOWLEDGE"
        self.base_url = base_url or settings.INUCHAT_BASE_URL
        self.endpoint = f"{self.base_url.rstrip('/')}/inuchat/chat"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "인천대학교 학칙, 졸업요건, 규정, 공지사항 관련 질문 내용",
                    }
                },
                "required": ["question"],
            },
        }

    async def execute(self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        question = params.get("question") or (context or {}).get("query") or ""
        if not question:
            return {"success": False, "error": "No question provided"}

        logger.info(f"Calling InuChat RAG Knowledge Tool for: '{question[:50]}'")
        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.post(
                    self.endpoint,
                    json={"question": question, "history": []},
                    headers={"X-Guest-Device-Id": "agent-core"},
                )
                if resp.status_code == 200:
                    text = resp.text
                    citations = self._extract_citations(text)
                    return {
                        "success": True,
                        "data": {
                            "rag_answer": text,
                            "question": question,
                            "citations": citations,
                        },
                    }
                else:
                    logger.warning(f"InuChat returned status {resp.status_code}: {resp.text[:100]}")
                    return {"success": False, "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            logger.error(f"InuChat request failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _extract_citations(self, answer: str) -> List[Dict[str, str]]:
        citations = []
        seen = set()

        # URLs
        for url in re.findall(r"https?://[^\s)\]]+", answer):
            if url not in seen:
                seen.add(url)
                citations.append({"type": "URL", "title": "관련 공지/원문 바로가기", "url": url})

        # Articles (학칙 제OO조)
        for art in re.findall(r"(제\s*\d+\s*조(?:의\s*\d+)?(?:\s*\([^)]+\))?)", answer):
            clean_art = art.strip()
            if clean_art not in seen:
                seen.add(clean_art)
                citations.append({
                    "type": "LAW",
                    "title": f"인천대학교 학칙 {clean_art}",
                    "url": "https://www.inu.ac.kr/inu/1560/subview.do",
                })

        return citations

