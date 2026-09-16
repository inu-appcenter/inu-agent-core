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

    async def stream_execute(self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        """
        Stream response tokens directly from InuChat /inuchat/chat for 0ms latency direct pass-through.
        Yields tuple: (token: str, is_done: bool, accumulated_text: str, citations: List[Dict[str, str]])
        """
        question = params.get("question") or (context or {}).get("query") or ""
        if not question:
            return

        logger.info(f"Stream calling InuChat RAG Knowledge Tool for: '{question[:50]}'")
        accumulated = []
        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
                async with client.stream(
                    "POST",
                    self.endpoint,
                    json={"question": question, "history": []},
                    headers={"X-Guest-Device-Id": "agent-core"},
                ) as resp:
                    if resp.status_code != 200:
                        logger.warning(f"InuChat stream returned status {resp.status_code}")
                        yield ("인천대학교 학사 규정 지식베이스를 조회하는 중 일시적인 지연이 발생했습니다.", True, "", [])
                        return

                    async for chunk in resp.aiter_text():
                        if chunk:
                            accumulated.append(chunk)
                            yield (chunk, False, "".join(accumulated), [])

            full_text = "".join(accumulated)
            citations = self._extract_citations(full_text)
            yield ("", True, full_text, citations)
        except Exception as e:
            logger.error(f"InuChat stream request failed: {e}", exc_info=True)
            if not accumulated:
                yield ("학사 규정 답변을 가져오는 중 일시적인 오류가 발생했습니다.", True, "", [])

    def _extract_citations(self, answer: str) -> List[Dict[str, str]]:
        citations = []
        seen = set()

        # 1. 마크다운 링크 추출: [공지사항 제목](https://...)
        md_link_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
        for match in md_link_pattern.finditer(answer):
            raw_title = match.group(1).strip()
            url = match.group(2).strip()
            if url not in seen and raw_title:
                # "원문 바로가기", "링크", "여기" 등 무의미한 텍스트가 아닌 경우 실제 공지 제목으로 사용
                meaningful_title = raw_title if len(raw_title) > 2 and not re.match(r"^(링크|바로가기|여기|클릭|url|link)$", raw_title.lower()) else ""
                seen.add(url)
                citations.append({
                    "type": "URL",
                    "title": meaningful_title or self._infer_url_title(url, answer),
                    "url": url,
                })

        # 2. 텍스트 라인 기반 추출: • 제목 : https://... 또는 1. 제목 - https://...
        line_pattern = re.compile(
            r"(?:^|[\r\n])\s*(?:[-*•]|\d+[.)])\s*([^\r\n:–—\-]+?)\s*[:–—\-]\s*(https?://[^\s)\]]+)",
            re.MULTILINE,
        )
        for match in line_pattern.finditer(answer):
            raw_title = match.group(1).strip()
            url = match.group(2).strip()
            if url not in seen and raw_title and len(raw_title) > 2:
                seen.add(url)
                citations.append({
                    "type": "URL",
                    "title": raw_title,
                    "url": url,
                })

        # 3. 단독 URL fallback: URL의 경로/게시판 특성 및 문맥을 활용한 제목 추론
        for url in re.findall(r"https?://[^\s)\]]+", answer):
            if url not in seen:
                seen.add(url)
                citations.append({
                    "type": "URL",
                    "title": self._infer_url_title(url, answer),
                    "url": url,
                })

        # 4. Articles (학칙 제OO조)
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

    def _infer_url_title(self, url: str, context: str) -> str:
        """
        URL 주변 텍스트나 도메인/경로로부터 유의미하고 명확한 제목을 추론합니다.
        """
        # 본문에서 해당 URL 바로 앞에 위치한 문장이나 키워드 검색
        escaped_url = re.escape(url)
        context_match = re.search(r"([가-힣a-zA-Z0-9\s()\[\]]{3,35})(?:\s*[:\-–—\(\[]?\s*)" + escaped_url, context)
        if context_match:
            candidate = context_match.group(1).strip()
            clean_cand = re.sub(r"^[-*•\d.)\s]+", "", candidate).strip()
            if len(clean_cand) >= 3 and not re.match(r"^(링크|바로가기|여기|클릭|출처|참고|url|link)$", clean_cand.lower()):
                return clean_cand

        # 도메인/경로 기반 구체적 출처 표기
        if "inu.ac.kr" in url:
            if "1560" in url or "rule" in url or "subview.do" in url:
                return "인천대학교 학칙 및 규정집 원문"
            return "인천대학교 학사 공지사항"
        elif "dorm.inu.ac.kr" in url:
            return "생활관(기숙사) 공지사항"

        return "관련 공지 원문"

