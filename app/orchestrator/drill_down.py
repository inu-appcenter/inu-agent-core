"""
Autonomous Parallel Deep Drill-Down Engine
Enables Antigravity / Deep Research level autonomous document exploration:
Detects when a search/list tool returned summaries, extracts Top-K candidate identifiers,
and fetches complete body contents in parallel via asyncio.gather.
"""
from typing import Dict, Any, List, Optional, Tuple
import asyncio
import re

from app.core.logging import logger
from app.tools.base import BaseTool
from app.tools.registry import tool_registry


# Foreign Key reference mappings: List Category / Tool -> Candidate ID field & Detail Tool Name
DRILL_DOWN_MAP = {
    "notices": {
        "detail_tool": "api_getNoticeDetail",
        "id_param": "id",
        "title_param": "title",
        "date_param": "createDate",
    },
    "departmentNotices": {
        "detail_tool": "api_getDepartmentNoticeDetail",
        "id_param": "id",
        "title_param": "title",
        "date_param": "createDate",
    },
    "councilNotices": {
        "detail_tool": "api_getCouncilNoticeDetail",
        "id_param": "councilNoticeId",
        "title_param": "title",
        "date_param": "date",
    },
    "courses": {
        "detail_tool": "api_getSyllabus",
        "id_param": "courseOfferingId",
        "title_param": "courseTitle",
    },
}

DRILL_DOWN_KEYWORDS = [
    "내용", "상세", "방법", "신청", "자세히", "어떻게", "언제까지", "조건", "기준",
    "일정", "마감", "서류", "양식", "자격", "절차", "비율", "계획", "금액", "선발", "세부"
]


class DrillDownEvaluator:
    """
    Evaluates list tool observations and orchestrates parallel drill-downs to detail endpoints.
    """

    @classmethod
    def should_drill_down(cls, user_query: str, tool_name: str, hop: int) -> bool:
        """
        Determines whether the user's intent requires deep full-body inspection.
        Only trigger on early hops (hop <= 2) to prevent infinite loops.
        """
        if hop > 2:
            return False

        q = user_query.lower()
        # If user explicitly asks for detailed body/procedure/guidelines
        return any(kw in q for kw in DRILL_DOWN_KEYWORDS)

    @classmethod
    def extract_top_candidates(
        cls, tool_name: str, tool_category: str, raw_data: Any, max_k: int = 3
    ) -> List[Tuple[str, Any, str]]:
        """
        Extracts up to max_k top candidates (domain_key, item_id, item_title) from a list tool result.
        Returns: [(domain_key, item_id, item_title), ...]
        """
        candidates: List[Tuple[str, Any, str]] = []

        if not isinstance(raw_data, dict):
            return candidates

        # Unwrap common API envelopes: {"rawData": ...} or {"data": ...}
        u_data = raw_data
        if isinstance(u_data.get("rawData"), dict):
            u_data = u_data["rawData"]
        if isinstance(u_data.get("data"), dict):
            u_data = u_data["data"]

        # Support list of sections if items contains the dictionary
        if isinstance(u_data.get("items"), list) and u_data["items"] and isinstance(u_data["items"][0], dict):
            first_item = u_data["items"][0]
            if any(k in first_item for k in ["notices", "departmentNotices", "schedules", "directory"]):
                u_data = first_item

        for domain_key in ["notices", "departmentNotices"]:
            domain_section = u_data.get(domain_key)
            if isinstance(domain_section, dict):
                items = domain_section.get("items", [])
                for item in items[:max_k]:
                    if isinstance(item, dict) and item.get("id"):
                        item_id = item["id"]
                        title = item.get("title") or "공지사항"
                        candidates.append((domain_key, item_id, title))

        # 2. Check Standard Notice List endpoints
        if not candidates:
            notice_items = (
                u_data.get("contents")
                or u_data.get("items")
                or (u_data.get("data") if isinstance(u_data.get("data"), list) else [])
            )
            if isinstance(notice_items, list):
                target_domain = "departmentNotices" if "department" in tool_name.lower() else "notices"
                for item in notice_items[:max_k]:
                    if isinstance(item, dict) and item.get("id"):
                        candidates.append((target_domain, item["id"], item.get("title", "공지사항")))

        # 3. Check Council Notice endpoints
        if not candidates and ("council" in tool_name.lower() or "post_3" in tool_name.lower()):
            c_items = u_data.get("contents") or u_data.get("items") or []
            if isinstance(c_items, list):
                for item in c_items[:max_k]:
                    if isinstance(item, dict) and item.get("id"):
                        candidates.append(("councilNotices", item["id"], item.get("title", "총학생회 공지")))

        # 4. Check Course Offerings (for syllabus drill-down)
        if not candidates and ("course" in tool_name.lower() or "offering" in tool_name.lower()):
            c_items = u_data.get("contents") or u_data.get("items") or []
            if isinstance(c_items, list):
                for item in c_items[:max_k]:
                    if isinstance(item, dict) and (item.get("id") or item.get("courseOfferingId")):
                        c_id = item.get("id") or item.get("courseOfferingId")
                        c_title = item.get("courseTitle") or item.get("courseName") or "개설강의"
                        candidates.append(("courses", c_id, c_title))

        return candidates[:max_k]

    @classmethod
    async def fetch_details_in_parallel(
        cls, candidates: List[Tuple[str, Any, str]], exec_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Executes parallel asynchronous HTTP calls via asyncio.gather to fetch full document bodies.
        Returns a list of fetched document dictionaries.
        """
        if not candidates:
            return []

        async def _fetch_single(domain_key: str, item_id: Any, title: str) -> Optional[Dict[str, Any]]:
            mapping = DRILL_DOWN_MAP.get(domain_key)
            if not mapping:
                return None

            detail_tool_name = mapping["detail_tool"]
            id_param = mapping["id_param"]

            target_tool = tool_registry.get_tool(detail_tool_name)
            if not target_tool:
                logger.debug(f"DrillDown: detail tool {detail_tool_name} not found in registry")
                return None

            args = {id_param: item_id}
            try:
                res = await target_tool.execute(args, exec_context)
                if isinstance(res, dict) and "error" not in res:
                    return {
                        "domain": domain_key,
                        "id": item_id,
                        "title": title,
                        "data": res.get("data") or res.get("content") or res,
                    }
            except Exception as e:
                logger.warning(f"DrillDown: failed to fetch detail for {domain_key}:{item_id}: {e}")
            return None

        tasks = [_fetch_single(d, i, t) for d, i, t in candidates]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_docs = []
        for r in results:
            if isinstance(r, dict):
                valid_docs.append(r)
        return valid_docs

    @classmethod
    def format_drill_down_summary(cls, documents: List[Dict[str, Any]]) -> str:
        """
        Formats parallel-fetched full documents into high-density ground truth text for the Synthesis LLM.
        """
        if not documents:
            return ""

        lines = [f"\n[심층 세부 열람 데이터 (총 {len(documents)}건 병렬 확인 완료)]:"]
        for idx, doc in enumerate(documents, start=1):
            title = doc.get("title", "상세 내용")
            item_id = doc.get("id")
            data = doc.get("data", {})

            content_text = ""
            if isinstance(data, dict):
                # Check for standard notice body fields
                content_text = (
                    data.get("content")
                    or data.get("body")
                    or data.get("description")
                    or data.get("noticeContent")
                    or ""
                )
                if isinstance(data, dict) and any(k in data for k in ["학습평가방법", "성적평가비율", "주별수업계획"]):
                    eval_parts = []
                    if data.get("과목명"): eval_parts.append(f"과목명: {data['과목명']}")
                    if data.get("교수"): eval_parts.append(f"교수: {data['교수']}")
                    if data.get("학습평가방법"): eval_parts.append(f"학습평가방법: {data['학습평가방법']}")
                    if data.get("성적평가비율"): eval_parts.append(f"성적평가비율: {json.dumps(data['성적평가비율'], ensure_ascii=False)}")
                    if data.get("주별수업계획"): eval_parts.append(f"주별수업계획: {json.dumps(data['주별수업계획'][:15], ensure_ascii=False)}")
                    content_text = "\n".join(eval_parts)
                elif not content_text and "평가비율" in str(data):
                    # Syllabus dictionary
                    content_text = str(data)
            elif isinstance(data, str):
                content_text = data

            # Clean HTML tags if present in notice body
            clean_content = re.sub(r"<[^>]+>", " ", content_text).strip()
            clean_content = re.sub(r"\s+", " ", clean_content)[:1200]  # Cap each body at 1200 chars for token balance

            lines.append(f"### 문서 #{idx}: [{title} (ID: {item_id})]")
            if clean_content:
                lines.append(f"{clean_content}\n")
            else:
                lines.append("본문 텍스트 없음 또는 첨부파일 안내형 공지\n")

        lines.append(
            "💡 [심층 분석 지침]: 위 본문 전문의 실제 신청 기한, 자격 조건, 제출 서류, 세부 기준을 교차 종합하여 "
            "사용자에게 명확하고 완성도 높은 안내를 제공하세요. 절대로 요약본만 읊지 말고 본문의 핵심 팩트를 답변에 온전히 포함하세요.\n"
        )
        return "\n".join(lines)
