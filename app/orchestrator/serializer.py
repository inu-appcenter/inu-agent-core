"""
Model Context Protocol (MCP) Generic Tool Data Serializer.

Eliminates domain-specific hardcoded text formatters (anti-pattern) and provides
lossless, compact, and structured JSON observations directly to the LLM.
Preserves all attributes (e.g. gradeEvaluationName, capacity, isuName, etc.)
while pruning nulls/empty structures to conserve token budget.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from app.core.logging import logger


class GenericMcpDataSerializer:
    """
    Generic MCP Data Serializer conforming to Model Context Protocol standards.
    """

    MAX_LIST_ITEMS = 15  # Slicing threshold to avoid context window explosion

    @classmethod
    def clean_data(cls, data: Any) -> Any:
        """
        Recursively prune None, empty strings, empty lists, and empty dicts.
        Preserves boolean False, integer 0, and non-empty values.
        Strips HTML tags like <mark> from strings.
        """
        if isinstance(data, dict):
            cleaned = {}
            for k, v in data.items():
                if isinstance(v, str):
                    v = re.sub(r"</?mark>", "", v).strip()
                cleaned_v = cls.clean_data(v)
                if cleaned_v is not None and cleaned_v != "" and cleaned_v != [] and cleaned_v != {}:
                    cleaned[k] = cleaned_v
            return cleaned
        elif isinstance(data, list):
            cleaned_list = []
            for item in data:
                if isinstance(item, str):
                    item = re.sub(r"</?mark>", "", item).strip()
                cleaned_item = cls.clean_data(item)
                if cleaned_item is not None and cleaned_item != "" and cleaned_item != [] and cleaned_item != {}:
                    cleaned_list.append(cleaned_item)
            return cleaned_list
        elif isinstance(data, str):
            return re.sub(r"</?mark>", "", data).strip()
        return data

    @classmethod
    def extract_core_payload(cls, raw: Any) -> Tuple[Any, Optional[str]]:
        """
        Unpack standard wrapper structures (rawData, content, data, items, etc.)
        and extract any server-provided summary message.
        """
        summary_msg = None
        payload = raw

        if isinstance(raw, dict):
            summary_msg = raw.get("summary") or raw.get("message")
            # Handle common Spring Boot / INU Portal wrapper structures
            if "rawData" in raw and raw["rawData"] is not None:
                payload = raw["rawData"]
            elif "uiComponent" in raw and isinstance(raw["uiComponent"], dict) and "data" in raw["uiComponent"]:
                payload = raw["uiComponent"]["data"]
            elif "content" in raw and raw["content"] is not None:
                payload = raw["content"]
            elif "data" in raw and raw["data"] is not None:
                payload = raw["data"]
            elif "items" in raw and raw["items"] is not None:
                payload = raw["items"]

        return payload, summary_msg

    @classmethod
    def serialize_for_llm(
        cls,
        tool_name: str,
        tool_category: str,
        tool_display_name: str,
        raw_res: Any,
        args: Optional[Dict[str, Any]] = None,
        bus_meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, str, Dict[str, Any], Any]:
        """
        Serializes tool execution result into:
        1. summary_out: Structured observation text for LLM history & system prompt
        2. status_state: 'completed' | 'empty' | 'failed'
        3. status_title: User-facing short status text
        4. ac_data_out: Extracted metadata (e.g. courseOfferingId) for slot filling
        5. sanitized_raw: Cleaned data suitable for CardSynthesizer
        """
        ac_data_out: Dict[str, Any] = {}
        tool_data: Any = raw_res

        # 1. Error check
        if isinstance(raw_res, dict) and "error" in raw_res:
            err_msg = raw_res.get("error", "알 수 없는 통신 오류")
            status_state = "failed"
            status_title = f"{tool_display_name} 조회 실패"
            summary_out = (
                f"\n[{tool_display_name} 조회 실패 ({tool_name})]:\n"
                f"- 오류 내용: {err_msg}\n"
                f"⚠️ 지침: 일시적인 교내 시스템 오류가 발생했음을 사용자에게 사실대로 안내하세요. 없는 정보를 임의로 지어내지 마세요.\n"
            )
            return summary_out, status_state, status_title, ac_data_out, tool_data

        # 2. Extract payload & summary
        payload, server_summary = cls.extract_core_payload(raw_res)

        # 3. Domain-specific light enhancements (without dropping fields)
        # BUS: Filter arrivals by validRoutes if requested stop has specific routes
        if tool_category == "BUS" and bus_meta and isinstance(bus_meta, dict):
            valid_routes = bus_meta.get("validRoutes") or []
            stop_name = bus_meta.get("stopName") or "정류소"
            tab_name = bus_meta.get("tabName") or "정류장"
            raw_arrivals = []
            if isinstance(payload, list):
                raw_arrivals = payload
            elif isinstance(payload, dict):
                raw_arrivals = payload.get("arrivals") or payload.get("items") or []

            filtered_arrivals = [
                item for item in raw_arrivals
                if isinstance(item, dict) and (not valid_routes or item.get("routeNo") in valid_routes)
            ] if valid_routes else raw_arrivals

            payload = {
                "stopName": stop_name,
                "tabName": tab_name,
                "validRoutes": valid_routes,
                "arrivals": filtered_arrivals,
            }
            tool_data = payload

        # Extract useful drill-down ids if present (e.g. courseOfferingId)
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            first_id = payload[0].get("id") or payload[0].get("courseOfferingId")
            if first_id:
                ac_data_out["courseOfferingId"] = first_id
                ac_data_out["targetCourseId"] = first_id
        elif isinstance(payload, dict):
            c_id = payload.get("id") or payload.get("courseOfferingId")
            if c_id:
                ac_data_out["courseOfferingId"] = c_id
                ac_data_out["targetCourseId"] = c_id

        # 4. Clean and prune empty values
        cleaned = cls.clean_data(payload)

        # 5. Slicing check for lists to protect LLM context window
        meta_info = {}
        if isinstance(cleaned, list):
            total_count = len(cleaned)
            if total_count > cls.MAX_LIST_ITEMS:
                cleaned = cleaned[:cls.MAX_LIST_ITEMS]
                meta_info = {
                    "_total_count": total_count,
                    "_displayed_count": cls.MAX_LIST_ITEMS,
                    "_note": f"전체 {total_count}건 중 상위 {cls.MAX_LIST_ITEMS}건만 표시되었습니다."
                }
        elif isinstance(cleaned, dict) and "items" in cleaned and isinstance(cleaned["items"], list):
            items_list = cleaned["items"]
            total_count = len(items_list)
            if total_count > cls.MAX_LIST_ITEMS:
                cleaned["items"] = items_list[:cls.MAX_LIST_ITEMS]
                cleaned["_meta"] = {
                    "_total_count": total_count,
                    "_displayed_count": cls.MAX_LIST_ITEMS,
                    "_note": f"전체 {total_count}건 중 상위 {cls.MAX_LIST_ITEMS}건만 표시되었습니다."
                }

        # 6. Determine status_state & status_title
        is_empty = False
        if cleaned is None or cleaned == "" or cleaned == [] or cleaned == {}:
            is_empty = True
        elif isinstance(cleaned, list) and len(cleaned) == 0:
            is_empty = True
        elif isinstance(cleaned, dict):
            # Special check for bus empty arrivals
            if tool_category == "BUS" and isinstance(cleaned.get("arrivals"), list) and len(cleaned["arrivals"]) == 0:
                is_empty = True
            elif cleaned.get("totalCount") == 0:
                is_empty = True
        elif server_summary and any(kw in server_summary for kw in ["결과가 없습니다", "0건", "등록된 정보가 없습니다", "찾지 못했습니다", "내역이 없습니다", "운영없음"]):
            if isinstance(cleaned, (list, dict)) and not cleaned:
                is_empty = True

        if is_empty:
            status_state = "empty"
            status_title = f"{tool_display_name} 결과 없음"
            if tool_category == "CAFETERIA":
                empty_msg = server_summary or "현재 등록된 식단 메뉴가 없습니다 (식당 운영 시간 외 또는 식단 미등록 상태)."
                guide_msg = "가상의 식단 메뉴를 지어내어 답변하지 마세요. 학생에게 '현재 등록된 식단 메뉴가 없습니다'라고 사실대로 안내하세요."
            else:
                empty_msg = server_summary or f"조회된 {tool_display_name} 데이터가 없습니다."
                guide_msg = "조회된 데이터가 없음을 사용자에게 명확하고 친절하게 안내하세요. 가상의 정보를 임의로 지어내지 마세요."

            summary_out = (
                f"\n[{tool_display_name} 조회 결과 ({tool_name})]:\n"
                f"- 상태: 결과 없음 (0건)\n"
                f"- 안내: {empty_msg}\n"
                f"💡 지침: {guide_msg}\n"
            )

        else:
            status_state = "completed"
            status_title = f"{tool_display_name} 확인 완료"

            serialized_payload = cleaned
            if meta_info and isinstance(serialized_payload, list):
                serialized_payload = {
                    "_meta": meta_info,
                    "items": cleaned,
                }

            json_str = json.dumps(serialized_payload, ensure_ascii=False)

            summary_parts = [
                f"\n[{tool_display_name} 실시간 조회 데이터 ({tool_name})]:",
            ]
            if server_summary:
                summary_parts.append(f"- 서버 요약: {server_summary}")
            summary_parts.append(f"```json\n{json_str}\n```")
            summary_parts.append(
                "💡 핵심 지침: 위 조회된 실제 데이터를 바탕으로 사용자의 질문에 정확하고 성실하게 답변하세요. "
                "데이터에 포함된 모든 속성(예: 평가방식, 학년, 학점, 교수, 시간, 장소, 정원 등)을 자유롭게 참조할 수 있습니다. "
                "데이터에 없는 가상의 사실을 임의로 지어내지 마세요."
            )
            summary_out = "\n".join(summary_parts) + "\n"

        return summary_out, status_state, status_title, ac_data_out, tool_data
