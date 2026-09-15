"""
Client Action Reporting & Card Synthesis Endpoint
Receives P2P execution results from mobile app Action Runner and synthesizes SDUI Cards.
"""
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException

from app.llm.schemas import (
    ClientActionResult,
    GenerativeCard,
    MetricCard,
    MetricCardItem,
    StatusCard,
    ListCard,
    ListItem,
    CardBadge,
)
from app.core.logging import logger

router = APIRouter()


@router.post("/action/report", summary="Report on-device Client Action result and synthesize SDUI Card")
async def report_action_result(report: ClientActionResult) -> Dict[str, Any]:
    logger.info(f"Received Client Action report: action_id={report.action_id}, success={report.success}")

    if not report.success:
        return {
            "status": "error",
            "action_id": report.action_id,
            "message": report.error_message or "학교 시스템 연동 중 오류가 발생했습니다.",
            "card": None,
        }

    data = report.data
    card: GenerativeCard

    # 1. LMS 과제/일정 결과
    if "lms" in report.action_id.lower() or (isinstance(data, dict) and ("events" in data or isinstance(data, list))):
        items: List[ListItem] = []
        raw_items = data.get("events", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])

        for idx, item in enumerate(raw_items[:5]):
            name = item.get("name") or item.get("fullname") or f"과제 {idx+1}"
            course = item.get("course", {}).get("fullname") if isinstance(item.get("course"), dict) else ""
            items.append(ListItem(
                title=name,
                subtitle=course if course else None,
                tag="마감 예정",
            ))

        card = ListCard(
            title="사이버캠퍼스(LMS) 과제 및 일정",
            items=items if items else [ListItem(title="예정된 과제가 없습니다.", subtitle="모든 과제를 완료했습니다.")],
            footer_text=f"총 {len(raw_items)}건 확인됨",
        )

    # 2. 도서관 열람실 좌석 결과
    elif "lib" in report.action_id.lower() or "seat" in report.action_id.lower():
        if isinstance(data, dict) and "room" in data:
            room_name = data.get("room", {}).get("name", "열람실")
            seat_no = data.get("seatNumber", "좌석")
            card = StatusCard(
                title=f"도서관 {room_name} ({seat_no}번)",
                status="이용 중",
                progress_percent=70,
                time_remaining=data.get("remainingTime", "이용 가능"),
                primary_action_label="좌석 연장하기",
                action_id="LIB_RENEW_SEAT",
            )
        else:
            card = StatusCard(
                title="도서관 좌석 상태",
                status="정상 확인됨",
                primary_action_label="좌석 현황 보기",
            )

    # 3. 포털 성적/학적 결과
    elif "portal" in report.action_id.lower() or "grade" in report.action_id.lower():
        sub_details: List[MetricCardItem] = []
        if isinstance(data, dict):
            for k, v in list(data.items())[:4]:
                sub_details.append(MetricCardItem(label=str(k), value=str(v)))

        card = MetricCard(
            title="포털 종합정보시스템 연동 결과",
            badge=CardBadge(text="조회 성공", theme="success"),
            main_metric=MetricCardItem(label="조회 상태", value="정상", highlight=True),
            sub_details=sub_details,
        )

    # 4. 기본 Fallback 카드
    else:
        card = ListCard(
            title="외부 시스템 연동 완료",
            items=[ListItem(title=f"작업 ID: {report.action_id}", subtitle="정상적으로 데이터가 수신되었습니다.")],
        )

    return {
        "status": "success",
        "action_id": report.action_id,
        "message": "성공적으로 데이터를 수신하여 카드를 생성했습니다.",
        "card": card.model_dump(),
    }
