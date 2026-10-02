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
            title="이러닝(LMS) 과제 및 일정",
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

    # 3. 포털 수강신청 / 시간표 결과
    elif "timetable" in report.action_id.lower() or (isinstance(data, dict) and ("timetable" in data or "timetableSsv" in data or "courses" in data)):
        items: List[ListItem] = []
        raw_courses = []
        if isinstance(data, list):
            raw_courses = data
        elif isinstance(data, dict):
            raw_courses = data.get("courses") or data.get("items") or data.get("timetable") or []

        for c in raw_courses[:6]:
            if isinstance(c, dict):
                c_name = c.get("courseName") or c.get("title") or c.get("subject") or "수강 과목"
                prof = c.get("professor") or c.get("profNm") or ""
                room = c.get("classroom") or c.get("room") or c.get("timeRoom") or ""
                time_slot = c.get("time") or c.get("timeStr") or ""
                sub_parts = [p for p in [f"교수: {prof}" if prof else "", time_slot, room] if p]
                items.append(
                    ListItem(
                        title=str(c_name),
                        subtitle=" · ".join(sub_parts) if sub_parts else "수강신청 과목",
                        tag="공식수강",
                        link="/timetable",
                    )
                )

        if not items:
            items.append(
                ListItem(
                    title="수강신청 내역 확인 완료",
                    subtitle="이번 학기 공식 수강신청 내역이 정상 연동되었습니다.",
                    tag="포털연동",
                    link="/timetable",
                )
            )

        card = ListCard(
            title="📋 학교 포털 수강신청 시간표",
            items=items,
            footer_text="학교 종합정보시스템(ERP) 공식 수강신청 데이터입니다.",
        )

    # 4. 포털 성적/학적 결과
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

    # 5. 기본 Fallback 카드
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
