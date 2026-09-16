"""
Library Agent Tool for inu-agent-core
Handles real-time seat queries, on-device seat reservation cards (LIBRARY_SEAT_CONFIRM),
and study room reservation cards (LIBRARY_STUDY_ROOM_CONFIRM).
"""
from typing import Dict, Any, List, Optional
import httpx
import re

from app.tools.base import BaseTool
from app.core.logging import logger

ROOM_ID_MAP = {
    "1열람": 1,
    "제1열람": 1,
    "2열람": 2,
    "제2열람": 2,
    "3열람": 3,
    "제3열람": 3,
    "1노트북": 4,
    "제1노트북": 4,
    "3노트북": 124,
    "제3노트북": 124,
    "힐링": 107,
    "힐링존": 107,
}

STUDY_ROOM_CATALOG = [
    {"id": 9, "name": "205호", "location": "중앙관 2층", "quota": "4~8명", "tags": ["전자칠판", "화이트보드"]},
    {"id": 10, "name": "206호", "location": "중앙관 2층", "quota": "4~8명", "tags": ["전자칠판", "화이트보드"]},
    {"id": 11, "name": "207호", "location": "중앙관 2층", "quota": "2~4명", "tags": ["소형 스터디", "모니터"]},
    {"id": 12, "name": "208호", "location": "중앙관 2층", "quota": "2~4명", "tags": ["소형 스터디", "모니터"]},
    {"id": 13, "name": "209호", "location": "중앙관 2층", "quota": "4~8명", "tags": ["화이트보드"]},
    {"id": 14, "name": "305호", "location": "중앙관 3층", "quota": "7~14명", "tags": ["대형 세미나", "빔프로젝터"]},
    {"id": 15, "name": "306호", "location": "중앙관 3층", "quota": "7~14명", "tags": ["대형 세미나", "빔프로젝터"]},
    {"id": 41, "name": "스터디룸-1", "location": "이룸관 3층", "quota": "2~4명", "tags": ["화이트보드"]},
    {"id": 42, "name": "스터디룸-2", "location": "이룸관 3층", "quota": "2~4명", "tags": ["화이트보드"]},
    {"id": 45, "name": "스터디룸-5", "location": "이룸관 3층", "quota": "4~8명", "tags": ["전자칠판", "화이트보드"]},
]


class LibrarySeatTool(BaseTool):
    def __init__(self):
        self.name = "library_reading_rooms_status"
        self.description = (
            "인천대학교 학산도서관 열람실 실시간 잔여 좌석 조회 및 "
            "열람실 특정 좌석 배정 신청(대화형 카드), 스터디룸 예약 신청을 처리합니다."
        )
        self.category = "LIBRARY"
        self.api_url = "https://lib.inu.ac.kr/pyxis-api/1/seat-rooms"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "enum": ["SEATS", "RESERVE_SEAT", "STUDY_ROOMS", "RESERVE_STUDY_ROOM"],
                            "description": "수행 작업: 'SEATS'(잔여 좌석 조회), 'RESERVE_SEAT'(좌석 배정 신청), 'STUDY_ROOMS'(스터디룸 목록), 'RESERVE_STUDY_ROOM'(스터디룸 예약)",
                        },
                        "room_name": {
                            "type": "string",
                            "description": "열람실 또는 스터디룸 명칭 (예: '제1열람실', '제1노트북실', '205호')",
                        },
                        "seat_no": {
                            "type": "string",
                            "description": "배정 희망 좌석 번호 (예: '25', '30', 'A12')",
                        },
                        "room_id": {
                            "type": "integer",
                            "description": "열람실 또는 스터디룸 ID (선택사항)",
                        },
                        "date": {
                            "type": "string",
                            "description": "스터디룸 이용 날짜 (YYYY-MM-DD)",
                        },
                        "begin_time": {
                            "type": "string",
                            "description": "시작 시간 (HH:mm)",
                        },
                        "end_time": {
                            "type": "string",
                            "description": "종료 시간 (HH:mm)",
                        },
                    },
                    "required": [],
                },
            },
        }

    def _resolve_room_id(self, room_name: str) -> Optional[int]:
        if not room_name:
            return None
        clean = room_name.replace(" ", "")
        for key, val in ROOM_ID_MAP.items():
            if key in clean:
                return val
        return None

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        query = (context.get("query") or arguments.get("query") or "").strip()
        target = (arguments.get("target") or "").upper()
        room_name = (arguments.get("room_name") or "").strip()
        seat_no = (arguments.get("seat_no") or "").strip()
        room_id = arguments.get("room_id")

        # 0. 쿼리 또는 파라미터에서 스터디룸 의도 자동 보정
        is_study_intent = "스터디" in query or "스터디" in room_name or target in ["STUDY_ROOMS", "RESERVE_STUDY_ROOM"]

        # 1. 좌석 배정 신청 (RESERVE_SEAT)
        if target == "RESERVE_SEAT" or (seat_no and not is_study_intent):
            resolved_room_id = room_id or self._resolve_room_id(room_name) or 1
            seat_data = {
                "roomName": room_name or "제1열람실",
                "seatNo": seat_no,
                "roomId": resolved_room_id,
            }
            logger.info(f"[LibrarySeatTool] Emitting RESERVE_SEAT card: {seat_data}")
            return {
                "mode": "RESERVE_SEAT",
                "component_type": "LIBRARY_SEAT_CONFIRM",
                "data": seat_data,
                "link": {"label": "학산도서관 좌석 배정", "route": "/services/library"},
                "instruction": (
                    f"사용자가 [{seat_data['roomName']}] {seat_data['seatNo'] + '번 ' if seat_data['seatNo'] else ''}좌석 배정을 요청했습니다.\n"
                    f"대화창에 [학산도서관 좌석 배정 확인 카드]를 즉시 띄웠습니다.\n"
                    f"학생에게 아래 카드에서 좌석 상태를 확인하고 [확인 및 배정 신청하기] 버튼을 누르면 기기에서 즉시 배정이 완료된다고 친절하게 안내하세요.\n"
                    f"절대로 '앱이나 사이트에 직접 들어가서 신청하라'고 안내하지 마십시오. 대화창의 카드 버튼을 통해 앱이 즉시 배정을 처리합니다."
                ),
            }

        # 2. 스터디룸 예약 신청 (RESERVE_STUDY_ROOM)
        if target == "RESERVE_STUDY_ROOM":
            study_data = {
                "roomName": room_name or "205호 스터디룸",
                "roomId": room_id or 9,
                "date": arguments.get("date") or "",
                "beginTime": arguments.get("begin_time") or "",
                "endTime": arguments.get("end_time") or "",
            }
            return {
                "mode": "RESERVE_STUDY_ROOM",
                "component_type": "LIBRARY_STUDY_ROOM_CONFIRM",
                "data": study_data,
                "link": {"label": "학산도서관 스터디룸 예약", "route": "/services/library?tab=study"},
                "instruction": (
                    f"대화창에 [{study_data['roomName']}] 스터디룸 예약 확인 카드를 띄웠습니다.\n"
                    f"학생에게 아래 카드에서 날짜, 시간을 확인하고 [확인 및 예약 신청하기] 버튼을 누르면 기기에서 즉시 예약이 완료된다고 안내하세요."
                ),
            }

        # 3. 스터디룸 목록 조회 (STUDY_ROOMS)
        if target == "STUDY_ROOMS" or is_study_intent:
            return {
                "mode": "STUDY_ROOMS",
                "component_type": "LIBRARY_STUDY_ROOMS",
                "rooms": STUDY_ROOM_CATALOG,
                "data": {
                    "rooms": STUDY_ROOM_CATALOG,
                    "notice": "스터디룸 예약은 1회 최대 2시간 가능합니다. (이용 시작 20분 내 입실 필수)",
                },
                "link": {"label": "학산도서관 스터디룸", "route": "/services/library?tab=study"},
                "instruction": (
                    "예약 가능한 학산도서관 스터디룸 목록입니다. 아래 카드에서 원하는 방을 터치하면 바로 예약 신청을 진행할 수 있습니다."
                ),
            }

        # 4. 기본: 실시간 열람실 좌석 현황 조회 (SEATS)
        params = {"branchGroupId": 1, "smufMethodCode": "PC"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) INTIP-Agent/1.0",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=8.0, verify=False) as client:
                resp = await client.get(self.api_url, params=params, headers=headers)
                if resp.status_code != 200:
                    return {"error": f"도서관 좌석 서버 응답 오류 (HTTP {resp.status_code})"}

                data = resp.json()
                raw_list = data.get("data", {}).get("list", []) or []

                rooms: List[Dict[str, Any]] = []
                for item in raw_list:
                    r_name = item.get("name", "")
                    if room_name and room_name not in r_name:
                        continue

                    seats = item.get("seats") or {}
                    total = seats.get("total", 0)
                    occupied = seats.get("occupied", 0)
                    available = seats.get("available", 0)

                    rooms.append({
                        "id": item.get("id"),
                        "name": r_name,
                        "branch": (item.get("branch") or {}).get("name", "학산도서관"),
                        "isChargeable": item.get("isChargeable", True),
                        "total_seats": total,
                        "occupied_seats": occupied,
                        "available_seats": available,
                        "seats": {
                            "total": total,
                            "occupied": occupied,
                            "available": available,
                        },
                    })

                logger.info(f"[LibrarySeatTool] Successfully queried {len(rooms)} reading rooms.")
                return {
                    "mode": "SEATS",
                    "component_type": "LIBRARY_ROOMS",
                    "total_count": len(rooms),
                    "rooms": rooms,
                    "data": {"rooms": rooms},
                    "link": {"label": "학산도서관 좌석 배정", "route": "/services/library"},
                    "notice": "학산도서관 실시간 좌석 배정 시스템 기준 데이터입니다.",
                    "instruction": (
                        "실시간 열람실 좌석 현황을 안내하고, 대화창 아래 제공된 실시간 열람실 카드에서 원하는 열람실을 터치하면 바로 좌석 선택 및 배정 화면으로 이동할 수 있음을 친절히 덧붙이세요."
                    ),
                }
        except Exception as e:
            logger.error(f"Failed to query library seat rooms: {e}", exc_info=True)
            return {"error": f"도서관 실시간 좌석 정보를 조회하는 중 오류가 발생했습니다: {str(e)}"}
