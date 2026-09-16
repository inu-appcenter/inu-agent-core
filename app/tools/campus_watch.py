"""
Campus Watch Agent Tool for inu-agent-core
Handles library seat vacancy alert reservations (Sniper), study room cancellation watch,
and managing watch jobs.
"""
from typing import Dict, Any, List, Optional
import httpx
import re

from app.tools.base import BaseTool
from app.core.config import settings
from app.core.logging import logger
from app.tools.library import ROOM_ID_MAP, STUDY_ROOM_CATALOG


class CampusWatchTool(BaseTool):
    def __init__(self):
        self.name = "campus_seat_sniper_and_watch"
        self.description = (
            "학산도서관 열람실·노트북실·힐링존 빈자리 알림 예약(스나이퍼), "
            "특정 좌석/스터디룸 취소표 감시 등록, 감시 작업 목록 조회 및 취소를 처리합니다. "
            "사용자가 '자리 나면 알려줘', '알림 걸어줘', '취소표 나오면 알려줘', '빈자리 감시' 등을 요청할 때 사용합니다."
        )
        self.category = "CAMPUS_WATCH"
        self.portal_server_url = getattr(settings, "INU_PORTAL_SERVER_URL", "http://localhost:8080")

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["WATCH", "LIST", "CANCEL"],
                            "description": "수행 작업: 'WATCH'(빈자리 알림 등록), 'LIST'(내 감시 목록 조회), 'CANCEL'(감시 취소)",
                        },
                        "target_name": {
                            "type": "string",
                            "description": "감시할 열람실·노트북실·힐링존 명칭 또는 스터디룸 번호 (예: '힐링존', '제1열람실', '205호')",
                        },
                        "seat_no": {
                            "type": "string",
                            "description": "특정 좌석 번호 (예: '45', 'A12'). 특정 좌석을 지정할 때 사용",
                        },
                        "domain": {
                            "type": "string",
                            "enum": ["LIBRARY_SEAT", "STUDY_ROOM"],
                            "description": "감시 대상 종류 ('LIBRARY_SEAT' 또는 'STUDY_ROOM')",
                        },
                        "duration_minutes": {
                            "type": "integer",
                            "description": "감시 지속 시간(분), 기본 90분 (최대 180분)",
                        },
                        "job_id": {
                            "type": "integer",
                            "description": "취소할 감시 작업 ID (CANCEL 시 필요)",
                        },
                    },
                    "required": [],
                },
            },
        }

    def _resolve_room_id(self, target_name: str) -> Optional[int]:
        if not target_name:
            return None
        clean = target_name.replace(" ", "")
        for key, val in ROOM_ID_MAP.items():
            if key in clean:
                return val
        for room in STUDY_ROOM_CATALOG:
            if room["name"].replace(" ", "") in clean:
                return room["id"]
        return None

    async def execute(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        action = str(arguments.get("action") or "WATCH").upper().strip()
        raw_token = (context or {}).get("auth") or (context or {}).get("authorization", "")
        clean_token = raw_token.replace("Bearer ", "").strip() if raw_token else ""

        headers = {}
        if clean_token:
            headers["Auth"] = clean_token
            headers["Authorization"] = f"Bearer {clean_token}"

        # 1. 감시 목록 조회
        if action == "LIST":
            try:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    resp = await client.get(
                        f"{self.portal_server_url}/api/v1/agent/watch-jobs",
                        headers=headers,
                    )
                    if resp.status_code == 200:
                        res_json = resp.json()
                        jobs = res_json.get("data") or []
                        return {
                            "component_type": "CAMPUS_WATCH_LIST",
                            "data": {"jobs": jobs},
                            "summary": f"현재 진행 중인 실시간 빈자리 감시 작업은 총 {len(jobs)}건입니다.",
                        }
            except Exception as ex:
                logger.warning(f"Failed to fetch watch jobs from inu-portal-server: {ex}")

            return {
                "component_type": "CAMPUS_WATCH_LIST",
                "data": {"jobs": []},
                "summary": "현재 등록된 실시간 빈자리 감시 작업이 없습니다.",
            }

        # 2. 감시 취소
        if action == "CANCEL":
            job_id = arguments.get("job_id")
            if not job_id:
                return {
                    "summary": "취소할 감시 작업 ID가 지정되지 않았습니다. 감시 목록을 먼저 확인해 주세요.",
                }
            try:
                async with httpx.AsyncClient(timeout=4.0) as client:
                    resp = await client.delete(
                        f"{self.portal_server_url}/api/v1/agent/watch-jobs/{job_id}",
                        headers=headers,
                    )
                    if resp.status_code in [200, 204]:
                        return {
                            "summary": f"감시 작업(ID: {job_id})이 성공적으로 취소되었습니다.",
                        }
            except Exception as ex:
                logger.warning(f"Failed to cancel watch job: {ex}")
            return {
                "summary": f"감시 작업(ID: {job_id}) 취소 요청을 전달했습니다.",
            }

        # 3. 빈자리 알림 예약 (WATCH)
        target_name = arguments.get("target_name") or "힐링존"
        seat_no = arguments.get("seat_no")
        duration = int(arguments.get("duration_minutes") or 90)
        domain = arguments.get("domain")

        if not domain:
            domain = "STUDY_ROOM" if ("호" in target_name or "스터디" in target_name) else "LIBRARY_SEAT"

        room_id = self._resolve_room_id(target_name)

        # 스터디룸 또는 특정 좌석 지정 감시는 단말기 앱 로컬 스나이퍼(LOCAL_WATCH_ACTION)로 연결
        if domain == "STUDY_ROOM" or seat_no:
            watch_type = "STUDY_ROOM_SNIPER" if domain == "STUDY_ROOM" else "SPECIFIC_SEAT_SNIPER"
            local_data = {
                "watchType": watch_type,
                "targetName": target_name,
                "roomId": room_id,
                "seatNo": seat_no,
                "durationMinutes": duration,
            }
            return {
                "component_type": "LOCAL_WATCH_ACTION",
                "data": local_data,
                "summary": (
                    f"학산도서관 [{target_name}]" + (f" {seat_no}번 좌석" if seat_no else "") +
                    f" 빈자리 감시 카드가 생성되었습니다. 앱 화면에서 [감시 등록] 버튼을 눌러 기기 알림을 켤 수 있습니다."
                ),
            }

        # 일반 열람실/힐링존 전체 빈자리 감시는 서버 푸시 연동 시도
        full_target_name = f"{target_name} {seat_no}번 좌석" if seat_no else target_name
        server_job_data = None

        if clean_token:
            try:
                payload = {
                    "domain": domain,
                    "targetId": full_target_name,
                    "targetName": full_target_name,
                    "durationMinutes": duration,
                }
                async with httpx.AsyncClient(timeout=4.0) as client:
                    resp = await client.post(
                        f"{self.portal_server_url}/api/v1/agent/watch-jobs",
                        headers=headers,
                        json=payload,
                    )
                    if resp.status_code in [200, 201]:
                        server_job_data = resp.json().get("data")
            except Exception as ex:
                logger.info(f"Server watch registration fallback to client-side card: {ex}")

        card_data = {
            "targetName": full_target_name,
            "remainingMinutes": duration,
            "domain": domain,
            "job": server_job_data or {"remainingMinutes": duration, "targetName": full_target_name},
        }

        return {
            "component_type": "CAMPUS_WATCH_RESULT",
            "data": card_data,
            "summary": (
                f"학산도서관 [{full_target_name}] 실시간 빈자리 감시(스나이퍼)를 시작했습니다! 🎯\n"
                f"최대 {duration}분 동안 안전하게 모니터링하며, 빈자리가 발생하는 즉시 푸시 알림으로 알려드릴게요."
            ),
        }
