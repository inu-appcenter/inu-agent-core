"""
Library Real-time Seat Availability Tool for inu-agent-core
Directly queries the public Pyxis API from https://lib.inu.ac.kr/pyxis-api/1/seat-rooms
without requiring student authentication or mobile app scraping.
"""
from typing import Dict, Any, List, Optional
import httpx

from app.tools.base import BaseTool
from app.core.logging import logger


class LibrarySeatTool(BaseTool):
    def __init__(self):
        self.name = "library_reading_rooms_status"
        self.description = "인천대학교 학술정보관(도서관) 열람실별 실시간 좌석 현황(전체 좌석, 사용 중 좌석, 잔여 좌석 수)을 실시간으로 조회합니다."
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
                        "room_name": {
                            "type": "string",
                            "description": "특정 열람실명 필터 (예: '제1열람실', '제2열람실', '노트북열람실'). 생략 시 전체 열람실 목록 조회",
                        }
                    },
                    "required": [],
                },
            },
        }

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch real-time seat availability from the public library pyxis endpoint.
        """
        params = {
            "branchGroupId": 1,
            "smufMethodCode": "PC",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        }

        filter_name = (arguments.get("room_name") or "").strip()

        try:
            async with httpx.AsyncClient(timeout=8.0, verify=False) as client:
                resp = await client.get(self.api_url, params=params, headers=headers)
                if resp.status_code != 200:
                    logger.warning(f"Library API returned status {resp.status_code}: {resp.text[:100]}")
                    return {"error": f"도서관 좌석 서버 응답 오류 (HTTP {resp.status_code})"}

                data = resp.json()
                raw_list = data.get("data", {}).get("list", []) or []

                rooms: List[Dict[str, Any]] = []
                for item in raw_list:
                    r_name = item.get("name", "")
                    if filter_name and filter_name not in r_name:
                        continue

                    seats = item.get("seats") or {}
                    total = seats.get("total", 0)
                    occupied = seats.get("occupied", 0)
                    available = seats.get("available", 0)
                    rate = round((occupied / total * 100), 1) if total > 0 else 0.0

                    rooms.append({
                        "id": item.get("id"),
                        "name": r_name,
                        "branch": (item.get("branch") or {}).get("name", "학술정보관"),
                        "total_seats": total,
                        "occupied_seats": occupied,
                        "available_seats": available,
                        "utilization_rate": f"{rate}%",
                        "is_chargeable": item.get("isChargeable", True),
                    })

                logger.info(f"Successfully retrieved {len(rooms)} library reading room statuses.")
                return {
                    "total_count": len(rooms),
                    "rooms": rooms,
                    "notice": "학술정보관 실시간 좌석 배정 시스템 기준 데이터입니다.",
                }
        except Exception as e:
            logger.error(f"Failed to query library seat rooms: {e}", exc_info=True)
            return {"error": f"도서관 실시간 좌석 정보를 조회하는 중 오류가 발생했습니다: {str(e)}"}
