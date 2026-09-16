"""
Notification and Settings Agent Tools for inu-agent-core
Migrated from inu-portal-server actions:
- ACTION_MANAGE_REMINDER
- ACTION_DAILY_BRIEF
- ACTION_NOTICE_KEYWORD
- ACTION_MY_SETTINGS
"""
import json
import logging
from typing import Any, Dict, List, Optional
import httpx

from app.core.config import settings
from app.tools.base import BaseTool

logger = logging.getLogger("inu-agent-core.tools.notification")


class ManageReminderTool(BaseTool):
    """
    맞춤 푸시 알림 예약 및 관리 도구 (ACTION_MANAGE_REMINDER)
    """
    def __init__(self):
        self.name = "action_manage_reminder"
        self.description = "특정 시각에 캠퍼스 정보(학식, 날씨, 버스 등)를 보내는 맞춤 푸시 알림을 생성, 조회, 수정, 삭제합니다."
        self.category = "REMINDER"
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["CREATE", "LIST", "DELETE", "TOGGLE"],
                    "description": "수행할 작업 (CREATE: 알림 생성, LIST: 목록 조회, DELETE: 삭제, TOGGLE: On/Off 변경)"
                },
                "targetTime": {
                    "type": "string",
                    "description": "알림 시각 (HH:mm 형식, 예: '11:30', '08:00')"
                },
                "targetTool": {
                    "type": "string",
                    "enum": ["CAFETERIA", "WEATHER", "BUS", "NOTICE"],
                    "description": "알림으로 발송할 대상 데이터 도구"
                },
                "title": {
                    "type": "string",
                    "description": "알림 제목 (예: '점심 학식 알림', '등교 버스 알림')"
                },
                "repeatType": {
                    "type": "string",
                    "enum": ["WEEKDAYS", "EVERYDAY", "ONCE"],
                    "description": "반복 주기 (WEEKDAYS: 평일, EVERYDAY: 매일, ONCE: 1회)"
                },
                "reminderId": {
                    "type": "integer",
                    "description": "삭제 또는 토글할 알림 ID"
                },
                "enabled": {
                    "type": "boolean",
                    "description": "토글 시 활성화 여부"
                }
            },
            "required": ["action"]
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            }
        }

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Any:
        base_url = settings.INU_PORTAL_SERVER_URL.rstrip("/")
        action = arguments.get("action", "LIST").upper()
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "inu-agent-core/0.1.0",
        }
        raw_token = context.get("auth") or context.get("authorization", "")
        clean_token = raw_token.replace("Bearer ", "").strip() if raw_token else ""
        if clean_token:
            headers["Auth"] = clean_token
            headers["Authorization"] = f"Bearer {clean_token}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                if action == "LIST":
                    res = await client.get(f"{base_url}/api/agent/reminders", headers=headers)
                elif action == "DELETE":
                    rem_id = arguments.get("reminderId")
                    if not rem_id:
                        return {"error": "삭제할 알림 ID(reminderId)가 필요합니다."}
                    res = await client.delete(f"{base_url}/api/agent/reminders/{rem_id}", headers=headers)
                elif action == "TOGGLE":
                    rem_id = arguments.get("reminderId")
                    enabled = arguments.get("enabled", True)
                    if not rem_id:
                        return {"error": "토글할 알림 ID(reminderId)가 필요합니다."}
                    res = await client.patch(
                        f"{base_url}/api/agent/reminders/{rem_id}/toggle",
                        params={"enabled": enabled},
                        headers=headers
                    )
                else:
                    res = await client.get(f"{base_url}/api/agent/reminders", headers=headers)
                
                if res.status_code >= 400:
                    return {"error": f"서버 응답 오류 (Status: {res.status_code})", "action": action}
                
                data = res.json()
                return data.get("data", data)
            except Exception as ex:
                logger.warning(f"ManageReminderTool failed: {ex}")
                return {"error": str(ex), "action": action}


class DailyBriefTool(BaseTool):
    """
    데일리 브리프 설정 도구 (ACTION_DAILY_BRIEF)
    """
    def __init__(self):
        self.name = "action_daily_brief"
        self.description = "매일 아침 시간표 및 학사일정을 요약해 보내주는 데일리 브리프의 수신 시각, On/Off, 일정 범위를 조회하거나 변경합니다."
        self.category = "DAILY_BRIEF"
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["GET", "UPDATE"],
                    "description": "GET: 현재 설정 조회, UPDATE: 설정 변경"
                },
                "time": {
                    "type": "string",
                    "description": "수신 시각 (HH:mm 형식, 예: '08:30', '08:00')"
                },
                "enabled": {
                    "type": "boolean",
                    "description": "데일리 브리프 활성화 여부"
                },
                "scope": {
                    "type": "string",
                    "enum": ["ALL", "SCHOOL_ONLY", "DEPT_ONLY"],
                    "description": "일정 브리핑 범위 (ALL: 전체, SCHOOL_ONLY: 학교만, DEPT_ONLY: 학과만)"
                }
            },
            "required": ["action"]
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            }
        }

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Any:
        base_url = settings.INU_PORTAL_SERVER_URL.rstrip("/")
        action = arguments.get("action", "GET").upper()
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "inu-agent-core/0.1.0",
        }
        raw_token = context.get("auth") or context.get("authorization", "")
        clean_token = raw_token.replace("Bearer ", "").strip() if raw_token else ""
        if clean_token:
            headers["Auth"] = clean_token
            headers["Authorization"] = f"Bearer {clean_token}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                if action == "UPDATE":
                    payload = {}
                    if "time" in arguments:
                        payload["time"] = arguments["time"]
                    if "enabled" in arguments:
                        payload["enabled"] = arguments["enabled"]
                    if "scope" in arguments:
                        payload["scope"] = arguments["scope"]
                    res = await client.put(f"{base_url}/api/daily-brief/settings", json=payload, headers=headers)
                else:
                    res = await client.get(f"{base_url}/api/daily-brief/settings", headers=headers)

                if res.status_code >= 400:
                    return {"error": f"서버 응답 오류 (Status: {res.status_code})"}
                
                data = res.json()
                return data.get("data", data)
            except Exception as ex:
                logger.warning(f"DailyBriefTool failed: {ex}")
                return {"error": str(ex)}


class NoticeKeywordTool(BaseTool):
    """
    공지 키워드 알림 구독 도구 (ACTION_NOTICE_KEYWORD)
    """
    def __init__(self):
        self.name = "action_notice_keyword"
        self.description = "새로운 공지사항이 올라올 때 알림을 받을 키워드를 등록, 조회, 삭제합니다."
        self.category = "KEYWORD"
        self.parameters_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["ADD", "LIST", "DELETE"],
                    "description": "ADD: 키워드 등록, LIST: 등록된 키워드 목록, DELETE: 키워드 삭제"
                },
                "keyword": {
                    "type": "string",
                    "description": "등록할 공지 키워드 (예: '장학금', '근로장학생', '수강신청')"
                },
                "category": {
                    "type": "string",
                    "description": "특정 카테고리 한정 (예: '장학금', '학사', '모집')"
                },
                "isExcluded": {
                    "type": "boolean",
                    "description": "제외 키워드 여부 (기본값 false)"
                },
                "keywordId": {
                    "type": "integer",
                    "description": "삭제할 키워드 ID"
                }
            },
            "required": ["action"]
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            }
        }

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Any:
        base_url = settings.INU_PORTAL_SERVER_URL.rstrip("/")
        action = arguments.get("action", "LIST").upper()
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "inu-agent-core/0.1.0",
        }
        raw_token = context.get("auth") or context.get("authorization", "")
        clean_token = raw_token.replace("Bearer ", "").strip() if raw_token else ""
        if clean_token:
            headers["Auth"] = clean_token
            headers["Authorization"] = f"Bearer {clean_token}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                if action == "ADD":
                    kw = arguments.get("keyword", "").strip()
                    if not kw:
                        return {"error": "등록할 키워드가 없습니다."}
                    params = {"keyword": kw, "isExcluded": arguments.get("isExcluded", False)}
                    if "category" in arguments:
                        params["category"] = arguments["category"]
                    res = await client.post(f"{base_url}/api/keywords", params=params, headers=headers)
                elif action == "DELETE":
                    kw_id = arguments.get("keywordId")
                    if not kw_id:
                        return {"error": "삭제할 키워드 ID(keywordId)가 필요합니다."}
                    res = await client.delete(f"{base_url}/api/keywords/{kw_id}", headers=headers)
                else:
                    res = await client.get(f"{base_url}/api/keywords", headers=headers)

                if res.status_code >= 400:
                    return {"error": f"서버 응답 오류 (Status: {res.status_code})"}
                
                data = res.json()
                return data.get("data", data)
            except Exception as ex:
                logger.warning(f"NoticeKeywordTool failed: {ex}")
                return {"error": str(ex)}


class MySettingsTool(BaseTool):
    """
    내 맞춤 설정 종합 조회 도구 (ACTION_MY_SETTINGS)
    """
    def __init__(self):
        self.name = "action_my_settings"
        self.description = "사용자가 등록한 공지 키워드 목록, 맞춤 알림 목록, 데일리 브리프 설정을 한 번에 종합 조회합니다."
        self.category = "SETTINGS"
        self.parameters_schema = {
            "type": "object",
            "properties": {},
        }

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            }
        }

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> Any:
        base_url = settings.INU_PORTAL_SERVER_URL.rstrip("/")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "inu-agent-core/0.1.0",
        }
        raw_token = context.get("auth") or context.get("authorization", "")
        clean_token = raw_token.replace("Bearer ", "").strip() if raw_token else ""
        if clean_token:
            headers["Auth"] = clean_token
            headers["Authorization"] = f"Bearer {clean_token}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                kw_res = await client.get(f"{base_url}/api/keywords", headers=headers)
                rem_res = await client.get(f"{base_url}/api/agent/reminders", headers=headers)
                brief_res = await client.get(f"{base_url}/api/daily-brief/settings", headers=headers)

                keywords = kw_res.json().get("data", []) if kw_res.status_code == 200 else []
                reminders = rem_res.json().get("data", []) if rem_res.status_code == 200 else []
                brief = brief_res.json().get("data", {}) if brief_res.status_code == 200 else {}

                return {
                    "keywords": keywords,
                    "reminders": reminders,
                    "dailyBrief": brief,
                }
            except Exception as ex:
                logger.warning(f"MySettingsTool failed: {ex}")
                return {"error": str(ex)}