"""
Card Synthesizer Module
Synthesizes rich Universal SDUI Cards (ListCard, MetricCard, StatusCard, ActionCard)
based on executed tool data or domain intents.
"""
from typing import Optional, Dict, Any, List
from app.llm.schemas import (
    GenerativeCard,
    ListCard,
    ListItem,
    MetricCard,
    MetricCardItem,
    StatusCard,
    CardBadge,
)
from app.core.logging import logger


class CardSynthesizer:
    @classmethod
    def synthesize_for_domain(
        cls,
        domain: str,
        tool_name: str = "",
        data: Optional[Any] = None,
        query: str = "",
    ) -> Optional[GenerativeCard]:
        """
        Synthesize an SDUI card based on tool execution data or domain context.
        """
        domain = domain.upper()

        if domain == "BUS":
            return cls._build_bus_card(data)
        elif domain == "CAFETERIA":
            return cls._build_cafeteria_card(data)
        elif domain == "TIMETABLE":
            return cls._build_timetable_card(data)
        elif domain == "NOTICE":
            return cls._build_notice_card(data)
        elif domain == "SCHEDULE":
            return cls._build_schedule_card(data)
        elif domain in ["INU_AI_KNOWLEDGE", "INU_AI", "CITATION"]:
            return cls._build_inuchat_citation_card(data)

        return None

    @classmethod
    def _build_bus_card(cls, data: Optional[Any] = None) -> Optional[ListCard]:
        if not data:
            return None
        items: List[ListItem] = []

        if isinstance(data, list):
            for item in data[:4]:
                if isinstance(item, dict):
                    name = item.get("routeName") or item.get("routeNo") or item.get("lineName") or "버스 노선"
                    eta = item.get("estimatedMinutes") or item.get("eta") or item.get("restStopCount") or "운행중"
                    stop = item.get("currentStation") or item.get("stopName") or ""
                    sub = f"{stop} · {eta}" if stop else str(eta)
                    items.append(
                        ListItem(
                            title=str(name),
                            subtitle=sub,
                            tag="실시간",
                        )
                    )
        elif isinstance(data, dict):
            arrivals = data.get("arrivals") or data.get("sections") or data.get("historyRecords") or []
            stop_name = data.get("stopName") or data.get("tabName") or ""
            for item in arrivals[:4]:
                if isinstance(item, dict):
                    route = item.get("routeNo") or item.get("routeName") or item.get("sectionName") or "노선"
                    time_val = item.get("arrivalEstimateTime") or item.get("time") or item.get("eta") or "운행중"
                    rest = item.get("restStopCount")
                    sub = f"{rest}개 전 ({time_val})" if rest else str(time_val)
                    items.append(
                        ListItem(
                            title=f"{route}번" if str(route).isdigit() else str(route),
                            subtitle=sub,
                            tag="도착예정",
                        )
                    )

        if not items:
            return None

        return ListCard(
            title="🚌 실시간 버스 도착 정보",
            items=items,
            footer_text="실시간 교통 및 신호 상태에 따라 도착 시간에 오차가 발생할 수 있습니다.",
        )

    @classmethod
    def _build_inuchat_citation_card(cls, data: Optional[Any] = None) -> Optional[ListCard]:
        if not data:
            return None
        citations = []
        if isinstance(data, dict):
            citations = data.get("citations") or []

        if not citations:
            return None

        items: List[ListItem] = []
        for cit in citations[:4]:
            if isinstance(cit, dict):
                title = cit.get("title", "관련 학칙/원문 바로가기")
                url = cit.get("url", "https://www.inu.ac.kr")
                tag = "공식학칙" if cit.get("type") == "LAW" else "공지출처"
                items.append(
                    ListItem(
                        title=title,
                        subtitle=url,
                        tag=tag,
                        link=url,
                    )
                )

        return ListCard(
            title="📜 INUChat 학사 지식베이스 공식 출처 (Citations)",
            items=items,
            footer_text="인천대학교 공식 규정집 및 학사 공지 원문을 기반으로 검증되었습니다.",
        )

    @classmethod
    def _build_cafeteria_card(cls, data: Optional[Any] = None) -> Optional[ListCard]:
        if not data:
            return None
        items: List[ListItem] = []

        if isinstance(data, list):
            for m in data[:5]:
                if isinstance(m, dict):
                    corner = m.get("name") or m.get("cornerName") or m.get("restaurant") or "식당"
                    menu_text = m.get("menu") or m.get("menuName") or ""
                    if menu_text and menu_text != "-":
                        items.append(
                            ListItem(
                                title=str(corner),
                                subtitle=str(menu_text).replace("\n", " | "),
                                tag=m.get("mealLabel", "식단"),
                            )
                        )
        elif isinstance(data, dict):
            caf_list = data.get("cafeterias") or data.get("menus") or []
            target_meal = data.get("targetMeal") or data.get("mealLabel") or "메뉴"
            for m in caf_list[:5]:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("corner") or "식당"
                    menu = m.get("menu") or m.get("name") or ""
                    if menu and menu != "-":
                        items.append(
                            ListItem(
                                title=str(name),
                                subtitle=str(menu).replace("\n", " | "),
                                tag=target_meal,
                            )
                        )

        if not items:
            return None

        return ListCard(
            title="🍲 오늘의 학식 식단표",
            items=items,
            footer_text="식당 운영 상황에 따라 조기 품절될 수 있습니다.",
        )

    @classmethod
    def _build_timetable_card(cls, data: Optional[Any] = None) -> Optional[ListCard]:
        if not data:
            return None
        items: List[ListItem] = []
        lectures = data if isinstance(data, list) else data.get("todayClasses", [])

        for lecture in lectures[:4]:
            if isinstance(lecture, dict):
                name = lecture.get("courseName") or lecture.get("subject") or "강의"
                time_str = lecture.get("time") or lecture.get("classTime") or ""
                room = lecture.get("classroom") or lecture.get("room") or ""
                items.append(
                    ListItem(
                        title=str(name),
                        subtitle=f"{time_str} ({room})".strip(),
                        tag="강의",
                    )
                )

        if not items:
            return None

        return ListCard(
            title="🗓️ 나의 수업 시간표",
            items=items,
            footer_text="수업 강의실 및 시간을 확인하세요.",
        )

    @classmethod
    def _build_notice_card(cls, data: Optional[Any] = None) -> Optional[ListCard]:
        if not data:
            return None
        items: List[ListItem] = []
        notices = data if isinstance(data, list) else data.get("contents", data.get("notices", []))

        for n in notices[:4]:
            if isinstance(n, dict):
                title = n.get("title") or "공지사항"
                date_val = n.get("createDate") or n.get("date") or ""
                cat = n.get("category") or n.get("subCategory") or "공지"
                items.append(
                    ListItem(
                        title=str(title),
                        subtitle=str(date_val),
                        tag=str(cat),
                    )
                )

        if not items:
            return None

        return ListCard(
            title="📢 인천대학교 최신 공지사항",
            items=items,
            footer_text="상세 내용은 인천대학교 포털에서 확인하실 수 있습니다.",
        )

    @classmethod
    def _build_schedule_card(cls, data: Optional[Any] = None) -> Optional[ListCard]:
        if not data:
            return None
        items: List[ListItem] = []
        schedules = data if isinstance(data, list) else data.get("schedules", [])

        for s in schedules[:4]:
            if isinstance(s, dict):
                title = s.get("title") or "학사일정"
                start = s.get("start") or s.get("startDate") or ""
                end = s.get("end") or s.get("endDate") or ""
                date_str = f"{start} ~ {end}" if end and end != start else str(start)
                items.append(
                    ListItem(
                        title=str(title),
                        subtitle=date_str,
                        tag="일정",
                    )
                )

        if not items:
            return None

        return ListCard(
            title="📅 주요 학사일정",
            items=items,
            footer_text="일정은 학사 운영 상황에 따라 변경될 수 있습니다.",
        )

