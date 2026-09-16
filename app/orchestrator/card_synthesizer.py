"""
Card Synthesizer Module
Synthesizes rich Universal SDUI Cards (ListCard, MetricCard, StatusCard, ActionCard)
based on executed tool data or domain intents.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, timedelta
from app.llm.schemas import (
    GenerativeCard,
    ListCard,
    ListItem,
    MetricCard,
    MetricCardItem,
    StatusCard,
    CardBadge,
    ComponentCard,
    CardLink,
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
        elif domain == "LMS":
            return cls._build_lms_card(data)
        elif domain == "DIRECTORY":
            return cls._build_directory_card(data)
        elif domain == "WEATHER":
            return cls._build_weather_card(data)
        elif domain == "LIBRARY":
            return cls._build_library_card(data)
        elif domain in ["PORTAL", "ACADEMIC"]:
            return cls._build_academic_card(data)
        elif domain in ["INU_AI_KNOWLEDGE", "INU_AI", "CITATION"]:
            return cls._build_inuchat_citation_card(data)

        return None

    @classmethod
    def _build_bus_card(cls, data: Optional[Any] = None) -> Optional[ListCard]:
        if not data:
            return None
        items: List[ListItem] = []

        arrivals = []
        valid_routes = []
        stop_name = "인천대입구역"

        if isinstance(data, dict):
            arrivals = data.get("arrivals") or []
            valid_routes = data.get("validRoutes") or []
            stop_name = data.get("stopName") or data.get("tabName") or "버스 정류장"
        elif isinstance(data, list):
            arrivals = data

        if arrivals:
            for item in arrivals[:4]:
                if isinstance(item, dict):
                    route = item.get("routeNo") or item.get("routeName") or "버스"
                    route_display = f"{route}번 버스" if not str(route).endswith("번") and str(route).replace("순환", "").isdigit() else f"{route} 버스"
                    time_raw = item.get("arrivalEstimateTime") or item.get("time")
                    rest = item.get("restStopCount")
                    
                    mins = 0
                    try:
                        if time_raw and str(time_raw).isdigit():
                            mins = int(time_raw) // 60
                    except Exception:
                        pass

                    if mins > 0:
                        sub = f"약 {mins}분 후 도착" + (f" ({rest}개 정류소 전)" if rest else "")
                    elif rest:
                        sub = f"{rest}개 정류소 전 (곧 도착)"
                    else:
                        sub = "곧 도착 예정"

                    items.append(
                        ListItem(
                            title=route_display,
                            subtitle=sub,
                            tag="실시간 도착",
                            link="/bus",
                        )
                    )
        elif valid_routes:
            for r in valid_routes[:4]:
                r_display = f"{r}번 버스" if not str(r).endswith("번") and str(r).replace("순환", "").isdigit() else f"{r} 버스"
                items.append(
                    ListItem(
                        title=r_display,
                        subtitle="현재 운행 대기 또는 도착 정보 없음",
                        tag="운행 정보",
                        link="/bus",
                    )
                )

        if not items:
            return None

        footer = "인팁(INTIP) 공식 등하교 버스 노선 기준" if arrivals else "인팁(INTIP) 공식 등하교 버스 노선 기준 (현재 도착 예정 버스 없음)"
        return ListCard(
            title=f"🚌 {stop_name} 버스 도착 정보",
            items=items,
            footer_text=footer,
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
                title = cit.get("title") or "인천대학교 공식 공지/학칙 원문"
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
                                link="/home/menu",
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
                                link="/home/menu",
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
                        link="/timetable",
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
                notice_url = n.get("url") or n.get("link")
                if not notice_url and n.get("id"):
                    notice_url = f"/notice/{n.get('id')}"

                items.append(
                    ListItem(
                        title=str(title),
                        subtitle=str(date_val),
                        tag=str(cat),
                        link=notice_url,
                    )
                )

        if not items:
            return None

        return ListCard(
            title="📢 인천대학교 최신 공지사항",
            items=items,
            footer_text="공지사항을 클릭하면 상세 내용으로 연결됩니다.",
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
                        link="/home/calendar",
                    )
                )

        if not items:
            return None

    @classmethod
    def _build_lms_card(cls, data: Optional[Any] = None) -> Optional[GenerativeCard]:
        if not data:
            return None

        # INTIP 웹 전용 고유 대화형 카드 (LmsAssignmentsCard) 우선 생성
        if isinstance(data, dict) and ("events" in data or "assignments" in data or "courses" in data):
            return ComponentCard(
                type="LMS_ASSIGNMENTS",
                data=data,
                link=CardLink(label="이러닝(LMS) 바로가기", route="https://lms.inu.ac.kr"),
            )

        items: List[ListItem] = []
        events = []
        courses = []
        if isinstance(data, dict):
            events = data.get("events") or data.get("assignments") or []
        elif isinstance(data, list):
            events = data

        for ev in events[:5]:
            if isinstance(ev, dict):
                ev_name = ev.get("name") or ev.get("title") or "과제"
                course_info = ev.get("course")
                c_name = course_info.get("fullname") if isinstance(course_info, dict) else str(course_info or "")

                # 마감 일시 정밀 파싱 (formattedtime 우선, 그 다음 Unix timestamp)
                due = ev.get("formattedtime")
                if not due:
                    ts = ev.get("timesort") or ev.get("timestart")
                    if isinstance(ts, (int, float)) and ts > 0:
                        try:
                            kst = timezone(timedelta(hours=9))
                            due = datetime.fromtimestamp(ts, tz=kst).strftime("%m월 %d일 %H:%M")
                        except Exception:
                            due = str(ts)
                if not due:
                    due = ev.get("timedue") or ev.get("dueDate") or ev.get("date") or "마감 예정"

                sub = f"{c_name} · {due}" if c_name else str(due)
                direct_url = ev.get("url") or (ev.get("action") or {}).get("url") or "https://lms.inu.ac.kr"

                items.append(
                    ListItem(
                        title=str(ev_name),
                        subtitle=sub,
                        tag="과제",
                        link=str(direct_url),
                    )
                )

        if not items and courses:
            for c in courses[:5]:
                if isinstance(c, dict):
                    c_name = c.get("fullname") or c.get("name") or "수강 강좌"
                    c_id = c.get("id")
                    c_url = f"https://lms.inu.ac.kr/course/view.php?id={c_id}" if c_id else "https://lms.inu.ac.kr"
                    items.append(
                        ListItem(
                            title=str(c_name),
                            subtitle="현재 수강 중인 강좌",
                            tag="강좌",
                            link=c_url,
                        )
                    )

        if not items:
            return None

        return ListCard(
            title="📝 이러닝(LMS) 과제 및 일정",
            items=items,
            footer_text="과제 제출 및 온라인 강의 수강은 이러닝 시스템에서 진행할 수 있습니다.",
        )

    @classmethod
    def _build_directory_card(cls, data: Optional[Any] = None) -> Optional[ListCard]:
        if not data:
            return None
        items: List[ListItem] = []
        contacts = data if isinstance(data, list) else (
            data.get("contents") or data.get("items") or data.get("contacts") or []
        )

        for c in contacts[:4]:
            if isinstance(c, dict):
                # College office contact vs individual directory entry
                dept_name = c.get("departmentName") or c.get("deptName")
                indiv_name = c.get("name")
                name = dept_name or indiv_name or "연락처"

                phone = c.get("officePhoneNumber") or c.get("phoneNumber") or c.get("phone") or c.get("tel") or ""
                location = c.get("officeLocation") or c.get("office") or c.get("location") or ""
                college = c.get("collegeName")
                position = c.get("position")
                detail_aff = c.get("detailAffiliation") or c.get("affiliation")

                tag = (college or position or detail_aff or "연락처")[:8]
                sub_parts = []
                if phone:
                    sub_parts.append(f"📞 {phone}")
                if location:
                    sub_parts.append(f"🏢 {location}")
                if c.get("email"):
                    sub_parts.append(f"✉️ {c.get('email')}")

                sub = " | ".join(sub_parts) if sub_parts else "연락처 정보"
                link_target = f"tel:{phone}" if phone else c.get("homepageUrl")

                items.append(
                    ListItem(
                        title=str(name),
                        subtitle=sub,
                        tag=str(tag),
                        link=link_target,
                    )
                )

        if not items:
            return None

        return ListCard(
            title="📞 교내 전화번호부 및 연락처",
            items=items,
            footer_text="전화번호 및 위치 정보를 확인하세요.",
        )

    @classmethod
    def _build_weather_card(cls, data: Optional[Any] = None) -> Optional[MetricCard]:
        if not data or not isinstance(data, dict):
            return None
        temp = data.get("temp") or data.get("temperature") or "--°C"
        sky = data.get("sky") or data.get("condition") or "맑음"
        pm10 = data.get("pm10") or data.get("airQuality") or "보통"
        rain = data.get("rain") or data.get("precipitation") or "0mm"

        return MetricCard(
            title="⛅ 송도 캠퍼스 실시간 날씨",
            main_metric=MetricCardItem(
                label="현재 기온",
                value=f"{temp}",
            ),
            sub_details=[
                MetricCardItem(label="하늘 상태", value=str(sky)),
                MetricCardItem(label="미세먼지", value=str(pm10)),
                MetricCardItem(label="강수량", value=str(rain)),
            ],
            footer_text="기상청 실시간 송도 캠퍼스 관측 데이터 기반",
        )

    @classmethod
    def _build_library_card(cls, data: Optional[Any] = None) -> Optional[GenerativeCard]:
        if not data:
            return None

        # 1. INTIP 전용 대화형 컴포넌트 카드 (LIBRARY_SEAT_CONFIRM, LIBRARY_STUDY_ROOM_CONFIRM 등)
        if isinstance(data, dict):
            comp_type = data.get("component_type")
            if comp_type:
                comp_data = data.get("data") or {}
                link_info = data.get("link") or {"label": "학산도서관 좌석 배정", "route": "/services/library"}
                card_title = "학산도서관 좌석 배정 신청 확인" if comp_type == "LIBRARY_SEAT_CONFIRM" else "학산도서관 스터디룸 예약 확인" if comp_type == "LIBRARY_STUDY_ROOM_CONFIRM" else "학산도서관"
                return ComponentCard(
                    type=comp_type,
                    title=card_title,
                    data=comp_data,
                    link=CardLink(label=link_info.get("label", "도서관"), route=link_info.get("route", "/services/library")),
                )

        # 2. 열람실 잔여 좌석 목록 컴포넌트 카드 (LIBRARY_ROOMS)
        raw_rooms = []
        if isinstance(data, dict):
            raw_rooms = data.get("rooms") or data.get("list") or []
        elif isinstance(data, list):
            raw_rooms = data

        if raw_rooms:
            # INTIP 고유 LibraryRoomsCard 규격 지원 (터치 시 좌석 선택 및 배정 연결)
            return ComponentCard(
                type="LIBRARY_ROOMS",
                title="학산도서관 실시간 열람실 좌석",
                data={"rooms": raw_rooms},
                link=CardLink(label="학산도서관 좌석 배정", route="/services/library"),
            )

        return None

    @classmethod
    def _build_academic_card(cls, data: Optional[Any] = None) -> Optional[GenerativeCard]:
        if not data or not isinstance(data, dict):
            return None

        # 프론트엔드/모바일/웹의 다양한 프로퍼티 명(카멜, 축약형 등)을 모두 지원하도록 정규화
        name = data.get("koreanName") or data.get("name") or data.get("studentName") or data.get("korNm") or "학우님"
        student_id = data.get("studentId") or data.get("id") or data.get("stdNo") or (f"{data.get('entryYear')}학번" if data.get("entryYear") else "")
        dept = data.get("departmentName") or data.get("department") or data.get("dept") or data.get("major") or data.get("deptName") or ""
        college = data.get("collegeName") or data.get("colgNm") or ""
        status = data.get("enrollmentStatus") or data.get("status") or data.get("academicStatus") or "재학"
        change = data.get("latestEnrollmentChange") or data.get("flSchregModGbn") or ""
        semester = data.get("completedSemesterCount") or data.get("completedSemesterName") or (f"{data.get('grade')}학년" if data.get("grade") else "")
        credits = data.get("acquiredCredits") or data.get("totalCredits") or data.get("credits") or ""
        gpa = data.get("gradeAverage") or data.get("gpa") or data.get("mrksAvg") or ""
        advisor = data.get("advisorProfessorName") or data.get("advisor") or data.get("profNm") or ""

        normalized_data = {
            **data,
            # 표준 카멜케이스
            "koreanName": name,
            "studentId": student_id,
            "departmentName": dept,
            "collegeName": college,
            "enrollmentStatus": status,
            "latestEnrollmentChange": change,
            "completedSemesterCount": semester,
            "acquiredCredits": credits,
            "gradeAverage": gpa,
            "advisorProfessorName": advisor,
            # 프론트엔드 축약형 호환
            "name": name,
            "department": dept,
            "status": status,
            "credits": credits,
            "totalCredits": credits,
            "grade": semester,
            "advisor": advisor,
        }

        # INTIP 전용 학적 카드 컴포넌트 (AcademicInfoCard) 생성
        return ComponentCard(
            type="ACADEMIC_INFO",
            title="학적 기본 정보",
            data=normalized_data,
            link=CardLink(label="학적 정보 상세보기", route="/mypage"),
        )
