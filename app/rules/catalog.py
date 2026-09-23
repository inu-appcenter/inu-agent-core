"""
Pre-defined Action Rules Catalog for external campus systems (LMS, Library, Portal)
"""
from typing import Dict
from app.rules.models import ActionRule, RuleTarget, RuleExtraction, FieldExtractor, PrivacyPolicy

DEFAULT_ACTION_RULES: Dict[str, ActionRule] = {
    # -------------------------------------------------------------------------
    # LMS (이러닝 Moodle)
    # -------------------------------------------------------------------------
    "LMS_GET_COURSES": ActionRule(
        action_id="LMS_GET_COURSES",
        domain="LMS",
        protocol="HTTP_REST",
        title="수강 강좌 목록 조회",
        description="이러닝(LMS)에서 학생이 현재 수강 중인 강좌 목록을 조회합니다.",
        version="1.0.0",
        target=RuleTarget(
            url="https://lms.inu.ac.kr/webservice/rest/server.php",
            method="GET",
            params={
                "wsfunction": "core_enrol_get_users_courses",
                "moodlewsrestformat": "json",
            },
        ),
        default_card_type="LIST_CARD",
    ),
    "LMS_GET_UPCOMING_ASSIGNMENTS": ActionRule(
        action_id="LMS_GET_UPCOMING_ASSIGNMENTS",
        domain="LMS",
        protocol="HTTP_REST",
        title="다가오는 과제 및 일정 조회",
        description="이러닝(LMS)에서 마감 예정인 과제, 퀴즈, 시험 일정을 조회합니다.",
        version="1.0.0",
        target=RuleTarget(
            url="https://lms.inu.ac.kr/webservice/rest/server.php",
            method="GET",
            params={
                "wsfunction": "core_calendar_get_action_events_by_timesort",
                "moodlewsrestformat": "json",
                "limitnum": 10,
            },
        ),
        default_card_type="LIST_CARD",
    ),

    # -------------------------------------------------------------------------
    # 도서관 (학산도서관 Pyxis)
    # -------------------------------------------------------------------------
    "LIB_GET_READING_ROOMS": ActionRule(
        action_id="LIB_GET_READING_ROOMS",
        domain="LIBRARY",
        protocol="HTTP_REST",
        title="열람실 전체 좌석 현황",
        description="학산도서관 열람실별 잔여 좌석 수 및 운영 상태를 조회합니다.",
        version="1.0.0",
        target=RuleTarget(
            url="https://lib.inu.ac.kr/pyxis-api/1/seat-rooms",
            method="GET",
            params={"branchGroupId": 1, "smufMethodCode": "PC"},
        ),
        default_card_type="STATUS_CARD",
    ),
    "LIB_GET_MY_SEAT": ActionRule(
        action_id="LIB_GET_MY_SEAT",
        domain="LIBRARY",
        protocol="HTTP_REST",
        title="현재 이용 중인 도서관 좌석 조회",
        description="현재 학생이 배정받아 이용 중인 열람실 좌석 정보 및 잔여 시간을 조회합니다.",
        version="1.0.0",
        target=RuleTarget(
            url="https://lib.inu.ac.kr/pyxis-api/1/api/seat-charges",
            method="GET",
        ),
        default_card_type="STATUS_CARD",
    ),
    "LIB_RENEW_SEAT": ActionRule(
        action_id="LIB_RENEW_SEAT",
        domain="LIBRARY",
        protocol="HTTP_REST",
        title="도서관 좌석 연장",
        description="현재 이용 중인 열람실 좌석의 이용 시간을 연장 신청합니다.",
        version="1.0.0",
        target=RuleTarget(
            url="https://lib.inu.ac.kr/pyxis-api/1/api/seat-renewed-charges",
            method="POST",
            body_template={"smufMethodCode": "MOBILE"},
        ),
        default_card_type="ACTION_CARD",
    ),

    # -------------------------------------------------------------------------
    # 학교 포털 (종합정보시스템 Nexacro ERP)
    # -------------------------------------------------------------------------
    "PORTAL_GET_ACADEMIC_RECORD": ActionRule(
        action_id="PORTAL_GET_ACADEMIC_RECORD",
        domain="PORTAL",
        protocol="NEXACRO_SSV",
        title="학적 기본 정보 조회",
        description="포털 종합정보시스템(ERP)에서 학번, 전공, 이수 학기, 학적 상태를 조회합니다.",
        version="1.0.0",
        target=RuleTarget(
            url="/uni/sreg/TsimCtr/findBaseSchregInfoOne.do",
            method="POST",
            params={"menuId": "M002043", "pgmId": "P001878"},
            headers={"REQFOUNDATAION": "nexacro", "Content-Type": "text/plain; charset=UTF-8"},
        ),
        privacy=PrivacyPolicy(
            mask_fields=["studentNumber", "residentRegistrationNumber"],
            retention="TRANSIENT",
        ),
        default_card_type="METRIC_CARD",
    ),
    "PORTAL_GET_SEMESTER_GRADES": ActionRule(
        action_id="PORTAL_GET_SEMESTER_GRADES",
        domain="PORTAL",
        protocol="NEXACRO_SSV",
        title="학기별 성적 상세 조회",
        description="포털 종합정보시스템(ERP)에서 이번 학기 수강 과목 및 성적, 취득 학점을 조회합니다.",
        version="1.0.0",
        target=RuleTarget(
            url="/uni/grad/TsimCtr/findGradeDetailList.do",
            method="POST",
            params={"menuId": "M002044", "pgmId": "P001880"},
            headers={"REQFOUNDATAION": "nexacro", "Content-Type": "text/plain; charset=UTF-8"},
        ),
        privacy=PrivacyPolicy(
            mask_fields=["studentNumber"],
            retention="TRANSIENT",
        ),
        default_card_type="LIST_CARD",
    ),
    "PORTAL_GET_STUDENT_TIMETABLE": ActionRule(
        action_id="PORTAL_GET_STUDENT_TIMETABLE",
        domain="PORTAL",
        protocol="NEXACRO_SSV",
        title="학생별 수강 시간표 조회",
        description="포털 종합정보시스템(ERP)에서 수강 신청 교과목, 강의실, 강의 시간표 목록을 조회합니다.",
        version="1.0.0",
        target=RuleTarget(
            url="/uni/cour/CorrCtr/findStdSukangAplyList.do",
            method="POST",
            params={"menuId": "M003150", "pgmId": "P001416"},
            headers={"REQFOUNDATAION": "nexacro", "Content-Type": "text/plain; charset=UTF-8"},
            body_template={
                "dataset": "DS_COND",
                "columns": ["deptClsfCd", "yy", "tmGbn", "stuno", "korNm", "pageType"],
                "values": {
                    "deptClsfCd": "0000587",
                    "pageType": "sukang"
                }
            },
        ),
        privacy=PrivacyPolicy(
            mask_fields=["studentNumber"],
            retention="TRANSIENT",
        ),
        default_card_type="LIST_CARD",
    ),
}
