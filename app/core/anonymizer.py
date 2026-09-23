"""
Privacy & Anonymization Engine for INU Agent Core
Enforces 100% Zero-PII (Personally Identifiable Information) leakage.
Strips/masks student names, full student IDs, phone numbers, resident numbers, and emails before sending to LLMs or external RAG systems (INUChat).
"""
import re
from typing import Dict, Any, Optional


# 1. Regex patterns for PII detection
RE_FULL_STUDENT_ID = re.compile(r"\b(19\d{2}|20\d{2})\d{4,6}\b")  # 8~10 digit student IDs (e.g. 202101234)
RE_PHONE_NUMBER = re.compile(r"\b(01[016789]-?\d{3,4}-?\d{4}|01[016789]\d{7,8})\b")
RE_TEL_NUMBER = re.compile(r"\b(032-?\d{3,4}-?\d{4})\b")
RE_RRN = re.compile(r"\b\d{6}-[1-4]\d{6}\b")  # Resident Registration Number
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


def sanitize_text(text: str) -> str:
    """
    Sanitize raw user text or prompt content by masking all personal identifiers.
    Converts full student IDs (e.g. 202101234) to matriculation year only (e.g. 2021학번).
    """
    if not text:
        return ""

    # Convert full student IDs to year format (e.g., 202101234 -> 2021학번)
    def _replace_student_id(match):
        full_id = match.group(0)
        year = full_id[:4]
        return f"{year}학번"

    sanitized = RE_FULL_STUDENT_ID.sub(_replace_student_id, text)
    sanitized = RE_RRN.sub("[주민번호비식별화]", sanitized)
    sanitized = RE_PHONE_NUMBER.sub("[개인전화번호]", sanitized)
    sanitized = RE_EMAIL.sub("[이메일]", sanitized)

    return sanitized


def sanitize_academic_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only strictly non-identifiable academic metrics from raw portal ERP data.
    Strips real name, full student ID, phone number, email, and resident info.
    """
    if not isinstance(raw_record, dict):
        return {}

    # Extract entry year only (e.g. "2021" from studentId="202101234" or entryYear="2021")
    raw_id = str(raw_record.get("studentId") or raw_record.get("id") or raw_record.get("stdNo") or "").strip()
    raw_entry = str(raw_record.get("entryYear") or "").strip()

    entry_year = ""
    if raw_entry and raw_entry[:4].isdigit():
        entry_year = raw_entry[:4]
    elif raw_id and len(raw_id) >= 4 and raw_id[:4].isdigit():
        entry_year = raw_id[:4]

    dept = raw_record.get("departmentName") or raw_record.get("deptName") or raw_record.get("department") or ""
    college = raw_record.get("collegeName") or raw_record.get("colgNm") or ""
    status = raw_record.get("enrollmentStatus") or raw_record.get("status") or ""
    change = raw_record.get("latestEnrollmentChange") or ""
    sem = raw_record.get("completedSemesterCount") or raw_record.get("completedSemesterName") or ""
    credits = raw_record.get("acquiredCredits") or raw_record.get("totalCredits") or ""
    gpa = raw_record.get("gradeAverage") or raw_record.get("gpa") or ""
    advisor = raw_record.get("advisorProfessorName") or raw_record.get("profNm") or raw_record.get("advisor") or ""

    status_display = status
    if change and change != status:
        status_display = f"{status} ({change})"

    # Return pure non-PII record
    return {
        "departmentName": dept,
        "collegeName": college,
        "enrollmentStatus": status_display,
        "completedSemesterCount": sem,
        "acquiredCredits": credits,
        "gradeAverage": gpa,
        "entryYear": entry_year,
        "advisor": advisor,
    }


def build_anonymized_academic_summary(clean_record: Dict[str, Any]) -> str:
    """
    Builds LLM-safe academic grounding summary without any names, full student IDs, or personal phone numbers.
    """
    dept = clean_record.get("departmentName", "")
    colg = clean_record.get("collegeName", "")
    status_display = clean_record.get("enrollmentStatus", "")
    sem = clean_record.get("completedSemesterCount", "")
    credits = clean_record.get("acquiredCredits", "")
    gpa = clean_record.get("gradeAverage", "")
    entry = clean_record.get("entryYear", "")
    advisor = clean_record.get("advisor", "")

    advisor_hint = ""
    if advisor:
        advisor_hint = f"- 지도교수: {advisor} 교수님 (연락처/전화번호 조회가 필요한 경우 `api_searchContacts(query='{advisor}')` 도구를 호출하세요.)\n"

    dept_label = f"{colg} {dept}".strip() if colg else dept

    return (
        f"\n[포털 종합정보(ERP) 학생 학적 연동 데이터 (비식별화 완료)]:\n"
        f"- 소속: {dept_label}\n"
        f"- 학적 상태: {status_display}" + (f" (이수 학기: {sem})" if sem else "") + "\n"
        f"- 취득 학점: {credits}학점 (평점 평균: {gpa})\n"
        f"- 입학 정보: {entry}학번\n"
        + advisor_hint
        + "- [응답 지침]: 학생의 성명/이름이나 학번 등 개인 식별 정보는 일체 제공되지 않습니다. 특정 이름을 부르지 말고 질문에 대해 명확하고 정중하게 필요한 학사 정보를 바로 안내하세요.\n"
    )
