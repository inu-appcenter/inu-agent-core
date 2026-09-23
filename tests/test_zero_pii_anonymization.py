from app.core.anonymizer import (
    sanitize_text,
    sanitize_academic_record,
    build_anonymized_academic_summary,
)


def test_sanitize_text_masks_full_student_id_and_phone():
    # 1. 8~9자리 풀 학번 -> 연도학번으로 변환
    raw_query = "내 학번 202101234 인데 졸업 요건 알려줘"
    clean = sanitize_text(raw_query)
    assert "202101234" not in clean
    assert "2021학번" in clean

    # 2. 휴대폰 번호 마스킹
    raw_phone = "내 전화번호 010-1234-5678 또는 01098765432 로 연락줘"
    clean_phone = sanitize_text(raw_phone)
    assert "010-1234-5678" not in clean_phone
    assert "01098765432" not in clean_phone
    assert "[개인전화번호]" in clean_phone

    # 3. 이메일 및 주민등록번호 마스킹
    raw_rrn = "내 주민번호 010203-1234567 이메일 test@inu.ac.kr"
    clean_rrn = sanitize_text(raw_rrn)
    assert "010203-1234567" not in clean_rrn
    assert "test@inu.ac.kr" not in clean_rrn


def test_sanitize_academic_record_strips_pii():
    raw_erp_data = {
        "koreanName": "홍길동",
        "studentId": "202101234",
        "entryYear": "2021",
        "departmentName": "컴퓨터공학부",
        "collegeName": "정보기술대학",
        "enrollmentStatus": "재학",
        "completedSemesterCount": "6",
        "acquiredCredits": "98",
        "gradeAverage": "3.85",
        "advisorProfessorName": "김교수",
        "phoneNumber": "010-1111-2222",
        "email": "student@inu.ac.kr",
        "resNo": "010101-1234567",
    }

    clean = sanitize_academic_record(raw_erp_data)

    # 실명, 풀학번, 전화번호, 이메일, 주민번호 등 PII 완전 제거 검증
    assert "name" not in clean
    assert "koreanName" not in clean
    assert "studentId" not in clean
    assert "phoneNumber" not in clean
    assert "email" not in clean
    assert "resNo" not in clean

    # 필요한 비식별 학적 정보만 보존
    assert clean["entryYear"] == "2021"
    assert clean["departmentName"] == "컴퓨터공학부"
    assert clean["collegeName"] == "정보기술대학"
    assert clean["enrollmentStatus"] == "재학"
    assert clean["completedSemesterCount"] == "6"
    assert clean["acquiredCredits"] == "98"
    assert clean["gradeAverage"] == "3.85"
    assert clean["advisor"] == "김교수"


def test_build_anonymized_academic_summary_contains_zero_pii():
    clean_record = {
        "departmentName": "컴퓨터공학부",
        "collegeName": "정보기술대학",
        "enrollmentStatus": "재학",
        "completedSemesterCount": "6",
        "acquiredCredits": "98",
        "gradeAverage": "3.85",
        "entryYear": "2021",
        "advisor": "김교수",
    }

    summary = build_anonymized_academic_summary(clean_record)

    assert "홍길동" not in summary
    assert "202101234" not in summary
    assert "010-" not in summary
    assert "2021학번" in summary
    assert "정보기술대학 컴퓨터공학부" in summary
    assert "학우님" in summary
