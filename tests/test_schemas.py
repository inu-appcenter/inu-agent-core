from app.llm.schemas import (
    MetricCard,
    MetricCardItem,
    StatusCard,
    ListCard,
    ListItem,
    ActionCard,
    CardBadge,
    ClientActionInstruction,
    ClientActionRequest,
)


def test_metric_card_schema():
    card = MetricCard(
        title="2026-1학기 성적 요약",
        badge=CardBadge(text="정상", theme="success"),
        main_metric=MetricCardItem(label="평점평균", value="4.12", highlight=True),
        sub_details=[MetricCardItem(label="취득학점", value="18")],
    )
    dumped = card.model_dump()
    assert dumped["card_type"] == "METRIC_CARD"
    assert dumped["main_metric"]["value"] == "4.12"
    assert dumped["badge"]["text"] == "정상"


def test_status_card_schema():
    card = StatusCard(
        title="도서관 3열람실 좌석 42번",
        status="이용 중",
        progress_percent=80,
        time_remaining="45분 남음",
        primary_action_label="이용 연장하기",
        action_id="RENEW_SEAT_42",
    )
    dumped = card.model_dump()
    assert dumped["card_type"] == "STATUS_CARD"
    assert dumped["progress_percent"] == 80


def test_list_card_schema():
    card = ListCard(
        title="마감 예정 과제",
        items=[
            ListItem(title="운영체제 과제", subtitle="내일 마감", tag="D-1"),
            ListItem(title="네트워크 실습", subtitle="3일 뒤 마감", tag="D-3"),
        ],
    )
    dumped = card.model_dump()
    assert dumped["card_type"] == "LIST_CARD"
    assert len(dumped["items"]) == 2


def test_client_action_instruction():
    inst = ClientActionInstruction(
        action_id="act_test_01",
        auth_domain="PORTAL",
        protocol="NEXACRO_SSV",
        request=ClientActionRequest(
            url="https://portal.inu.ac.kr/grade.do",
            method="POST",
            params={"stuno": "202201234"},
        ),
    )
    dumped = inst.model_dump()
    assert dumped["auth_domain"] == "PORTAL"
    assert dumped["protocol"] == "NEXACRO_SSV"
