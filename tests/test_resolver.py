from app.orchestrator.resolver import ToolParameterResolver


def test_bus_stop_parameter_resolution():
    args_inip = ToolParameterResolver.resolve_arguments("api_getBusArrivals", "BUS", "인천대입구역 셔틀 언제 와?")
    assert args_inip["bstopId"] == "164000396"

    args_jungmun = ToolParameterResolver.resolve_arguments("api_getBusArrivals", "BUS", "정문 버스 몇분 남아?")
    assert args_jungmun["bstopId"] == "164000385"

    args_eng = ToolParameterResolver.resolve_arguments("api_getBusArrivals", "BUS", "공과대학 버스 도착시간")
    assert args_eng["bstopId"] == "164000377"

    args_default = ToolParameterResolver.resolve_arguments("api_getBusArrivals", "BUS", "지금 셔틀버스 언제 오는지 알려줘")
    assert "bstopId" in args_default
    assert args_default["bstopId"] in ["164000396", "164000385"]


def test_cafeteria_parameter_resolution():
    args_caf = ToolParameterResolver.resolve_arguments("api_getCafeteriaMenu", "CAFETERIA", "학생식당 오늘 메뉴")
    assert args_caf["cafeteria"] == "학생식당"
    assert 1 <= args_caf["day"] <= 7

    args_dorm = ToolParameterResolver.resolve_arguments("api_getCafeteriaMenu", "CAFETERIA", "기숙사식당 메뉴 뭐야?")
    assert args_dorm["cafeteria"] == "제1기숙사식당"


def test_notice_parameter_resolution():
    args_notice = ToolParameterResolver.resolve_arguments("api_getNotices", "NOTICE", "장학금 공지 알려줘")
    assert args_notice["category"] == "장학"
