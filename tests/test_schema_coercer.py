import pytest
from app.tools.coercer import SchemaCoercer

def test_schema_coercer_integer_and_day():
    schema = {
        "properties": {
            "day": {"type": "integer", "description": "요일, 월요일=1부터 일요일=7"},
            "count": {"type": "integer", "description": "개수"}
        }
    }
    
    # 1. Coerce "TODAY" / "오늘" to weekday int
    res1 = SchemaCoercer.coerce(schema, {"day": "오늘", "count": "10"})
    assert isinstance(res1["day"], int)
    assert 1 <= res1["day"] <= 7
    assert res1["count"] == 10
    
    # 2. String digit
    res2 = SchemaCoercer.coerce(schema, {"day": "5", "count": "3"})
    assert res2["day"] == 5
    assert res2["count"] == 3

def test_schema_coercer_enum_and_synonyms():
    schema = {
        "properties": {
            "mealType": {
                "type": "string",
                "enum": ["AUTO", "BREAKFAST", "LUNCH", "DINNER"]
            },
            "cafeteria": {
                "type": "string",
                "enum": ["전체", "학생식당", "제1기숙사식당", "2호관(교직원)식당"]
            }
        }
    }
    
    # 1. Case-insensitivity
    res1 = SchemaCoercer.coerce(schema, {"mealType": "lunch", "cafeteria": "전체"})
    assert res1["mealType"] == "LUNCH"
    assert res1["cafeteria"] == "전체"
    
    # 2. Korean semantic synonyms
    res2 = SchemaCoercer.coerce(schema, {"mealType": "점심", "cafeteria": "학식"})
    assert res2["mealType"] == "LUNCH"
    assert res2["cafeteria"] == "학생식당"
    
    res3 = SchemaCoercer.coerce(schema, {"mealType": "저녁", "cafeteria": "1긱"})
    assert res3["mealType"] == "DINNER"
    assert res3["cafeteria"] == "제1기숙사식당"

def test_schema_coercer_boolean():
    schema = {
        "properties": {
            "enabled": {"type": "boolean"}
        }
    }
    assert SchemaCoercer.coerce(schema, {"enabled": "true"})["enabled"] is True
    assert SchemaCoercer.coerce(schema, {"enabled": "false"})["enabled"] is False
    assert SchemaCoercer.coerce(schema, {"enabled": "yes"})["enabled"] is True

def test_schema_coercer_parameter_aliases():
    # 1. Tool expecting 'q', caller passes 'query'
    schema_q = {
        "properties": {
            "q": {"type": "string"}
        }
    }
    assert SchemaCoercer.coerce(schema_q, {"query": "수강신청"}) == {"q": "수강신청"}
    assert SchemaCoercer.coerce(schema_q, {"keyword": "장학"}) == {"q": "장학"}

    # 2. Tool expecting 'query', caller passes 'q'
    schema_query = {
        "properties": {
            "query": {"type": "string"}
        }
    }
    assert SchemaCoercer.coerce(schema_query, {"q": "수강신청"}) == {"query": "수강신청"}


def test_schema_coercer_course_offerings_dept_and_grade():
    schema = {
        "properties": {
            "year": {"type": "integer"},
            "term": {"type": "string"},
            "deptName": {"type": "string"},
            "hyNames": {"type": "array"},
            "keyword": {"type": "string"},
        }
    }

    # 1. Alias '컴공' -> '컴퓨터공학부', '2학년' -> ['2']
    res1 = SchemaCoercer.coerce(schema, {"deptName": "컴공", "hyNames": "2학년"})
    assert res1["deptName"] == "컴퓨터공학부"
    assert res1["hyNames"] == ["2"]
    assert "keyword" not in res1

    # 2. Alias '데사' -> '데이터과학과'
    res2 = SchemaCoercer.coerce(schema, {"deptName": "데사"})
    assert res2["deptName"] == "데이터과학과"
    assert "keyword" not in res2

    # 3. If keyword accidentally equals deptName, keyword is popped
    res3 = SchemaCoercer.coerce(schema, {"deptName": "컴퓨터공학부", "keyword": "컴퓨터공학부"})
    assert res3["deptName"] == "컴퓨터공학부"
    assert "keyword" not in res3

    # 4. Valid course keyword is preserved
    res4 = SchemaCoercer.coerce(schema, {"deptName": "컴퓨터공학부", "keyword": "자료구조"})
    assert res4["deptName"] == "컴퓨터공학부"
    assert res4["keyword"] == "자료구조"

