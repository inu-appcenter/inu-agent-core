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
