import httpx
import json

token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIzIiwicm9sZXMiOlsiUk9MRV9BRE1JTiJdLCJpYXQiOjE3OTEwMDMyMDIsImV4cCI6MTc5MTA4OTYwMn0.9oyGF4ahhWa7n0Me_kxybevXtOc7ttm4zSaOU-aG3bU"
headers = {"Auth": token, "Authorization": f"Bearer {token}"}
base = "https://portal-dev.inuappcenter.kr"

def check(name, path, params=None):
    try:
        r = httpx.get(f"{base}{path}", params=params, headers=headers, timeout=5)
        print(f"=== [{name}] {path} (params={params}) ===")
        print(f"Status: {r.status_code}")
        try:
            d = r.json()
            if isinstance(d, dict):
                data = d.get("data")
                msg = d.get("msg")
                print(f"Msg: {msg}, Data type: {type(data).__name__}")
                if isinstance(data, dict):
                    print(f"Keys: {list(data.keys())}")
                    for k in ["contents", "content", "items", "total", "pages"]:
                        if k in data:
                            v = data[k]
                            if isinstance(v, list):
                                print(f"  {k}: length={len(v)}")
                                if v:
                                    print(f"  First item: {json.dumps(v[0], ensure_ascii=False)[:200]}")
                            else:
                                print(f"  {k}: {v}")
                elif isinstance(data, list):
                    print(f"List length: {len(data)}")
                    if data:
                        print(f"  First item: {json.dumps(data[0], ensure_ascii=False)[:200]}")
            else:
                print(f"Raw: {str(d)[:200]}")
        except Exception as ex:
            print(f"Non-JSON response: {r.text[:200]}")
    except Exception as e:
        print(f"Request failed: {e}")
    print()

if __name__ == "__main__":
    # 1. 개설강의 (CourseOfferings) 연도/학기별 조회
    print("--- 1. 개설강의 (CourseOfferings) ---")
    check("CourseOfferings (no params)", "/api/course-offerings")
    check("CourseOfferings 2026-SECOND", "/api/course-offerings", {"year": 2026, "term": "SECOND"})
    check("CourseOfferings 2024-SECOND", "/api/course-offerings", {"year": 2024, "term": "SECOND"})
    check("CourseOfferings 2024-FIRST", "/api/course-offerings", {"year": 2024, "term": "FIRST"})

    # 2. 총학생회 공지 (CouncilNotices)
    print("--- 2. 총학생회 공지 (CouncilNotices) ---")
    check("CouncilNotices", "/api/councilNotices")

    # 3. 학과 공지 (Department Notices)
    print("--- 3. 학과 공지 (Department Notices) ---")
    check("DeptNotices (COM_ENG)", "/api/notices/department", {"department": "COMPUTER_ENGINEERING"})

    # 4. 전체 학교 공지 (All Notices)
    print("--- 4. 학교 공식 공지 (Notices) ---")
    check("Notices", "/api/notices")

    # 5. 분실물 (Lost)
    print("--- 5. 분실물 (Lost) ---")
    check("Lost", "/api/lost")

    # 6. 동아리 (Clubs)
    print("--- 6. 동아리 (Clubs) ---")
    check("Clubs", "/api/clubs")

    # 7. 강의계획서 (Syllabus)
    print("--- 7. 강의계획서 (Syllabus) ---")
    check("Syllabus", "/api/syllabus", {"courseOfferingId": 1})

    # 8. 성적 (Grades)
    print("--- 8. 성적 (Grades) ---")
    check("Grades All", "/api/grades/all")
    check("Grades 2024-FIRST", "/api/grades", {"year": 2024, "term": "FIRST"})

    # 9. 통합검색 (Unified Search)
    print("--- 9. 통합 검색 (UnifiedSearch) ---")
    check("UnifiedSearch (q=컴퓨터)", "/api/search/unified", {"q": "컴퓨터"})
    check("UnifiedSearch (q=총학생회)", "/api/search/unified", {"q": "총학생회"})
    check("UnifiedSearch (q=장학금)", "/api/search/unified", {"q": "장학금"})
    check("UnifiedSearch (q=동아리)", "/api/search/unified", {"q": "동아리"})
