# [아키텍처 제안서] 차세대 시맨틱 도구 리트리버(Semantic Tool Retriever) 전환 로드맵

- **문서 버전**: v1.0.0
- **작성 일자**: 2026-09-16
- **대상 시스템**: `inu-agent-core` (인천대학교 앱센터 챗불이 오케스트레이션 코어)
- **문서 상태**: Proposed / Archiving

---

## 1. 제안 배경 및 현주소 분석

### 1.1 현재 아키텍처 (`ToolPruner`)의 동작 방식
현재 `inu-agent-core`는 인천대학교 포털 서버(`inu-portal-server`)의 121개 OpenAPI REST 엔드포인트를 실시간으로 동기화하여 에이전트 도구로 등록하고 있습니다. 그러나 상용 로컬/클라우드 LLM(Gemma 27B 등)의 컨텍스트 윈도우 한계와 함수 호출(Function Calling) 환각을 방지하기 위해 질의당 3~4개의 도구만 선별하여 주입해야 합니다.

현재의 도구 프루닝(`app/orchestrator/pruner.py`) 방식은 다음과 같습니다:
1. **키워드 사전 매칭**: `CATEGORY_KEYWORDS` 딕셔너리에 도메인별 단어 목록(예: `CAFETERIA`: 학식, 식당, 밥, 메뉴...)을 하드코딩.
2. **점수 가중치 룰**: 카테고리 매칭 시 `+10점`, 도구명/설명 일치 시 `+5점`, 경로 변수(`{scheduleId}`) 미포함 시 `-15점`, 검색 엔드포인트 `+3점`.
3. **특정 의도 하드코딩 if문**: `"나 졸업"`, `"내 학점"` 감지 시 포털과 규정 카테고리를 강제 활성화(`is_first_person_grad`).

### 1.2 현재 방식의 명확한 한계와 기술 부채
1. **유지보수 확장성 제약 (Maintenance Bottleneck)**:
   - 스프링 서버에 새로운 도메인(예: "시설 예약", "분실물", "취업 공채") API가 추가될 때마다 개발자가 에이전트 코드를 열어 키워드 사전을 갱신하고 재배포해야 합니다.
2. **자연어 표현 한계**:
   - 오타("학시익"), 동의어("학관 밥", "인입역"), 복합 질문("학식 먹고 탈 버스 시간") 등 사전에 없는 변형 표현에 취약합니다.
3. **가중치 파편화**:
   - 도구가 200개, 300개로 늘어날수록 `+10점`, `-15점` 같은 매직 넘버 규칙이 서로 충돌할 가능성이 높아집니다.

---

## 2. 현대 엔터프라이즈 AI 에이전트 아키텍처 비교

| 비교 항목 | Pattern A: 키워드 룰셋 (현재) | Pattern B: Pure LLM 라우팅 | Pattern C: 시맨틱 도구 리트리버 (제안) |
| :--- | :--- | :--- | :--- |
| **선별 방식** | 정적 키워드 사전 매칭 | LLM이 120개 도구를 매번 읽고 판단 | **도구 임베딩 벡터 유사도 검색 (Tool RAG)** |
| **레이턴시** | 매우 빠름 (< 0.5ms) | 매우 느림 (+1.5s ~ 3.0s 추가 LLM 호출) | **매우 빠름 (1~5ms 인메모리 벡터 연산)** |
| **비용/토큰** | 0 토큰 | 질의당 수천 토큰 낭비 | **0 추가 토큰 (로컬 임베딩 시 무료)** |
| **코드 하드코딩** | `CATEGORY_KEYWORDS`, 점수제 존재 | 없음 (프롬프트 의존) | **완전 제로 하드코딩 (100% 동적 인덱싱)** |
| **새 API 대응** | 코드 수정 및 배포 필수 | 자동 (프롬프트 내 주입) | **서버 기동 시 자동 인덱싱 (코드 수정 불필요)** |
| **정확도/강건성** | 사전 범위 내에서만 작동 | 도구 수 증가 시 환각 위험 | **동의어, 오타, 문맥 완벽 대응** |

---

## 3. 제안 아키텍처: In-Memory Semantic Tool Retriever (Tool RAG)

### 3.1 전체 처리 파이프라인

```mermaid
flowchart TD
    subgraph Server_Boot [서버 기동 시 (Lifespan)]
        A[Spring Boot OpenAPI /v3/api-docs] --> B[120개 OpenApiTool 파싱]
        B --> C[도구 텍스트화: name + description + parameters]
        C --> D[Embedding Model: bge-m3 / text-embedding-3]
        D --> E[(In-Memory Tool Vector Store / NumPy Index)]
    end

    subgraph Runtime_Query [실시간 질의 처리]
        Q[사용자 질문 유입] --> G{하드 가드레일?}
        G -- 코딩/순수수학 --> R[즉시 거절 응답 (0ms)]
        G -- 정상 학사 질문 --> H[질문 텍스트 임베딩 생성]
        H --> I[E: Tool Vector Store 코사인 유사도 검색]
        I --> J[Top 3~4개 도구 동적 추출]
        J --> K[LLM ReAct / Tool Argument 리졸버]
        K --> L[도구 실행 및 SDUI 카드 합성]
    end
```

### 3.2 핵심 메커니즘
1. **Tool Document Serialization**:
   각 도구를 검색 가능한 단일 문서로 표현합니다.
   ```text
   [도구명]: api_getCafeterias
   [카테고리]: CAFETERIA
   [설명]: 인천대학교 학생식당 및 교직원식당의 오늘 메뉴, 식단표, 가격을 조회합니다.
   [파라미터]: date(날짜), mealType(조식/중식/석식)
   ```
2. **In-Memory Embedding Index**:
   - 120개 도구의 벡터는 차원이 384~1024차원일 때 총 메모리 사용량이 **1MB 미만**입니다. 별도의 외부 Vector DB(Pinecone 등)를 거칠 필요 없이, FastAPI 메모리 내에서 NumPy Cosine Similarity 연산으로 **1ms 이내**에 상위 3개를 찾아냅니다.
3. **임베딩 모델 선택지**:
   - **옵션 1 (권장 - 완전 무료/로컬)**: `fastembed` 라이브러리의 `BAAI/bge-small-en-v1.5` 또는 `bge-m3-korean` (CPU에서 15ms 내 연산, 외부 의존성 없음).
   - **옵션 2 (클라우드 API)**: OpenAI `text-embedding-3-small` 또는 Google Gemini Embedding (비용 거의 0에 수렴).

---

## 4. 구현 설계 청사진 (PoC 코드 스케치)

### 4.1 `app/orchestrator/retriever.py`
```python
from typing import List, Tuple
import numpy as np
from app.tools.base import BaseTool
from app.llm.embeddings import get_embeddings  # fastembed or openai

class SemanticToolRetriever:
    def __init__(self):
        self.tools: List[BaseTool] = []
        self.vectors: np.ndarray = np.empty((0, 384))

    async def index_tools(self, tools: List[BaseTool]):
        """서버 기동 시 1회 호출되어 120개 도구를 벡터화"""
        self.tools = tools
        tool_texts = [
            f"도구명: {t.name}\n설명: {t.description}\n경로: {getattr(t, 'path', '')}"
            for t in tools
        ]
        embeddings = await get_embeddings(tool_texts)
        self.vectors = np.array(embeddings, dtype=np.float32)
        # L2 정규화 (코사인 유사도 내적 계산용)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.vectors = self.vectors / np.maximum(norms, 1e-9)

    async def retrieve(self, query: str, top_k: int = 4) -> List[BaseTool]:
        """런타임에 사용자 질의와 가장 의미적으로 유사한 도구 top_k 반환"""
        if len(self.tools) == 0 or len(self.vectors) == 0:
            return []

        query_vec = np.array(await get_embeddings([query]), dtype=np.float32)[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec /= query_norm

        scores = np.dot(self.vectors, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [self.tools[i] for i in top_indices if scores[i] > 0.25]
```

### 4.2 기존 룰과의 완전한 분리
- `app/orchestrator/pruner.py`의 `CATEGORY_KEYWORDS`, `score += 10`, `score -= 15`, `is_first_person_grad` 등의 **모든 키워드 사전과 조건문을 영구 제거**.
- 새 API 추가 시 스프링 부트 서버의 `/v3/api-docs`를 다시 긁어오기만 하면 **자동으로 새 도구가 벡터 인덱스에 편입**됨.

---

## 5. 단계별 실행 로드맵

```
[Phase 1: 기반 마련]
- fastembed 또는 임베딩 클라이언트 모듈 app/llm/embeddings.py 추가
- SemanticToolRetriever 단위 테스트 작성 및 기존 22개 테스트 통과 검증

[Phase 2: 섀도우 런 (Shadow Run)]
- 기존 ToolPruner와 SemanticToolRetriever의 추천 결과를 로그로 비교
- 8대 핵심 도메인(버스, 학식, 일정, 도서관, 공지 등) 매칭 정합성 99% 확인

[Phase 3: 전면 전환 및 룰 코드 폐기]
- app/orchestrator/pruner.py 폐기 및 SemanticToolRetriever 활성화
- 완전한 Zero-Hardcoding, Zero-Rule AI 에이전트 완성
```

---

## 6. 결론 요약
- **현재 구현체**: 비즈니스 데이터(학점, 버스 노선, 학식 등)는 100% 동적 연동되었으나, 도구 선택 계층에 키워드/점수제 룰이 일부 남아있음.
- **차세대 구현체**: 인메모리 임베딩 기반의 `SemanticToolRetriever`를 적용하면 코드 내의 모든 휴리스틱을 완전히 걷어내고, API 변경에 자율 대응하는 완전한 지능형 에이전트로 도약할 수 있음.
