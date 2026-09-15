# inu-agent-core

> **인천대학교 앱센터(INU AppCenter) 중앙 AI 에이전트 오케스트레이션 코어 서버**

`inu-agent-core`는 인천대학교 포털 서비스(INTIP), 생활관(UNIDorm) 등 앱센터의 다양한 서비스 및 학교 시스템(LMS, 종합정보시스템, 도서관)을 하나로 엮어 제공하는 독립 AI 에이전트 중추 서버입니다.

---

## 🌟 주요 기능 및 아키텍처

- **LiteLLM / vLLM (Gemma 27B) 연동:** 별도 GPU 인프라와 OpenAI 규격 비동기 통신 (`/v1`).
- **Gemma 27B 도구 다이어트 (Tool Pruning):** 모델 주의력(Attention) 유지를 위해 질문 의도에 맞춰 도구 3~4개 선별 주입.
- **W3C 표준 SSE 스트리밍:** 실시간 토큰 생성 및 이벤트 스트리밍 (`POST /api/v1/chat/stream`).
- **4대 범용 Server-Driven UI (SDUI) 카드 스키마:**
  - `MetricCard`: 장학금, 성적 요약 등 핵심 지표형 카드
  - `StatusCard`: 열람실 좌석, 세탁기 등 잔여 시간/진행률 카드
  - `ListCard`: 과제 마감 목록, 공지사항 등 리스트형 카드
  - `ActionCard`: 외박 신청 등 원터치 승인 액션 카드
- **쿠콘 방식 Client Action Protocol:** 개인정보보호 및 학교 방화벽 준수를 위해 모바일 앱 단말기가 직접 학교 시스템을 P2P 호출하는 지침(Instruction) 제공.

---

## 📁 프로젝트 구조

```
inu-agent-core/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── chat.py         # SSE 실시간 스트리밍 엔드포인트
│   │       ├── health.py       # 헬스 체크 엔드포인트
│   │       └── router.py       # API 라우터
│   ├── core/
│   │   ├── config.py           # Pydantic Settings (.env 로드)
│   │   └── logging.py          # 정형 로깅 설정
│   ├── llm/
│   │   ├── client.py           # AsyncOpenAI LiteLLM/vLLM 클라이언트
│   │   └── schemas.py          # SDUI 카드 & Client Action 프로토콜 스키마
│   ├── orchestrator/
│   │   ├── engine.py           # Tool Pruning & 에이전트 실행 루프
│   │   └── prompts.py          # 시스템 프롬프트 및 페르소나 설정
│   ├── tools/
│   │   └── base.py             # BaseTool 추상 클래스
│   └── main.py                 # FastAPI 애플리케이션 진입점
├── tests/                      # 단위 테스트 스위트
├── .env.example                # 환경 변수 템플릿
├── .gitignore                  # Git 추적 제외 설정
├── Dockerfile                  # 컨테이너 빌드 파일
├── docker-compose.yml          # App + Redis 통합 실행 파일
└── requirements.txt            # Python 의존성 목록
```

---

## 🚀 로컬 실행 방법

### 1. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 열어 GPU 서버 URL(LLM_BASE_URL) 및 API Key를 설정합니다.
```

### 2. 가상환경 및 패키지 설치
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. 서버 실행
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 테스트 실행
```bash
pytest tests
```

### 5. Docker Compose로 실행
```bash
docker-compose up -d --build
```

---

## 🚢 CI/CD 및 배포 (GitHub Actions)

`inu-agent-core`는 GitHub Actions를 통해 Docker Hub 빌드 및 학교 서버 원격 자동 배포(SSH/SCP)를 지원합니다.

* **운영 배포 (`cicd-prod.yml`)**: `main` 브랜치에 코드가 푸시되거나 수동 트리거(`workflow_dispatch`) 시 실행됩니다.
* **개발 배포 (`cicd-dev.yml`)**: `develop` / `dev` 브랜치에 코드가 푸시되거나 수동 트리거 시 실행됩니다.

### 필요 GitHub Secrets 설정
GitHub Repository의 **Settings > Secrets and variables > Actions**에 아래 시크릿을 등록합니다:

| 분류 | Secret Key | 설명 |
| :--- | :--- | :--- |
| **DockerHub** | `USERNAME` | Docker Hub 계정명 / Organization 이름 |
| | `TOKEN` | Docker Hub Access Token |
| | `PROJECT_NAME` | 이미지 레포지토리 명 (`inu-agent-core`) |
| **Server SSH** | `HOST_ADDRESS` | 학교 대상 서버 IP 주소 또는 도메인 |
| | `SERVER_USERNAME` | 서버 SSH 접속 계정명 (예: `ubuntu`) |
| | `SERVER_KEY` | SSH 접속용 Private Key (`id_rsa` / `id_ed25519`) |
| | `SERVER_PORT` | SSH 포트 번호 (기본: `22`) |
| | `MOUNT_PATH` | 서버 내 배포 디렉토리 경로 (예: `/home/ubuntu/apps/inu-agent-core`) |
| **App ENV** | `ENV_PROD` | 운영 환경용 `.env` 파일 내용 전체 |
| | `ENV_DEV` | *(선택)* 개발 환경용 `.env` 파일 내용 전체 |

