# 🔍 BlindInsight AI

**AI 기반 회사 분석 및 커리어 상담 플랫폼**

BlindInsight AI는 블라인드(Blind.com) 리뷰 데이터를 수집하고 AI로 분석하여, 개인화된 회사 분석과 커리어 상담을 제공하는 종합 플랫폼입니다.

![BlindInsight AI Architecture](https://img.shields.io/badge/AI-LangGraph%20%2B%20RAG-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge)

## 🌟 핵심 기능

### 🤖 AI 회사 분석
- **다중 에이전트 병렬 분석**: 기업문화, 연봉/복지, 경영진, 커리어, 워라벨을 독립적으로 분석
- **실시간 데이터 통합**: 최신 리뷰 데이터 활용
- **컨텍스트 기반 답변**: RAG (Retrieval Augmented Generation) 기술로 정확한 정보 제공

### 🔍 AI 검색 (Supervisor Chat)
- **자연어 질의**: 복잡한 회사 관련 질문을 자연어로 처리
- **다차원 검색**: 문화, 급여, 성장성, 커리어 등 모든 영역 통합 검색
- **실시간 응답**: 벡터 DB 기반 빠른 검색과 AI 추론



## 🏗️ 시스템 아키텍처

### 전체 데이터 플로우
```
📥 데이터 수집        📊 데이터 처리         🤖 AI 분석          💻 웹 인터페이스
─────────────      ─────────────      ─────────────      ─────────────
Blind Crawler  →   Vector Migration  →   AI Agents    →   Streamlit UI
(tools/)           (migrate_reviews)     (LangGraph)       (frontend/)
```

### 상세 아키텍처
```
🌐 Streamlit Web Interface (main.py → frontend/app.py)
├── 🏠 홈페이지 & 대시보드
├── 🔍 AI 회사 분석 페이지
└── 💬 AI 검색 (Supervisor Chat)

🧠 AI Analysis Engine (LangGraph Multi-Agent)
├── 🏛️ 기업문화 분석 에이전트
├── 💰 급여 및 복지 분석 에이전트
├── 📈 경영진 분석 에이전트
└── 🎯 커리어 분석 에이전트
└── 🎯 워라벨 분석 에이전트

💾 Data Processing Pipeline
├── 🕷️ 크롤링 (tools/blind_review_crawler.py)
├── 🔄 벡터 변환 (migrate_reviews.py)
└── 📚 지식 베이스 (ChromaDB)

🔗 Integration Layer
├── 🤝 MCP (Model Context Protocol)
├── 📡 RAG (Retrieval Augmented Generation)
└── 🔍 Vector Search Engine
```

## 🚀 빠른 시작

### 1. 환경 설정

#### 필수 요구사항
- Python 3.10 이상
- OpenAI API 키
- Chrome 브라우저 (데이터 수집용)

#### 가상환경 생성 및 활성화
```bash
# Windows (PowerShell)
python -m venv blindinsight-env
.\blindinsight-env\Scripts\Activate.ps1

# Windows (Command Prompt)
blindinsight-env\Scripts\activate.bat

# macOS/Linux
python -m venv blindinsight-env
source blindinsight-env/bin/activate
```

#### 의존성 설치
```bash
# 권장: uv 사용 (빠른 설치)
pip install uv
uv pip install -r requirements.txt

# 또는 기본 pip 사용
pip install -r requirements.txt
```

### 2. 환경변수 설정
`.env` 파일을 프로젝트 루트에 생성:
```bash
# 필수 설정
OPENAI_API_KEY=your_openai_api_key_here

# 선택적 설정 (고급 기능용)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true

# 데이터베이스 경로 (기본값 사용 권장)
VECTOR_DB_PATH=./data/vector_db
JSON_DATA_PATH=./tools/data

# AI 처리 설정
AI_BATCH_SIZE=30
EMBEDDING_MODEL=text-embedding-3-large
```

### 3. 애플리케이션 실행
```bash
# 메인 애플리케이션 시작
python main.py

# 또는 직접 Streamlit 실행
streamlit run main.py

# 브라우저에서 http://localhost:8501 접속
```

## 📚 상세 사용 가이드

### 🕷️ 1단계: 데이터 수집 (선택사항)

새로운 회사 데이터가 필요한 경우에만 수행:

```bash
# 대화형 크롤링 (권장)
python tools/blind_review_crawler.py

# 단일 회사 크롤링
python -c "
from tools.blind_review_crawler import run_single_company_crawl
run_single_company_crawl('NAVER', pages=25, use_ai_classification=True)
"
```

**크롤링 옵션**:
- **AI 분류**: 정확하지만 OpenAI API 비용 발생
- **키워드 분류**: 무료이지만 정확도 낮음
- **배치 처리**: API 비용 90% 절약하는 최적화 기능

### 🔄 2단계: 데이터 벡터화 및 마이그레이션

수집된 리뷰 데이터를 AI 검색이 가능한 형태로 변환:

```bash
# 전체 데이터 마이그레이션
python migrate_reviews.py

# 특정 회사만 처리
python migrate_reviews.py NAVER

# 도움말 확인
python migrate_reviews.py --help
```

**마이그레이션 과정**:
1. JSON 파일에서 청크 데이터 로드
2. OpenAI 임베딩 모델로 벡터 생성 (배치 처리로 최적화)
3. ChromaDB에 카테고리별 컬렉션으로 저장
4. 데이터 무결성 검증 및 성능 보고서 생성

### 🌐 2단계: 웹 애플리케이션 사용


#### AI 회사 분석 페이지
```
🏢 회사 선택 → 📋 분석 설정 → 🤖 AI 분석 → 📊 결과 확인
```

**분석 탭 구성**:
- **커뮤니티**: 실제 직원 후기 및 평점
- **기본 정보**: 회사 개요 및 기본 데이터
- **AI 분석**: 5개 에이전트 병렬 분석 결과

#### AI 검색 (Supervisor Chat)
```
💬 자연어 질문 → 🔍 벡터 검색 → 🧠 AI 추론 → ✨ 정확한 답변
```

**질문 예시**:
- "삼성전자에서 워라밸이 좋은 팀은 어디인가요?"
- "SK하이닉스의 연봉은 얼마인가요"

## 🔧 고급 설정

### 개발 모드 활성화
`src/blindinsight/models/base.py`에서:
```python
class Settings(BaseSettings):
    enable_debug_mode: bool = True  # False에서 True로 변경
```

### AI 에이전트 설정 커스터마이징
`src/blindinsight/agents/base.py`에서 기본 설정 수정 가능:
- 검색 임계값 조정
- 응답 길이 제한
- 병렬 처리 여부 설정

### 벡터 DB 컬렉션 관리
```python
# 컬렉션 상태 확인
from blindinsight.rag.knowledge_base import KnowledgeBase
kb = KnowledgeBase()
kb.get_collection_stats()

# 특정 회사 데이터만 삭제
kb.delete_company_data("NAVER")
```


### 코드 품질 검사
```bash
# 코드 포맷팅
black src/

# 타입 검사
mypy src/

# 린트 검사
flake8 src/
```

### 개발 워크플로우
1. **기능 개발**: 새로운 AI 에이전트나 분석 기능 추가
2. **테스트 작성**: 단위 테스트 및 통합 테스트 작성
3. **품질 검사**: black, mypy, flake8으로 코드 검증
4. **수동 테스트**: Streamlit 인터페이스에서 기능 확인

## 📊 성능 최적화

### 배치 처리 최적화
- **크롤링**: 대용량 배치 처리로 API 호출 90% 절약
- **벡터화**: 100개 문서씩 배치 임베딩 생성
- **저장**: 500개씩 ChromaDB 배치 저장

### 메모리 관리
- **스트리밍 처리**: 대용량 파일을 청크 단위로 처리
- **캐싱**: 자주 사용되는 결과 메모리 캐싱
- **비동기 처리**: I/O 대기 시간 최소화

### 응답 시간 최적화
- **병렬 에이전트**: 4개 에이전트 동시 실행
- **벡터 검색**: 유사도 0.7 이상만 반환
- **결과 캐싱**: 동일 질문 즉시 응답

## 🗂️ 프로젝트 구조

```
BlindInsight/
├── 📁 src/blindinsight/          # 메인 소스 코드
│   ├── 🤖 agents/                # AI 에이전트 모듈
│   │   ├── base.py              # 기본 에이전트 클래스
│   │   ├── culture_agent.py     # 문화 분석 에이전트
│   │   ├── compensation_agent.py # 급여 분석 에이전트
│   │   ├── growth_agent.py      # 성장성 분석 에이전트
│   │   └── career_agent.py      # 커리어 분석 에이전트
│   │
│   ├── 💬 chat/                  # 대화형 AI 모듈
│   │   ├── modern_supervisor.py # Supervisor Chat 엔진
│   │   ├── state.py            # 대화 상태 관리
│   │   └── workflow.py         # 대화 워크플로우
│   │
│   ├── 🎨 frontend/              # 웹 인터페이스
│   │   └── app.py              # Streamlit 메인 앱
│   │
│   ├── 🔗 mcp/                   # MCP 통합 모듈
│   │   ├── client.py           # MCP 클라이언트
│   │   └── providers.py        # 데이터 제공자
│   │
│   ├── 📊 models/                # 데이터 모델
│   │   ├── base.py             # 기본 설정 및 모델
│   │   ├── analysis.py         # 분석 요청/응답 모델
│   │   └── user.py             # 사용자 프로필 모델
│   │
│   └── 📚 rag/                   # RAG 시스템
│       ├── knowledge_base.py   # 지식 베이스 관리
│       └── json_processor.py   # JSON 데이터 처리
│
├── 🛠️ tools/                     # 데이터 수집 도구
│   ├── blind_review_crawler.py # 블라인드 크롤러
│   ├── enhanced_category_processor.py # 카테고리 처리
│   ├── keyword_dictionary.py   # 키워드 사전
│   └── README.md              # 도구 사용법
│
├── 🗄️ data/                      # 데이터 디렉토리
│   ├── vector_db/             # ChromaDB 벡터 데이터베이스
│   ├── embeddings/            # 임베딩 캐시
│   └── cache/                 # 기타 캐시 파일
│
├── 🧪 tests/                     # 테스트 코드
│   ├── unit/                  # 단위 테스트
│   ├── integration/           # 통합 테스트
│   └── fixtures/              # 테스트 데이터
│
├── 📋 main.py                    # 메인 진입점
├── 🔄 migrate_reviews.py         # 데이터 마이그레이션 스크립트
├── 📝 requirements.txt           # Python 의존성
├── ⚙️ .env.example              # 환경변수 예시
└── 📖 README.md                 # 이 파일
```

## 🔍 주요 컴포넌트 상세

### 🤖 AI 에이전트 시스템
**LangGraph 기반 다중 에이전트 아키텍처**
- **BaseAgent**: 모든 에이전트의 공통 기능 (RAG 검색, MCP 통합, LLM 호출)
- **병렬 실행**: 4개 에이전트 동시 분석으로 속도 향상
- **결과 통합**: 각 에이전트 결과를 종합한 최종 리포트 생성

### 📚 RAG (Retrieval Augmented Generation)
**ChromaDB 기반 벡터 검색 시스템**
- **임베딩 모델**: OpenAI text-embedding-3-large (3072차원)
- **컬렉션**: 문화, 급여, 커리어, 인터뷰, 일반 정보별 분리 저장
- **검색 최적화**: 유사도 임계값 0.7, 최대 20개 문서 반환


## 🚨 문제 해결

### 일반적인 오류

#### 애플리케이션 실행 오류
```bash
# 환경변수 확인
python main.py --check-requirements

# 가상환경 확인
which python  # 또는 where python (Windows)

# 의존성 재설치
pip install -r requirements.txt --force-reinstall
```

#### OpenAI API 오류
- **API 키 확인**: `.env` 파일의 OPENAI_API_KEY 검증
- **할당량 확인**: OpenAI 계정의 사용량 및 잔액 확인
- **모델 접근**: text-embedding-3-large 모델 사용 가능 여부 확인

#### ChromaDB 오류
```bash
# 벡터 DB 초기화
rm -rf data/vector_db/
python migrate_reviews.py

# 권한 문제 (Linux/macOS)
chmod -R 755 data/
```

#### 메모리 부족
- 배치 크기 축소: `AI_BATCH_SIZE=10` (기본값 30)
- 크롤링 페이지 수 감소: 25페이지 → 10페이지
- 시스템 메모리 확인 및 다른 애플리케이션 종료

### 로그 파일 위치
- **애플리케이션 로그**: `blindinsight.log`
- **크롤링 로그**: `tools/blind_crawler.log`
- **Streamlit 로그**: 터미널에 실시간 출력

## 🤝 기여하기

### 개발 환경 설정
1. Repository fork 및 clone
2. 개발용 의존성 설치: `pip install -r requirements.txt`
3. pre-commit 훅 설정: `pre-commit install`
4. 테스트 실행 확인: `pytest tests/`

### 코드 기여 가이드라인
- **코드 스타일**: Black (line length: 88)
- **타입 힌트**: mypy strict 모드 준수
- **테스트**: 새 기능에 대한 단위 테스트 필수
- **문서화**: docstring 및 주석으로 코드 의도 명확히

### 새로운 AI 에이전트 추가
1. `src/blindinsight/agents/` 에 새 에이전트 클래스 생성
2. `BaseAgent` 상속 및 필수 메서드 구현
3. `frontend/app.py`에 UI 통합
4. 테스트 코드 작성

## 📄 라이센스

이 프로젝트는 개인 연구 및 학습 목적으로 개발되었습니다.

**사용 제한사항**:
- 상업적 이용 금지
- 개인정보 및 민감 정보 수집 금지
- 블라인드 서비스 약관 준수 필수

## 🙏 감사의 말

- **Blind.com**: 익명 직장 후기 플랫폼 제공
- **OpenAI**: GPT 및 임베딩 모델 API 제공
- **LangChain/LangGraph**: AI 에이전트 프레임워크 제공
- **Streamlit**: 웹 인터페이스 프레임워크 제공

---

**📧 문의사항이나 제안사항이 있으시면 언제든 연락해주세요!**

> ⭐ **도움이 되셨다면 Star를 눌러주세요!**
>
> 지속적인 개발과 개선에 큰 힘이 됩니다. 🚀