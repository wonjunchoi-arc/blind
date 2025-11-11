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

#### ✅ 필수 설정
`.env` 파일을 프로젝트 루트에 생성:
```bash
# OpenAI API 키 (필수)
OPENAI_API_KEY=your_openai_api_key_here
```

#### 🔧 선택적 설정

**AI 모델 설정** - 각 에이전트별 모델 커스터마이징
```bash
# 기본 에이전트 모델 (미설정시 gpt-4o-mini 사용)
# 📍 사용처: src/blindinsight/models/base.py (Settings 클래스)
DEFAULT_AGENT_MODEL=gpt-4o-mini-2024-07-18

# Supervisor 및 도구 모델
# 📍 사용처: src/blindinsight/chat/modern_supervisor.py (멀티 에이전트 조율)
SUPERVISOR_MODEL=gpt-4o-mini-2024-07-18
AGENT_TOOL_MODEL=gpt-4o-mini-2024-07-18

# 개별 에이전트 모델 (선택) - 에이전트별 다른 모델 사용 가능
# 📍 사용처: src/blindinsight/agents/*.py (각 전문 에이전트)
SALARY_BENEFITS_AGENT_MODEL=gpt-4o-mini-2024-07-18      # 급여/복지 분석
COMPANY_CULTURE_AGENT_MODEL=gpt-4o-mini-2024-07-18      # 기업문화 분석
WORK_LIFE_BALANCE_AGENT_MODEL=gpt-4o-mini-2024-07-18    # 워라밸 분석
MANAGEMENT_AGENT_MODEL=gpt-4o-mini-2024-07-18           # 경영진 분석
CAREER_GROWTH_AGENT_MODEL=gpt-4o-mini-2024-07-18        # 커리어 성장 분석
QUALITY_EVALUATOR_AGENT_MODEL=gpt-4o-mini-2024-07-18    # 품질 평가
```

**검색 및 재랭킹 설정** - RAG 성능 튜닝용
```bash
# 검색 파라미터
# 📍 사용처: src/blindinsight/agents/base.py (RAG 검색 시스템)
DEFAULT_SEARCH_K=10              # 기본 검색 결과 수
MAX_SEARCH_K=50                  # 최대 검색 결과 수
RERANK_SEARCH_MULTIPLIER=2       # 재랭킹시 검색 배수

# 앙상블 검색 가중치 (BM25 + 벡터 검색)
# 📍 사용처: src/blindinsight/rag/retriever.py (하이브리드 검색)
ENSEMBLE_WEIGHT_BM25=0.3         # 키워드 검색 가중치 (합계 1.0)
ENSEMBLE_WEIGHT_VECTOR=0.7       # 의미 검색 가중치
RELEVANCE_SCORE_THRESHOLD=0.3    # 최소 관련성 점수 (0.0~1.0)
```

**데이터 처리 설정**
```bash
# 카테고리별 청크 처리 파라미터
# 📍 사용처: tools/enhanced_category_processor.py (리뷰 분류 및 청크 생성)
CATEGORY_CHUNK_SIZE=1            # 카테고리별 청크 크기
CATEGORY_MAX_CHUNK_SIZE=2        # 최대 청크 크기
CATEGORY_MIN_SENTENCE_LENGTH=5   # 최소 문장 길이

# 배치 처리 설정
# 📍 사용처: src/blindinsight/models/base.py (임베딩 생성)
AI_BATCH_SIZE=30                 # 임베딩 배치 크기 (메모리 부족시 감소)
MAX_CHUNK_LENGTH=300             # 최대 청크 길이 (토큰 제한용)
EMBEDDING_MODEL=text-embedding-3-small  # OpenAI 임베딩 모델
```

**데이터베이스 및 경로 설정**
```bash
# 벡터 데이터베이스 경로 (기본값 사용 권장)
# 📍 사용처: src/blindinsight/models/base.py, migrate_reviews.py
VECTOR_DB_PATH=./data/embeddings        # ChromaDB 저장 경로
JSON_DATA_PATH=./data/vectordb          # JSON 청크 데이터 경로
DATA_DIRECTORY=./data                   # 전체 데이터 루트 디렉토리

# 데이터베이스 URL
# 📍 사용처: src/blindinsight/models/base.py (SQLite 연결)
DATABASE_URL=sqlite:///C:/blind/data/blindinsight.db
```

**디버깅 및 로깅**
```bash
# LangSmith 추적 (LangGraph 워크플로우 디버깅용)
# 📍 사용처: 전역 환경 변수 (LangChain 자동 감지)
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true

# 로깅 레벨
# 📍 사용처: src/blindinsight/models/base.py (애플리케이션 로그)
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR
```

#### 💡 환경 변수 우선순위
1. `.env` 파일의 명시적 설정
2. `Settings` 클래스의 기본값
3. 시스템 환경 변수

#### ⚙️ 환경 변수 검증
```bash
# 환경 변수 및 의존성 확인
python main.py --check-requirements
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

새로운 회사 데이터가 필요한 경우에만 수행합니다.
상세한 크롤링 가이드는 **[tools/README.md](tools/README.md)**를 참고하세요.

**Quick Start**:
```bash
# 대화형 크롤링
python tools/blind_review_crawler.py
```

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
- "현대차의 기업문화는 어때요?"
- "SK하이닉스의 연봉은 얼마인가요"

## 📊 성능 최적화

### 배치 처리 최적화
- **크롤링**: 대용량 배치 처리로 API 호출 90% 절약
- **벡터화**: 100개 문서씩 배치 임베딩 생성
- **저장**: 500개씩 ChromaDB 배치 저장

### 응답 시간 최적화
- **병렬 에이전트**: 4개 에이전트 동시 실행
- **벡터 검색**: 유사도 0.7 이상만 반환

## 🗂️ 프로젝트 구조

```
BlindInsight/
├── 📁 src/blindinsight/          # 메인 소스 코드
│   │
│   ├── 🤖 agents/                # AI 에이전트 모듈
│   │   ├── base.py              # 기본 에이전트 클래스 (BaseAgent, AgentConfig)
│   │   ├── salary_benefits_agent.py      # 급여/복지 분석 에이전트
│   │   ├── company_culture_agent.py      # 기업문화 분석 에이전트
│   │   ├── work_life_balance_agent.py    # 워라밸 분석 에이전트
│   │   ├── management_agent.py           # 경영진 분석 에이전트
│   │   ├── career_growth_agent.py        # 커리어 성장 분석 에이전트
│   │   ├── quality_evaluator_agent.py    # 품질 평가 에이전트
│   │   └── __init__.py          # 에이전트 모듈 초기화
│   │
│   ├── 💬 chat/                  # 대화형 AI 모듈 (Supervisor Chat)
│   │   ├── modern_supervisor.py # LangGraph 기반 멀티 에이전트 조율자
│   │   ├── state.py            # SupervisorState 및 대화 상태 관리
│   │   ├── workflow.py         # Handoff 도구 및 워크플로우 로직
│   │   └── __init__.py         # Chat 모듈 초기화
│   │
│   ├── 🎨 frontend/              # 웹 인터페이스 (Streamlit)
│   │   ├── app.py              # Streamlit 메인 앱 (홈, AI 분석, AI 검색)
│   │   ├── components.py       # UI 컴포넌트 (차트, 카드, 테이블)
│   │   ├── utils.py            # 프론트엔드 유틸리티 함수
│   │   └── __init__.py         # Frontend 모듈 초기화
│   │
│   │
│   ├── 📊 models/                # 데이터 모델 및 설정
│   │   ├── base.py             # Settings 클래스 (환경 변수 관리)
│   │   ├── analysis.py         # 분석 요청/응답 모델
│   │   ├── company.py          # 회사 정보 모델
│   │   ├── user.py             # 사용자 프로필 모델
│   │   └── __init__.py         # Models 모듈 초기화
│   │
│   ├── 📚 rag/                   # RAG (Retrieval Augmented Generation) 시스템
│   │   ├── knowledge_base.py   # KnowledgeBase 클래스 (ChromaDB 관리)
│   │   ├── json_processor.py   # ChunkDataLoader (JSON → ChromaDB 마이그레이션)
│   │   ├── retriever.py        # RAGRetriever (하이브리드 검색, 재랭킹)
│   │   ├── embeddings.py       # BatchEmbeddingManager (배치 임베딩 생성)
│   │   ├── document_processor.py # 문서 전처리 및 청킹
│   │   └── __init__.py         # RAG 모듈 초기화
│   │
│   └── __init__.py              # BlindInsight 패키지 초기화
│
├── 🛠️ tools/                     # 데이터 수집 및 전처리 도구
│   ├── blind_review_crawler.py # 블라인드 리뷰 크롤러 (Selenium 기반)
│   ├── enhanced_category_processor.py # AI 기반 카테고리 분류기
│   ├── keyword_dictionary.py   # 카테고리별 키워드 사전
│   ├── text_processor.py       # 텍스트 전처리 유틸리티
│   ├── config.py               # 크롤러 설정
│   ├── data/                   # 크롤링된 JSON 데이터 저장소
│   │   └── (회사명)_reviews_*.json
│   └── README.md              # 도구 사용법
│
├── 🗄️ data/                      # 데이터 디렉토리
│   ├── embeddings/            # ChromaDB 벡터 데이터베이스
│   │   ├── company_culture/   # 기업문화 컬렉션
│   │   ├── work_life_balance/ # 워라밸 컬렉션
│   │   ├── management/        # 경영진 컬렉션
│   │   ├── salary_benefits/   # 급여/복지 컬렉션
│   │   ├── career_growth/     # 커리어 성장 컬렉션
│   │   └── general/           # 일반 정보 컬렉션
│   ├── vectordb/              # JSON 청크 데이터 (마이그레이션 전)
│   └── blindinsight.db        # SQLite 데이터베이스 (선택)
│
├── 📋 main.py                    # 애플리케이션 진입점
├── 🔄 migrate_reviews.py         # 데이터 마이그레이션 스크립트 (배치 최적화)
├── 📝 requirements.txt           # Python 의존성 목록
├── ⚙️ .env.example              # 환경변수 템플릿
├── ⚙️ .env                       # 환경변수 설정 (생성 필요)
└── 📖 README.md                 # 프로젝트 문서
```

### 주요 모듈 설명

#### 🤖 **agents/** - AI 에이전트 시스템
- **BaseAgent**: 모든 에이전트의 공통 기능 (RAG 검색, MCP 통합, LLM 호출)
- **전문 에이전트**: 각 분석 영역별 특화된 프롬프트와 로직 제공
- **병렬 실행**: 5개 에이전트 동시 분석으로 속도 향상

#### 💬 **chat/** - Supervisor Chat 시스템
- **ModernSupervisorAgent**: LangGraph StateGraph 기반 멀티 에이전트 조율
- **Command Handoffs**: 도구 호출을 통한 에이전트 간 작업 위임
- **SupervisorState**: 대화 상태 및 히스토리 관리

#### 📚 **rag/** - RAG 시스템
- **KnowledgeBase**: ChromaDB 벡터 데이터베이스 관리
- **RAGRetriever**: BM25 + 벡터 검색 하이브리드 앙상블
- **BatchEmbeddingManager**: 배치 임베딩 생성으로 API 비용 90% 절감

#### 🛠️ **tools/** - 데이터 수집 도구
- **blind_review_crawler**: Selenium 기반 블라인드 리뷰 자동 수집
- **enhanced_category_processor**: GPT 기반 리뷰 카테고리 자동 분류
- **배치 처리**: 대용량 리뷰 데이터 효율적 처리

## 🔍 주요 컴포넌트 상세

### 🤖 AI 에이전트 시스템
**LangGraph 기반 다중 에이전트 아키텍처**
- **BaseAgent**: 모든 에이전트의 공통 기능 (RAG 검색, LLM 호출)
- **병렬 실행**: 5개 에이전트 동시 분석으로 속도 향상
- **결과 통합**: 각 에이전트 결과를 종합한 최종 리포트 생성

### 📚 RAG (Retrieval Augmented Generation)
**ChromaDB 기반 벡터 검색 시스템**
- **임베딩 모델**: OpenAI text-embedding-3-large
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



#### 메모리 부족
- 배치 크기 축소: `AI_BATCH_SIZE=10` (기본값 30)
- 크롤링 페이지 수 감소: 25페이지 → 10페이지
- 시스템 메모리 확인 및 다른 애플리케이션 종료

### 로그 파일 위치
- **애플리케이션 로그**: `blindinsight.log`
- **크롤링 로그**: `tools/blind_crawler.log`
- **Streamlit 로그**: 터미널에 실시간 출력



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

