# 🔍 BlindInsight AI

**AI 기반 회사 분석 및 커리어 상담 플랫폼**

BlindInsight AI는 익명 직장 리뷰 데이터를 분석하여 데이터 기반의 커리어 인사이트를 제공하는 AI 플랫폼입니다.

## 🎯 주요 기능

### 🏢 회사 분석
- **문화 분석**: 실제 직장인들의 솔직한 회사 문화 리뷰 분석
- **연봉 정보**: 포지션별, 경력별 실제 연봉 데이터 제공
- **성장 기회**: 회사의 안정성과 성장 가능성 분석
- **면접 정보**: 실제 면접 후기와 준비 팁 제공

### 💼 개인화된 커리어 상담
- **맞춤형 추천**: 개인 프로필 기반 최적 회사 추천
- **스킬 갭 분석**: 목표 포지션을 위한 필요 역량 분석
- **커리어 로드맵**: 단계별 경력 발전 전략 제시
- **시장 동향 분석**: 업계 트렌드와 기회 분석

### 🔧 핵심 기술 스택
- **LangGraph**: 멀티 에이전트 워크플로우 오케스트레이션
- **RAG (Retrieval-Augmented Generation)**: ChromaDB 기반 지식 검색
- **MCP (Model Context Protocol)**: 외부 데이터 소스 통합
- **Streamlit**: 인터랙티브 웹 인터페이스
- **OpenAI GPT**: 자연어 처리 및 분석

## 🚀 빠른 시작

### 1. 요구사항
- Python 3.9 이상
- [uv](https://docs.astral.sh/uv/) (Python 패키지 매니저)
- OpenAI API 키

### 2. 설치

```bash
# 저장소 클론
git clone https://github.com/wonjunchoi-arc/blind.git
cd blind

# uv 설치 (아직 설치하지 않았다면)
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# uv로 가상환경 생성 및 의존성 설치
uv venv blindinsight-env
uv pip install -r requirements.txt
```

### 3. 가상환경 활성화

```bash
# Windows (PowerShell)
.\blindinsight-env\Scripts\Activate.ps1

# Windows (Command Prompt)
blindinsight-env\Scripts\activate.bat

# macOS/Linux
source blindinsight-env/bin/activate
```

### 4. 환경 설정

```bash
# .env.example을 .env로 복사
cp .env.example .env

# .env 파일을 열고 실제 API 키를 입력하세요
# OPENAI_API_KEY="your_openai_api_key_here"
```

⚠️ **보안 주의사항**: 
- `.env` 파일은 절대로 Git에 커밋하지 마세요
- API 키는 안전하게 보관하고 공유하지 마세요

### 5. 실행

```bash
# 메인 애플리케이션 실행
python main.py

# 또는 Streamlit으로 직접 실행
streamlit run main.py
```

브라우저에서 `http://localhost:8501`로 접속하세요.

## 📁 프로젝트 구조

```
blind/
├── src/blindinsight/           # 메인 패키지
│   ├── models/                 # 데이터 모델
│   │   ├── base.py            # 기본 설정 및 모델
│   │   ├── analysis.py        # 분석 요청/응답 모델
│   │   ├── user.py            # 사용자 프로필 모델
│   │   └── company.py         # 회사 데이터 모델
│   ├── workflow/              # LangGraph 워크플로우
│   │   ├── state.py           # 워크플로우 상태 관리
│   │   ├── nodes.py           # 개별 처리 노드들
│   │   └── graph.py           # 워크플로우 오케스트레이션
│   ├── agents/                # AI 에이전트들
│   │   ├── base.py            # 기본 에이전트 클래스
│   │   ├── culture_agent.py   # 문화 분석 에이전트
│   │   ├── compensation_agent.py  # 연봉 분석 에이전트
│   │   ├── growth_agent.py    # 성장성 분석 에이전트
│   │   └── career_agent.py    # 커리어 분석 에이전트
│   ├── rag/                   # RAG 시스템
│   │   ├── embeddings.py      # 임베딩 및 벡터 저장소
│   │   ├── document_processor.py  # 문서 처리 및 청킹
│   │   ├── retriever.py       # 검색 및 재랭킹
│   │   └── knowledge_base.py  # 통합 지식 베이스
│   ├── mcp/                   # MCP 통합 레이어
│   │   ├── client.py          # MCP 클라이언트
│   │   ├── providers.py       # 데이터 제공자들
│   │   └── handlers.py        # 데이터 처리 핸들러
│   └── frontend/              # Streamlit 프론트엔드
│       ├── app.py             # 메인 애플리케이션
│       ├── components.py      # UI 구성 요소
│       └── utils.py           # 유틸리티 함수
├── data/                      # 데이터 디렉토리
├── logs/                      # 로그 파일
├── requirements.txt           # Python 의존성
├── pyproject.toml            # 패키지 설정
├── main.py                   # 메인 실행 파일
└── README.md                 # 이 파일
```

## 🔧 설정 및 구성

### 환경 변수

```bash
# 필수 설정
OPENAI_API_KEY=your_openai_api_key

# 선택 설정
EMBEDDING_DIMENSIONS=3072       # 임베딩 차원 수
VECTOR_DB_PATH=./data/vector_db # 벡터 DB 경로
LOG_LEVEL=INFO                  # 로그 레벨
```

### 데이터 소스 설정

`src/blindinsight/models/base.py`에서 설정을 변경할 수 있습니다:

```python
class Settings(BaseSettings):
    # OpenAI 설정
    openai_api_key: str
    embedding_dimensions: int = 3072
    
    # 데이터베이스 설정
    vector_db_path: str = "./data/vector_db"
    
    # MCP 설정
    mcp_enabled: bool = True
```

## 🎨 사용 가이드

### 1. 회사 분석

1. **회사 검색**: 메인 페이지에서 분석하고 싶은 회사명 입력
2. **분석 유형 선택**: 종합 분석, 문화 분석, 연봉 분석 등 선택
3. **결과 확인**: 다양한 차트와 인사이트로 결과 확인

### 2. 커리어 상담

1. **프로필 설정**: 설정 페이지에서 개인 프로필 입력
2. **AI 채팅**: 커리어 관련 질문을 AI와 대화
3. **맞춤형 조언**: 개인화된 커리어 조언 받기

### 3. 데이터 탐색

1. **키워드 검색**: 관심 있는 키워드로 검색
2. **필터 적용**: 회사, 카테고리별 필터링
3. **상세 정보**: 개별 리뷰와 데이터 상세 확인

## 🧪 개발 및 테스트

### 개발 환경 설정

```bash
# 개발용 의존성 설치
pip install -e ".[dev]"

# 코드 품질 검사
flake8 src/
black src/
mypy src/

# 테스트 실행
pytest tests/
```

### 데이터 추가

새로운 회사 데이터를 추가하려면:

1. `src/blindinsight/mcp/providers.py`에서 데이터 제공자 수정
2. MCP 클라이언트를 통해 실제 API 연동
3. 지식 베이스에 데이터 추가

## 🚀 배포

### Docker를 사용한 배포

```bash
# Dockerfile 생성 (예시)
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 클라우드 배포

- **Streamlit Cloud**: GitHub 연동으로 간편 배포
- **Heroku**: `requirements.txt`와 `setup.sh` 파일 추가
- **AWS/GCP**: 컨테이너화 후 클라우드 서비스 배포

## 📊 성능 최적화

### 캐싱 전략

- **Streamlit 캐시**: 자주 사용되는 데이터 캐싱
- **임베딩 캐시**: 임베딩 생성 결과 캐싱
- **API 응답 캐시**: 외부 API 호출 결과 캐싱

### 메모리 관리

- **배치 처리**: 대량 데이터 처리 시 배치 단위로 처리
- **스트리밍**: 대용량 파일 처리 시 스트리밍 방식 사용
- **정기 정리**: 불필요한 캐시 및 임시 파일 정리

## 🤝 기여하기

1. Fork 저장소
2. 기능 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 지원 및 문의

- **이슈 신고**: GitHub Issues 페이지 활용
- **기능 요청**: GitHub Discussions 또는 Issues
- **문서 개선**: README 또는 코드 주석 개선 PR

## 🔮 로드맵

### v1.1 (계획)
- [ ] 실제 Blind API 연동
- [ ] 추가 데이터 소스 연동 (원티드, 잡플래닛)
- [ ] 고급 감정 분석 모델 적용
- [ ] 실시간 데이터 업데이트

### v1.2 (계획)
- [ ] 회사 비교 기능 강화
- [ ] 개인화 추천 알고리즘 개선
- [ ] 모바일 반응형 UI 개선
- [ ] 다국어 지원

### v2.0 (장기)
- [ ] 기업 대상 B2B 서비스
- [ ] 실시간 채팅 상담
- [ ] 커리어 트래킹 대시보드
- [ ] AI 면접 준비 서비스

---

**BlindInsight AI** - 데이터 기반 커리어 의사결정을 위한 AI 플랫폼

Built with ❤️ by the BlindInsight Team