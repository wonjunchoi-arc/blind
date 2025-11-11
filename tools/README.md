# BlindInsight 데이터 수집 도구

블라인드(Blind.com) 리뷰 데이터를 수집하고 AI 기반으로 분석하기 위한 크롤링 및 처리 도구 모음입니다.

**현재 버전**: v3.2 (배치 처리 최적화)

## 🚀 주요 기능

### 🔍 블라인드 리뷰 크롤링 (`blind_review_crawler.py`)
- **배치 최적화된 크롤링**: API 호출 횟수를 최대 90% 절약하는 대용량 배치 처리
- **AI 기반 분류**: OpenAI API를 활용한 정확한 리뷰 카테고리 분류
- **키워드 분류 대안**: API 비용 없이 사용 가능한 키워드 기반 분류
- **LangSmith 통합**: 성능 추적 및 분석을 위한 트레이싱 지원
- **직무별 연도별 분석**: 상세한 메타데이터 수집으로 정밀한 분석 가능

### 📊 리뷰 분류 카테고리
- **커리어 성장** (career_growth): 승진, 학습 기회, 스킬 개발
- **급여 및 복지** (salary_benefits): 연봉, 보너스, 복리후생
- **업무와 삶의 균형** (work_life_balance): 근무시간, 휴가, 업무 강도
- **사내 문화** (company_culture): 동료 관계, 조직 문화, 소통
- **경영진** (management): 리더십, 의사결정, 비전

## 🛠️ 설치 및 환경 설정

### 필수 요구사항
- Python 3.10+
- Chrome 브라우저 (크롤링용)
- OpenAI API 키 (AI 분류 사용시)

### 환경변수 설정
```bash
# .env 파일 생성
OPENAI_API_KEY=your_openai_api_key_here
LANGSMITH_API_KEY=your_langsmith_api_key_here  # 선택사항
LANGSMITH_TRACING=true  # LangSmith 트레이싱 활성화시
AI_BATCH_SIZE=30  # 배치 크기 설정 (기본값)
```

### 의존성 설치
```bash
pip install selenium beautifulsoup4 pandas tqdm python-dotenv langsmith openai
```

## 📋 사용법

### 1. 단일 기업 크롤링
```python
from tools.blind_review_crawler import run_single_company_crawl

# 기본 사용법 (AI 분류)
success = run_single_company_crawl(
    company_code="NAVER",
    pages=25,
    headless=False,
    use_ai_classification=True,
    openai_api_key=None,  # .env에서 자동 로드
    enable_spell_check=True
)

# 키워드 분류 (무료)
success = run_single_company_crawl(
    company_code="NAVER",
    pages=25,
    headless=False,
    use_ai_classification=False
)
```

### 2. 여러 기업 일괄 크롤링
```python
from tools.blind_review_crawler import run_multiple_companies_crawl

company_list = ["NAVER", "카카오", "삼성전자", "LG전자"]

results = run_multiple_companies_crawl(
    company_list=company_list,
    pages=25,
    headless=True,
    delay_between_companies=30,
    use_ai_classification=True,
    openai_api_key=None,  # .env에서 자동 로드
    enable_spell_check=True
)

print(f"성공: {len(results['success'])}개")
print(f"실패: {len(results['failed'])}개")
print(f"API 호출 절약: {results['total_api_calls_saved']}회")
```

### 3. 명령행 실행
```bash
# 대화형 모드로 실행
python tools/blind_review_crawler.py

# 선택 옵션:
# 1. 단일 기업 크롤링
# 2. 여러 기업 크롤링 (직접 입력)
# 3. 한국 상위 50개 기업 일괄 크롤링
```

## ⚡ 배치 최적화 기능

### 기존 방식 vs 배치 최적화
```
기존 방식: 각 리뷰마다 개별 AI API 호출
- 100개 리뷰 = 100회 API 호출
- 높은 비용과 느린 처리 속도

배치 최적화: 모든 청크를 수집 후 대용량 배치 처리
- 100개 리뷰 = 3-4회 API 호출 (30개씩 배치)
- 90% API 호출 절약 + 빠른 처리 속도
```

### 처리 프로세스
1. **리뷰 수집**: 모든 페이지에서 리뷰 데이터 추출
2. **청크 생성**: 리뷰를 분석 가능한 청크 단위로 분할
3. **배치 분류**: 대용량 배치로 AI 분류 실행
4. **벡터DB 저장**: 최적화된 형태로 ChromaDB에 저장

## 📁 출력 파일 구조

크롤링 완료 후 다음 형태로 파일이 생성됩니다:
```
data/vectordb/
├── 20250915_카카오_ai_batch_vectordb.json
├── 20250915_네이버_ai_batch_vectordb.json
├── 20250915_삼성전자_keyword_vectordb.json  (키워드 분류시)
└── ...
```

**파일명 형식**: `{날짜}_{회사명}_{분류방법}_vectordb.json`
- **날짜**: YYYYMMDD 형식
- **회사명**: 크롤링한 회사명
- **분류방법**: `ai_batch` (AI 분류) 또는 `keyword` (키워드 분류)

**출력 경로**: `./data/vectordb` (크롤러 실행 디렉토리 기준)
- 프로젝트 루트에서 실행시: `data/vectordb/`
- tools 디렉토리에서 실행시: `tools/data/vectordb/`

### 출력 데이터 형식
각 JSON 파일은 다음 구조를 가집니다:
```json
{
  "metadata": {
    "company": "카카오",
    "total_chunks": 1010,
    "ai_classified_chunks": 1010,
    "keyword_classified_chunks": 0,
    "created_at": "2025-09-15T14:04:19.793641",
    "classification_method": "ai_batch",
    "version": "v3.2_batch_optimized"
  },
  "chunks": [
    {
      "id": "work_life_balance_pros_카카오_0000_00",
      "content": "자율 근무제가 장점이었는데 사라짐 업무량이 많지 않은 편이고...",
      "metadata": {
        "company": "카카오",
        "category": "work_life_balance",
        "category_kr": "업무와 삶의 균형",
        "content_type": "pros",
        "is_positive": true,
        "source_section": "장점",
        "priority": "primary",
        "rating": 4.0,
        "confidence_score": 0.75,
        "classification_method": "ai_batch",
        "employee_status": "현직원",
        "position": "IT 엔지니어",
        "year": "2022",
        "sentence_count": 2,
        "chunk_index": 0,
        "content_length": 46
      }
    }
  ]
}
```

**파일 구조 설명**:
- **metadata**: 전체 파일 메타정보 (회사명, 총 청크 수, 분류 방법, 버전)
- **chunks**: 개별 청크 배열
  - **id**: 청크 고유 식별자 (카테고리_타입_회사_인덱스)
  - **content**: 실제 리뷰 내용
  - **metadata**: 청크별 상세 메타데이터
    - **category**: 카테고리 코드 (work_life_balance, salary_benefits 등)
    - **category_kr**: 한글 카테고리명
    - **content_type**: 콘텐츠 타입 (pros, cons, title, improvement)
    - **is_positive**: 긍정/부정 여부
    - **confidence_score**: AI 분류 신뢰도 (0.0~1.0)
    - **classification_method**: 분류 방법 (ai_batch 또는 keyword)
    - **rating**: 원본 평점 (1.0~5.0)

## ⚠️ 주의사항

### 블라인드 로그인
1. 블라인드 앱에서 로그인
2. '더보기' → '블라인드 웹 로그인' 클릭
3. 인증번호 입력
4. 크롤러에서 Enter 키 입력


## 📞 문제 해결

### 일반적인 오류
- **드라이버 오류**: Chrome 브라우저 버전 확인
- **로그인 실패**: 블라인드 앱에서 웹 로그인 재시도
- **API 오류**: OpenAI API 키 및 할당량 확인
- **메모리 부족**: 배치 크기를 줄이거나 페이지 수 감소

### 로그 파일
실행 중 발생하는 오류는 `blind_crawler.log` 파일에 기록됩니다.

```bash
# 로그 확인
tail -f blind_crawler.log
```

## 🔄 통합 워크플로우

이 도구로 수집된 데이터는 다음과 같이 활용됩니다:

### 1️⃣ 데이터 수집 (이 문서)
```bash
python tools/blind_review_crawler.py
# 출력: tools/data/vectordb/(company)_reviews_vectordb_*.json
```

### 2️⃣ 벡터화 및 ChromaDB 저장
```bash
python migrate_reviews.py
# 또는 특정 회사만
python migrate_reviews.py NAVER
```

**처리 과정**:
- JSON 청크 데이터 로드
- OpenAI 임베딩 생성 (배치 최적화)
- ChromaDB 컬렉션별 저장 (company_culture, salary_benefits 등)
- 데이터 무결성 검증 및 성능 리포트 생성

### 3️⃣ AI 분석 활용
```bash
python main.py
# Streamlit UI에서 회사 분석 및 AI 검색 사용
```

**전체 파이프라인**:
```
🕷️ Crawl → 📦 Vectorize → 🤖 Analyze
(tools/)   (migrate)      (frontend/app.py)
```

전체 프로세스에 대한 자세한 내용은 [프로젝트 루트 README.md](../README.md)를 참고하세요.