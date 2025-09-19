# BlindInsight 데이터 수집 도구

블라인드(Blind.com) 리뷰 데이터를 수집하고 AI 기반으로 분석하기 위한 크롤링 및 처리 도구 모음입니다.

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
from blind_review_crawler import run_single_company_crawl

# 기본 사용법 (AI 분류)
success = run_single_company_crawl(
    company_code="NAVER",
    pages=25,
    headless=False,
    use_ai_classification=True,
    enable_spell_check=True
)

# 키워드 분류 (무료)
success = run_single_company_crawl(
    company_code="NAVER",
    pages=25,
    use_ai_classification=False
)
```

### 2. 여러 기업 일괄 크롤링
```python
from blind_review_crawler import run_multiple_companies_crawl

company_list = ["NAVER", "카카오", "삼성전자", "LG전자"]

results = run_multiple_companies_crawl(
    company_list=company_list,
    pages=25,
    headless=True,
    delay_between_companies=30,
    use_ai_classification=True
)

print(f"성공: {len(results['success'])}개")
print(f"실패: {len(results['failed'])}개")
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
├── NAVER_reviews_vectordb_2024-01-15.json
├── 카카오_reviews_vectordb_2024-01-15.json
└── 삼성전자_reviews_vectordb_2024-01-15.json
```

### 출력 데이터 형식
각 JSON 파일은 다음 구조를 가집니다:
```json
{
  "company": "NAVER",
  "created_at": "2024-01-15T10:30:00",
  "total_reviews": 150,
  "total_chunks": 450,
  "categories": {
    "career_growth": 90,
    "salary_benefits": 95,
    "work_life_balance": 85,
    "company_culture": 100,
    "management": 80
  },
  "chunks": [
    {
      "id": "review_NAVER_0001_pros",
      "content": "리뷰 내용...",
      "category": "career_growth",
      "confidence": 0.85,
      "metadata": {
        "company": "NAVER",
        "employee_type": "현직원",
        "position": "소프트웨어 엔지니어",
        "year": "2024",
        "chunk_type": "pros"
      }
    }
  ]
}
```

## 🔧 고급 설정

### 크롤링 옵션
- `pages`: 크롤링할 페이지 수 (기본: 25)
- `headless`: 헤드리스 모드 실행 여부
- `wait_timeout`: 요소 대기 시간 (초)
- `output_dir`: 출력 디렉토리 경로

### AI 분류 옵션
- `use_ai_classification`: AI 분류 사용 여부
- `enable_spell_check`: 맞춤법 검사 활성화
- `batch_size`: 배치 처리 크기 (기본: 30)

### 안정성 설정
- `delay_between_companies`: 기업간 대기 시간 (초)
- Chrome 옵션 자동 설정으로 안정적 크롤링
- 에러 복구 및 재시도 로직 내장

## ⚠️ 주의사항

### 블라인드 로그인
1. 블라인드 앱에서 로그인
2. '더보기' → '블라인드 웹 로그인' 클릭
3. 인증번호 입력
4. 크롤러에서 Enter 키 입력

### 윤리적 사용
- **적절한 사용**: 개인 연구, 회사 분석, 커리어 결정 지원
- **금지 사항**: 상업적 악용, 개인정보 수집, 과도한 서버 부하
- **준수 사항**: robots.txt 확인, 적절한 지연 시간 설정

### 성능 최적화 팁
- 헤드리스 모드로 실행하여 리소스 절약
- 배치 크기를 시스템 성능에 맞게 조정
- 적절한 대기 시간으로 서버 부하 방지

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

1. **데이터 수집** (`blind_review_crawler.py`)
2. **벡터화 및 저장** (`../migrate_reviews.py`)
3. **AI 분석 활용** (`../src/blindinsight/frontend/app.py`)

전체 프로세스에 대한 자세한 내용은 상위 디렉토리의 README.md를 참고하세요.