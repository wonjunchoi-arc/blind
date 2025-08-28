# 🔍 블라인드 기업 리뷰 크롤링 도구 v2.0

블라인드(Blind) 웹사이트에서 기업 리뷰 데이터를 효율적으로 수집하고 **RAG(Retrieval-Augmented Generation) 시스템**에 최적화된 형태로 변환하는 도구입니다.

## ✨ 주요 기능

### 기본 크롤링 기능
- 🏢 **다중 기업 지원**: 네이버, 카카오, 삼성전자 등 주요 기업 리뷰 크롤링
- 📊 **상세 데이터 수집**: 총점, 세부 평점(커리어향상, 워라밸, 급여복지, 사내문화, 경영진), 제목, 장단점 등 완전한 리뷰 정보
- 🛡️ **안전한 크롤링**: 적절한 지연시간과 에러 처리로 안정적인 데이터 수집
- 📝 **상세 로깅**: 크롤링 과정 추적 및 디버깅 지원

### RAG 최적화 기능 (NEW!)
- 🤖 **RAG 시스템 연동**: 벡터 검색과 키워드 검색을 모두 지원하는 최적화된 JSON 구조
- 🔍 **고급 키워드 추출**: 카테고리별 키워드 분류 및 맥락(context) 보존
- 📈 **감성 분석**: 긍정/부정 감정 분석 및 점수화
- 🧠 **인텔리전스 플래그**: 야근, 갑질, 업무압박 등 특수 상황 자동 탐지
- 📄 **의미적 콘텐츠 생성**: 벡터 임베딩에 최적화된 자연어 형태 변환
- 💾 **이중 출력**: 기존 Excel 파일 + RAG 최적화 JSON 파일 동시 생성

## 📋 필요 조건

### 시스템 요구사항
- Python 3.8 이상
- Chrome 브라우저
- 블라인드 계정 (웹 로그인 가능)

### Python 패키지
```bash
pip install -r requirements.txt
```

주요 패키지:
- `selenium>=4.15.0`: 웹 자동화
- `pandas>=2.0.0`: 데이터 처리
- `openpyxl>=3.1.0`: Excel 파일 생성
- `beautifulsoup4>=4.12.0`: HTML 파싱

## 🚀 사용법

### 1. 기본 실행 (Excel + RAG JSON 모두 생성)
```bash
# 네이버 리뷰 10페이지 크롤링 (기본 RAG 활성화)
python run_crawler.py --company naver --pages 10

# 카카오 리뷰 20페이지 크롤링 (헤드리스 모드)
python run_crawler.py --company kakao --pages 20 --headless
```

### 2. RAG 전용 실행
```bash
# RAG JSON만 생성 (Excel 생략)
python run_crawler.py --company samsung --pages 30 --rag-only

# RAG 기능 강제 활성화
python run_crawler.py --company naver --pages 15 --rag
```

### 3. 기존 방식 (RAG 비활성화)
```bash
# Excel 파일만 생성 (RAG 기능 비활성화)
python run_crawler.py --company lg --pages 25 --no-rag

# 커스텀 파일명으로 저장
python run_crawler.py --company samsung --pages 50 --output samsung_reviews_2024.xlsx --no-rag
```

### 4. 고급 옵션
```bash
# 상세 로그와 함께 실행
python run_crawler.py --company naver --pages 5 --verbose

# 설정만 확인 (실제 크롤링 안함)
python run_crawler.py --company kakao --pages 10 --dry-run

# 타임아웃과 지연시간 조정
python run_crawler.py --company naver --pages 10 --timeout 15 --delay 3

# 설정 요약 보기
python run_crawler.py --config-summary
```

### 5. 직접 코드 실행
```python
from blind_review_crawler import BlindReviewCrawler

# 크롤러 초기화 (RAG 활성화)
crawler = BlindReviewCrawler(headless=False, wait_timeout=10)

# 리뷰 크롤링 (자동으로 Excel + RAG JSON 생성)
crawler.crawl_reviews(
    base_url="https://www.teamblind.com/kr/company/NAVER/reviews",
    last_page=10,
    company_name="naver"
)

# 리소스 정리
crawler.close()
```

## ⚙️ 설정

### 회사 정보 추가 (`config.py`)
```python
COMPANIES = {
    "your_company": {
        "name": "회사명",
        "company_code": "COMPANY_CODE",  # 블라인드 회사 코드
        "url": "https://www.teamblind.com/kr/company/COMPANY_CODE/reviews",
        "max_pages": 30,
        "output_file": f"blind_review_company_{datetime.now().strftime('%Y%m%d')}.xlsx"
    }
}
```

### RAG 설정 조정 (`config.py`)
```python
RAG_SETTINGS = {
    "enable_rag_optimization": True,  # RAG 기능 활성화
    
    "keyword_extraction": {
        "enable": True,
        "context_length": 15,  # 키워드 주변 맥락 길이
        "min_frequency": 1,
        "max_keywords_per_review": 20
    },
    
    "sentiment_analysis": {
        "enable": True,
        "method": "keyword_based",
        "positive_threshold": 0.1,
        "negative_threshold": -0.1
    },
    
    "content_generation": {
        "semantic_content": True,   # 벡터 검색용 콘텐츠
        "keyword_content": True,    # 키워드 검색용 콘텐츠
        "max_content_length": 2000
    },
    
    "metadata_enrichment": {
        "search_metadata": True,    # 검색 메타데이터
        "intelligence_flags": True, # 인텔리전스 플래그
        "rating_categorization": True
    }
}
```

### 크롤링 설정 조정
```python
CRAWLING_SETTINGS = {
    "headless": False,  # 브라우저 창 표시 여부
    "wait_timeout": 10,  # 요소 대기 시간
    "page_load_delay": 3,  # 페이지 로딩 대기 시간
    "between_pages_delay": 2,  # 페이지 간 간격
    "max_retries": 3,  # 최대 재시도 횟수
}
```

## 📊 출력 데이터

### 1. 기본 Excel 파일 (예: `naver_basic_20250827.xlsx`)

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| 총점 | 전체 평점 | 4.2 |
| 커리어 향상 | 커리어 발전 점수 | 4.0 |
| 업무와 삶의 균형 | 워라밸 점수 | 3.5 |
| 급여 및 복지 | 연봉/복지 점수 | 4.5 |
| 사내 문화 | 회사 문화 점수 | 4.0 |
| 경영진 | 경영진 만족도 | 3.8 |
| 제목 | 리뷰 제목 | "좋은 회사입니다" |
| 직원유형 | 재직 상태 | "현직원" |
| 장점 | 회사 장점 | "복지가 좋음" |
| 단점 | 회사 단점 | "업무 강도 높음" |

### 2. RAG 최적화 JSON 파일 (예: `naver_rag_optimized_20250827.json`)

```json
{
  "metadata": {
    "company": "naver",
    "crawl_timestamp": "2025-08-27T14:30:00",
    "total_reviews": 100,
    "avg_rating": 3.8,
    "rating_distribution": {
      "high (4.0+)": 45,
      "medium (3.0-3.9)": 40,
      "low (<3.0)": 15
    }
  },
  "reviews": [
    {
      "id": "review_naver_0001",
      "raw_data": {
        "총점": 4.2,
        "제목": "개발자로 성장하기 좋은 환경",
        "직원유형": "퇴사자",
        "장점": "기술 스택이 최신이고...",
        "단점": "야근이 잦고..."
      },
      "processed_data": {
        "content_semantic": "naver에 대한 퇴사자 평가입니다...",
        "content_keyword": "회사: naver\n직원유형: 퇴사자...",
        "extracted_keywords": [
          {
            "keyword": "야근",
            "category": "worklife",
            "context": "야근이 잦고 업무 강도가",
            "frequency": 1
          }
        ],
        "sentiment_analysis": {
          "score": -0.033,
          "overall": "neutral"
        }
      },
      "search_metadata": {
        "has_worklife_mention": true,
        "has_salary_mention": true,
        "rating_level": "high"
      },
      "intelligence_flags": {
        "overtime": true,
        "toxic_culture": false,
        "work_pressure": true
      }
    }
  ]
}
```

## 🔧 RAG 최적화 처리 단계

### 3단계: 키워드 추출
미리 정의된 카테고리별 키워드 사전을 사용하여 텍스트에서 키워드를 찾고, 각 키워드의 주변 맥락(15자)을 함께 저장합니다.

```python
def extract_keywords_with_context(self, text: str):
    # 키워드 카테고리 사전
    keyword_categories = {
        "worklife": ["야근", "잔업", "오버타임", "워라밸", "퇴근시간", "휴가", "눈치", "밤샘"],
        "salary": ["연봉", "급여", "월급", "기본급", "상여금", "보너스", "인센티브"],
        "culture": ["분위기", "문화", "동료", "상사", "갑질", "회식", "조직문화"],
        "growth": ["성장", "승진", "교육", "경력", "기회", "발전", "커리어"],
        "benefits": ["복지", "보험", "스톡옵션", "주식", "식대", "교통비"],
        "management": ["경영진", "임원", "대표", "리더십", "비전"]
    }
    
    found_keywords = []
    for category, keywords in keyword_categories.items():
        for keyword in keywords:
            if keyword in text:
                context = self._extract_context(text, keyword, 15)  # 앞뒤 15자 추출
                found_keywords.append({
                    "keyword": keyword,
                    "category": category,
                    "context": context,
                    "frequency": text.count(keyword)
                })
```

**예시 결과:**
```json
{
  "keyword": "야근",
  "category": "worklife", 
  "context": "복지도 좋고 야근은 거의 없어요",
  "frequency": 1
}
```

### 4단계: 의미적 콘텐츠 생성
원본 리뷰 데이터를 벡터 임베딩에 최적화된 자연어 형태로 재구성합니다. 평점을 텍스트로 변환하고 구조화된 설명을 생성합니다.

```python
def create_semantic_content(self, review_data: Dict) -> str:
    rating_text = self._rating_to_text(review_data["총점"])
    
    semantic_content = f"""
{review_data["제목"]}

{review_data["회사"]}에 대한 {review_data["직원유형"]} 평가입니다.
총점 {review_data["총점"]}점으로 {rating_text} 수준의 만족도를 보입니다.

평가 세부사항:
- 커리어 향상: {review_data["커리어향상"]}점
- 업무와 삶의 균형: {review_data["워라밸"]}점  
- 급여 및 복지: {review_data["급여복지"]}점
- 사내 문화: {review_data["사내문화"]}점
- 경영진: {review_data["경영진"]}점

주요 장점: {review_data["장점"]}
주요 단점: {review_data["단점"]}

이 리뷰는 {review_data["직원유형"]}의 관점에서 회사의 전반적인 근무 환경과 조직 문화를 평가한 내용입니다.
    """.strip()

def _rating_to_text(self, score: float) -> str:
    if score >= 4.5: return "매우 높은"
    elif score >= 4.0: return "높은"
    elif score >= 3.5: return "보통 이상의"
    elif score >= 3.0: return "보통"
    else: return "아쉬운"
```

### 5단계: 감성 분석
미리 정의된 긍정/부정 키워드 목록을 사용하여 장점과 단점 텍스트를 분석하고 감성 점수를 계산합니다.

```python
def analyze_sentiment(self, pros_text: str, cons_text: str):
    positive_keywords = ["좋", "만족", "훌륭", "최고", "추천", "성장", "기회"]
    negative_keywords = ["힘들", "아쉬운", "문제", "스트레스", "갑질", "눈치"]
    
    # 긍정/부정 키워드 개수 세기
    positive_count = sum(1 for word in positive_keywords if word in pros_text)
    negative_count = sum(1 for word in negative_keywords if word in cons_text)
    
    # 전체 단어 수로 정규화
    total_words = len((pros_text + cons_text).split())
    sentiment_score = (positive_count - negative_count) / total_words
    
    # 전반적 감성 판단
    if sentiment_score > 0.1: overall = "positive"
    elif sentiment_score < -0.1: overall = "negative"
    else: overall = "neutral"
    
    return {
        "score": round(sentiment_score, 3),
        "positive_signals": positive_count,
        "negative_signals": negative_count,
        "overall": overall
    }
```

### 6단계: 인텔리전스 플래그 생성
특정 패턴을 나타내는 키워드 조합을 찾아서 불린 플래그를 생성합니다. 야근, 갑질, 업무압박 등의 특수 상황을 자동으로 탐지합니다.

```python
def create_intelligence_flags(self, review_data: Dict):
    full_text = f"{review_data['제목']} {review_data['장점']} {review_data['단점']}"
    
    # 인텔리전스 패턴 정의
    intelligence_patterns = {
        "overtime_indicators": ["야근", "잔업", "늦게까지", "밤늦게"],
        "toxic_culture_indicators": ["갑질", "눈치", "강압적", "일방적"],
        "salary_issues_indicators": ["연봉삭감", "급여문제", "인센티브없음"],
        "work_pressure_indicators": ["압박", "스트레스", "힘들", "무리한"]
    }
    
    flags = {}
    # 각 패턴별로 플래그 생성
    for flag_name, indicators in intelligence_patterns.items():
        flags[flag_name.replace("_indicators", "")] = any(
            indicator in full_text for indicator in indicators
        )
    
    # 복합 플래그 생성
    flags["has_negative_worklife"] = (
        flags.get("overtime", False) and 
        any(neg in full_text for neg in ["힘들", "스트레스", "문제"])
    )
    
    flags["has_positive_growth"] = (
        any(growth in full_text for growth in ["성장", "기회", "발전"]) and
        any(pos in full_text for pos in ["좋", "만족", "추천"])
    )
    
    return flags
```

**예시 결과:**
```json
{
  "overtime": true,           // "야근" 키워드 발견
  "toxic_culture": true,      // "갑질" 키워드 발견
  "salary_issues": false,     // 급여 관련 문제 키워드 없음
  "work_pressure": false,     // 업무 압박 키워드 없음
  "has_negative_worklife": false,  // 야근 + 부정어 조합 없음
  "has_positive_growth": false     // 성장 + 긍정어 조합 없음
}
```


## 🛡️ 안전한 사용을 위한 가이드라인

### 1. 적절한 지연시간 설정
```bash
# 서버 부하 방지를 위해 페이지 간 2-5초 간격 권장
python run_crawler.py --company naver --pages 10 --delay 3
```

### 2. 배치 크롤링
```bash
# 대량 데이터는 여러 번에 나누어 크롤링
python run_crawler.py --company samsung --pages 20  # 1차
# 잠시 대기 후
python run_crawler.py --company samsung --pages 20 --output samsung_part2.xlsx  # 2차
```

### 3. RAG 처리 성능 고려
```bash
# RAG 처리로 인한 시간 증가 고려
python run_crawler.py --company naver --pages 5 --rag --verbose  # 소량으로 테스트

# 대량 처리 시 RAG 비활성화 후 별도 처리 고려
python run_crawler.py --company samsung --pages 100 --no-rag
```

## 🔧 문제 해결

### 자주 발생하는 문제

1. **ChromeDriver 오류**
   - Chrome 버전과 ChromeDriver 버전 일치 확인
   - 자동 업데이트된 Chrome과 수동 설치한 ChromeDriver 간 버전 차이 해결

2. **로그인 실패**
   - 블라인드 앱에서 로그인 상태 확인
   - "더보기" → "블라인드 웹 로그인" 다시 시도
   - 인증번호 입력 시간 충분히 대기

3. **RAG JSON 생성 실패**
   ```bash
   # RAG 관련 에러 확인
   python run_crawler.py --company naver --pages 1 --rag --verbose
   
   # RAG 기능 비활성화 후 기본 크롤링 확인
   python run_crawler.py --company naver --pages 1 --no-rag
   ```

4. **메모리 부족 (RAG 처리 시)**
   ```bash
   # 페이지 수를 줄이고 여러 번 실행
   python run_crawler.py --company naver --pages 10 --rag
   # 또는 RAG 기능 일부만 활성화 (config.py에서 조정)
   ```

5. **빈 데이터 또는 일부 데이터 누락**
   ```bash
   # 타임아웃 시간 증가
   python run_crawler.py --company naver --pages 5 --timeout 15 --delay 5
   ```

### 디버깅

```bash
# 상세 로그로 문제 진단
python run_crawler.py --company naver --pages 1 --verbose --rag

# 설정만 확인
python run_crawler.py --company naver --pages 1 --dry-run --rag

# 설정 요약 확인
python run_crawler.py --config-summary
```

## 📁 파일 구조

```
blind-crawler/
├── blind_review_crawler.py    # 메인 크롤링 클래스 (RAG 프로세서 포함)
├── config.py                  # 설정 파일 (RAG 설정 포함)
├── run_crawler.py            # 실행 스크립트 (RAG 옵션 지원)
├── requirements.txt          # 패키지 의존성
├── README.md                # 이 파일
├── data/                    # 크롤링 결과 저장
│   └── reviews/
│       ├── company_basic_YYYYMMDD.xlsx        # 기본 Excel 파일
│       └── company_rag_optimized_YYYYMMDD.json # RAG JSON 파일
├── logs/                    # 로그 파일 저장
│   └── blind_crawler_YYYYMMDD.log
└── examples/                # 사용 예시
    ├── sample_usage.py
    └── rag_integration.py   # RAG 시스템 연동 예시
```

## 📝 로그 확인

크롤링 과정은 다음 위치에 로그로 저장됩니다:
- 콘솔: 실시간 진행 상황 (RAG 처리 단계 포함)
- 파일: `./logs/blind_crawler_YYYYMMDD.log`

### RAG 로그 예시
```
2025-08-27 14:30:15 - INFO - RAG 최적화 크롤링 시작: 총 10페이지
2025-08-27 14:30:20 - INFO - 페이지 1/10 처리 중
2025-08-27 14:30:25 - INFO - 페이지 1에서 5개 리뷰 발견
2025-08-27 14:30:26 - INFO - [1-1] 개발자로 성장하기 좋은 환경 | 총점: 3.8
2025-08-27 14:30:27 - INFO - RAG 키워드 추출: 8개 키워드 발견
2025-08-27 14:30:28 - INFO - 감성 분석 완료: neutral (score: -0.033)
2025-08-27 14:35:00 - INFO - RAG 최적화 JSON 저장 완료: naver_rag_optimized_20250827.json
```

## 🎯 성능 및 처리 시간

| 작업 | 기본 모드 | RAG 모드 | 비고 |
|------|-----------|----------|------|
| 페이지당 처리 시간 | 5-10초 | 8-15초 | RAG 처리로 인한 추가 시간 |
| 리뷰당 처리 시간 | 1-2초 | 2-4초 | 키워드 추출, 감성 분석 포함 |
| 메모리 사용량 | ~100MB | ~200MB | JSON 구조 생성으로 인한 증가 |
| 10페이지 예상 시간 | 2-5분 | 5-10분 | 페이지당 리뷰 수에 따라 변동 |

## ⚠️ 주의사항

### 일반 사용 주의사항
1. **이용 약관 준수**: 블라인드 웹사이트의 이용 약관을 확인하고 준수하세요
2. **적절한 사용**: 과도한 요청으로 서버에 부하를 주지 마세요
3. **개인정보 보호**: 수집된 데이터의 개인정보를 적절히 보호하세요
4. **상업적 이용**: 상업적 목적으로 사용 전 법적 검토가 필요할 수 있습니다

### RAG 관련 주의사항
1. **처리 시간**: RAG 처리로 인해 크롤링 시간이 약 2배 증가할 수 있습니다
2. **메모리 사용량**: JSON 구조 생성으로 메모리 사용량이 증가합니다
3. **데이터 품질**: RAG 최적화 과정에서 일부 정보 손실이 있을 수 있으므로 원본 데이터도 함께 보존됩니다
4. **설정 조정**: RAG 기능별로 활성화/비활성화가 가능하므로 필요에 따라 조정하세요

## 🆕 버전 히스토리

### v2.0 (2025-08-27)
- RAG(Retrieval-Augmented Generation) 최적화 기능 추가
- 키워드 추출 및 카테고리 분류 시스템
- 감성 분석 및 인텔리전스 플래그 기능
- 벡터 검색과 키워드 검색을 위한 이중 콘텐츠 구조
- RAG 최적화 JSON 출력 기능
- 고급 메타데이터 및 검색 최적화 기능

### v1.0
- 기본 블라인드 리뷰 크롤링 기능
- Excel 출력 기능
- 다중 회사 지원
- 기본 에러 처리 및 로깅

## 🤝 기여

버그 리포트, 기능 요청, 코드 기여는 GitHub Issues를 통해 환영합니다.

### 개발 환경 설정
```bash
# 개발 의존성 설치
pip install -r requirements-dev.txt

# 테스트 실행
python -m pytest tests/

# 코드 품질 검사
flake8 .
black .
```

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다. 상업적 사용 시 별도 검토가 필요합니다.

---

**📞 지원이 필요하시나요?**
- 이슈 생성: GitHub Issues
- 설정 관련: `config.py` 파일 확인
- RAG 설정: `RAG_SETTINGS` 섹션 참조
- 로그 확인: `./logs/` 디렉토리
- 성능 최적화: `CRAWLING_SETTINGS` 조정