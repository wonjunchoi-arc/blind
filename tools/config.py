"""
블라인드 크롤링 설정 파일 (RAG 최적화 지원)

주요 설정:
- 크롤링 대상 URL 및 페이지 설정
- RAG 최적화 옵션 추가
- 브라우저 옵션 및 성능 튜닝
- 출력 파일 형식 및 위치 설정
- 에러 처리 및 재시도 설정

사용법:
1. COMPANIES 딕셔너리에 크롤링할 회사 정보 추가
2. RAG_SETTINGS에서 RAG 최적화 옵션 조정  
3. OUTPUT_SETTINGS에서 결과 저장 방식 설정
"""
# config.py

from datetime import datetime
import os

# =============================================================================
# 회사별 크롤링 대상 설정
# =============================================================================

COMPANIES = {
    # 회사별 설정 예시 (실제 사용 시 ### 부분을 회사 코드로 변경)
    "naver": {
        "name": "네이버",
        "company_code": "NAVER",  # 실제 블라인드 회사 코드로 변경
        "url": "https://www.teamblind.com/kr/company/NAVER/reviews",
        "max_pages": 50,
        "output_file": f"blind_review_naver_{datetime.now().strftime('%Y%m%d')}.xlsx"
    },
    "kakao": {
        "name": "카카오",
        "company_code": "KAKAO",  # 실제 블라인드 회사 코드로 변경
        "url": "https://www.teamblind.com/kr/company/KAKAO/reviews", 
        "max_pages": 30,
        "output_file": f"blind_review_kakao_{datetime.now().strftime('%Y%m%d')}.xlsx"
    },
    "samsung": {
        "name": "삼성전자",
        "company_code": "SAMSUNG",  # 실제 블라인드 회사 코드로 변경
        "url": "https://www.teamblind.com/kr/company/삼성전자/reviews",
        "max_pages": 50,
        "output_file": f"blind_review_samsung_{datetime.now().strftime('%Y%m%d')}.xlsx"
    },
    "lg": {
        "name": "LG전자",
        "company_code": "LG",
        "url": "https://www.teamblind.com/kr/company/LG/reviews",
        "max_pages": 40,
        "output_file": f"blind_review_lg_{datetime.now().strftime('%Y%m%d')}.xlsx"
    }
    # 필요에 따라 더 많은 회사 추가 가능
}
# =============================================================================
# RAG 최적화 설정 (새로 추가)
# =============================================================================

RAG_SETTINGS = {
    # RAG 기능 활성화
    "enable_rag_optimization": True,  # True: RAG 최적화 JSON 생성, False: 기본 Excel만
    
    # 키워드 추출 설정
    "keyword_extraction": {
        "enable": True,
        "context_length": 15,  # 키워드 주변 맥락 길이
        "min_frequency": 1,    # 최소 빈도
        "max_keywords_per_review": 20  # 리뷰당 최대 키워드 수
    },
    
    # 감성 분석 설정
    "sentiment_analysis": {
        "enable": True,
        "method": "keyword_based",  # keyword_based, advanced (향후 확장)
        "positive_threshold": 0.1,  # 긍정 판단 임계값
        "negative_threshold": -0.1  # 부정 판단 임계값
    },
    
    # 콘텐츠 생성 설정
    "content_generation": {
        "semantic_content": True,   # 벡터 검색용 콘텐츠 생성
        "keyword_content": True,    # 키워드 검색용 콘텐츠 생성
        "max_content_length": 2000  # 최대 콘텐츠 길이
    },
    
    # 메타데이터 설정
    "metadata_enrichment": {
        "search_metadata": True,    # 검색 메타데이터 생성
        "intelligence_flags": True, # 인텔리전스 플래그 생성
        "rating_categorization": True  # 평점 카테고리화
    },
    
    # 청킹 최적화 설정
    "chunking_optimization": {
        "enable": True,
        "chunk_types": ["summary", "pros", "cons", "category"],  # 청크 유형
        "min_chunk_length": 50,   # 최소 청크 길이
        "max_chunk_length": 500   # 최대 청크 길이
    }
}

# =============================================================================
# 크롤링 동작 설정
# =============================================================================

CRAWLING_SETTINGS = {
    # 브라우저 설정
    "headless": False,  # True: 브라우저 창 숨김, False: 브라우저 창 표시
    "window_size": (1920, 1080),  # 브라우저 창 크기
    "wait_timeout": 10,  # 요소 대기 시간 (초)
    
    # 페이지 로딩 설정
    "page_load_delay": 3,  # 페이지 로딩 후 대기 시간 (초)
    "element_load_delay": 0.5,  # 요소 클릭 후 대기 시간 (초)
    "between_pages_delay": 2,  # 페이지 간 간격 (초)
    
    # 재시도 설정  
    "max_retries": 3,  # 실패 시 최대 재시도 횟수
    "retry_delay": 5,  # 재시도 간 대기 시간 (초)
    
    # 성능 설정
    "batch_size": 10,  # 배치 단위로 처리할 리뷰 수
    "memory_limit": 1000,  # 메모리 제한 (MB)
    
    # 안전 설정
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "request_headers": {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.8,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }
}

# =============================================================================
# 출력 및 저장 설정 (RAG 지원)
# =============================================================================

OUTPUT_SETTINGS = {
    # 파일 저장 설정
    "output_directory": "./data/reviews/",  # 출력 디렉토리
    "file_format": "xlsx",  # 기본 파일 형식: excel, csv
    
    # Excel 설정
    "excel_engine": "openpyxl",  # Excel 엔진
    "include_index": False,  # 인덱스 포함 여부
    "sheet_name": "Reviews",  # 시트명
    
    # RAG JSON 설정 (새로 추가)
    "rag_json": {
        "enable": True,  # RAG JSON 생성 여부
        "indent": 2,     # JSON 들여쓰기
        "ensure_ascii": False,  # ASCII 인코딩 강제 여부
        "filename_suffix": "_rag_optimized"  # 파일명 접미사
    },
    
    # 파일 네이밍 설정
    "filename_patterns": {
        "basic_excel": "{company}_basic_{date}.xlsx",
        "rag_json": "{company}_rag_optimized_{date}.json",
        "analysis_csv": "{company}_analysis_{date}.csv"  # 향후 분석용
    },
    
    # 데이터 설정
    "column_names": [
        "총점", "커리어 향상", "업무와 삶의 균형", "급여 및 복지",
        "사내 문화", "경영진", "제목", "직원유형", "장점", "단점",
        "크롤링일시", "페이지번호"  # 추가 메타데이터
    ],
    
    # 백업 설정
    "create_backup": True,  # 기존 파일 백업 여부
    "backup_suffix": "_backup",  # 백업 파일 접미사
}

# =============================================================================
# 로깅 설정
# =============================================================================

LOGGING_SETTINGS = {
    # 로그 레벨 설정
    "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "console_log": True,  # 콘솔 로그 출력 여부
    "file_log": True,  # 파일 로그 저장 여부
    
    # 로그 파일 설정
    "log_directory": "./logs/",  # 로그 디렉토리
    "log_filename": f"blind_crawler_{datetime.now().strftime('%Y%m%d')}.log",
    "log_encoding": "utf-8",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_date_format": "%Y-%m-%d %H:%M:%S",
    
    # 로그 순환 설정
    "rotate_logs": True,  # 로그 파일 순환 여부
    "max_log_size": 10 * 1024 * 1024,  # 최대 로그 파일 크기 (10MB)
    "backup_count": 5,  # 백업 로그 파일 수
}

# =============================================================================
# 셀렉터 설정 (웹사이트 구조 변경 시 수정)
# =============================================================================

SELECTORS = {
    # 로그인 관련
    "login_button": ".btn_signin",
    
    # 리뷰 관련 (수정됨)
    "review_items": ".review_item", 
    "rating_section": ".rating",
    "more_rating_button": ".more_rating",
    "total_score": ".rating .num",
    "detail_scores": "div.review_item_inr > div.rating > div.more_rating > span > span > span > div.ly_rating > div.rating_wp > span.desc > i.blind",
    "review_title": "div.review_item_inr > h3.rvtit > a",
    "employee_status": "div.review_item_inr > div.auth",
    
    # 장점/단점 셀렉터 (새로 수정)
    "pros_section": "strong.sbt:contains('장점')",
    "cons_section": "strong.sbt:contains('단점')",
    "parag_content": ".parag p",
}

# =============================================================================
# 유틸리티 함수
# =============================================================================

def get_company_config(company_key):
    """
    회사별 설정 정보를 반환
    
    Args:
        company_key: 회사 키 (예: "naver", "kakao")
        
    Returns:
        dict: 회사 설정 정보
        
    Raises:
        KeyError: 존재하지 않는 회사 키인 경우
    """
    if company_key not in COMPANIES:
        available_companies = list(COMPANIES.keys())
        raise KeyError(f"'{company_key}' 회사 설정을 찾을 수 없습니다. 사용 가능한 회사: {available_companies}")
    
    return COMPANIES[company_key]

def get_rag_settings():
    """RAG 설정 반환"""
    return RAG_SETTINGS

def is_rag_enabled():
    """RAG 기능 활성화 여부 확인"""
    return RAG_SETTINGS.get("enable_rag_optimization", False)

def ensure_directories():
    """필요한 디렉토리들을 생성"""
    directories = [
        OUTPUT_SETTINGS["output_directory"],
        LOGGING_SETTINGS["log_directory"]
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"디렉토리 생성: {directory}")

def get_output_filepath(company_key, file_type="basic", custom_filename=None):
    """
    출력 파일 경로를 생성
    
    Args:
        company_key: 회사 키
        file_type: 파일 유형 ("basic", "rag", "analysis")
        custom_filename: 커스텀 파일명 (선택사항)
        
    Returns:
        str: 전체 파일 경로
    """
    company_config = get_company_config(company_key)
    date_str = datetime.now().strftime('%Y%m%d')
    
    if custom_filename:
        filename = custom_filename
    else:
        patterns = OUTPUT_SETTINGS["filename_patterns"]
        
        if file_type == "basic":
            filename = patterns["basic_excel"].format(
                company=company_key, date=date_str
            )
        elif file_type == "rag":
            filename = patterns["rag_json"].format(
                company=company_key, date=date_str
            )
        elif file_type == "analysis":
            filename = patterns["analysis_csv"].format(
                company=company_key, date=date_str
            )
        else:
            filename = company_config["output_file"]
    
    return os.path.join(OUTPUT_SETTINGS["output_directory"], filename)

# =============================================================================
# 검증 함수
# =============================================================================

def validate_config():
    """설정값 검증"""
    errors = []
    
    # 필수 설정 확인
    if not COMPANIES:
        errors.append("COMPANIES 설정이 비어있습니다")
    
    # URL 형식 확인
    for company_key, config in COMPANIES.items():
        if not config.get("url", "").startswith("https://"):
            errors.append(f"{company_key}: 올바르지 않은 URL 형식")
        
        if not isinstance(config.get("max_pages", 0), int) or config.get("max_pages", 0) <= 0:
            errors.append(f"{company_key}: max_pages는 양의 정수여야 합니다")
    
    # 크롤링 설정 확인
    if CRAWLING_SETTINGS["wait_timeout"] <= 0:
        errors.append("wait_timeout은 양수여야 합니다")
    
    if CRAWLING_SETTINGS["max_retries"] < 0:
        errors.append("max_retries는 0 이상이어야 합니다")
    
    # RAG 설정 검증
    if RAG_SETTINGS["enable_rag_optimization"]:
        if RAG_SETTINGS["keyword_extraction"]["context_length"] <= 0:
            errors.append("keyword_extraction.context_length는 양수여야 합니다")
        
        if not (-1 <= RAG_SETTINGS["sentiment_analysis"]["positive_threshold"] <= 1):
            errors.append("sentiment_analysis.positive_threshold는 -1~1 사이여야 합니다")
    
    if errors:
        raise ValueError(f"설정 오류 발견:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True

def print_config_summary():
    """설정 요약 출력"""
    print("=" * 60)
    print("블라인드 크롤러 설정 요약")
    print("=" * 60)
    print(f"지원 회사: {', '.join(COMPANIES.keys())}")
    print(f"RAG 최적화: {'활성화' if is_rag_enabled() else '비활성화'}")
    print(f"출력 디렉토리: {OUTPUT_SETTINGS['output_directory']}")
    print(f"로그 디렉토리: {LOGGING_SETTINGS['log_directory']}")
    print("=" * 60)

# 초기화 시 검증 실행
if __name__ == "__main__":
    validate_config()
    ensure_directories()
    print_config_summary()
    print("✅ 설정 검증 완료")