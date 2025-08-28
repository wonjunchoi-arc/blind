"""
블라인드 크롤링 실행 스크립트 (RAG 최적화 지원)

간편한 실행 도구:
- 설정 파일 기반 자동 크롤링
- RAG 최적화 JSON 자동 생성
- 회사별 맞춤 설정 적용
- 진행 상황 실시간 모니터링
- 에러 발생 시 자동 복구

사용법:
python run_crawler.py --company naver --pages 10
python run_crawler.py --company kakao --output custom_filename.xlsx --rag
python run_crawler.py --company samsung --pages 50 --headless --no-rag
python run_crawler.py --help  # 도움말 보기
"""
# run_crawler.py

import argparse
import sys
import os
from pathlib import Path

# 현재 스크립트의 디렉토리를 Python 경로에 추가
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from blind_review_crawler import BlindReviewCrawler
    from config import (
        get_company_config, 
        ensure_directories, 
        get_output_filepath,
        validate_config,
        get_rag_settings,
        is_rag_enabled,
        print_config_summary,
        CRAWLING_SETTINGS,
        OUTPUT_SETTINGS,
        RAG_SETTINGS
    )
except ImportError as e:
    print(f"필요한 모듈을 import할 수 없습니다: {e}")
    print("다음 파일들이 같은 디렉토리에 있는지 확인해주세요:")
    print("- blind_review_crawler.py")
    print("- config.py")
    sys.exit(1)

import logging

def setup_logging(verbose=False):
    """로깅 설정"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def parse_arguments():
    """명령행 인자 파싱 (RAG 옵션 추가)"""
    parser = argparse.ArgumentParser(
        description="블라인드 기업 리뷰 크롤링 도구 (RAG 최적화 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_crawler.py --company naver --pages 10
  python run_crawler.py --company kakao --output kakao_reviews.xlsx --rag
  python run_crawler.py --company samsung --pages 50 --headless --no-rag

RAG 관련 옵션:
  --rag            RAG 최적화 JSON 생성 (기본값: config 설정 따름)
  --no-rag         RAG 기능 비활성화
  --rag-only       RAG JSON만 생성 (Excel 생략)

지원하는 회사:
  - naver: 네이버
  - kakao: 카카오  
  - samsung: 삼성전자
  - lg: LG전자
  
주의사항:
  - 크롤링 전 블라인드 로그인이 필요합니다
  - 과도한 요청으로 인한 IP 차단을 방지하기 위해 적절한 간격을 두고 실행하세요
  - RAG 최적화 시 처리 시간이 증가할 수 있습니다
        """
    )
    
    # 기본 인자들
    parser.add_argument(
        "--company", 
        type=str, 
        required=True,
        help="크롤링할 회사 (예: naver, kakao, samsung, lg)"
    )
    
    parser.add_argument(
        "--pages",
        type=int,
        help="크롤링할 페이지 수 (기본값: 설정 파일의 값)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="출력 파일명 (기본값: 설정 파일의 값)"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="헤드리스 모드로 실행 (브라우저 창 숨김)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=CRAWLING_SETTINGS["wait_timeout"],
        help=f"요소 대기 시간 (초, 기본값: {CRAWLING_SETTINGS['wait_timeout']})"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=CRAWLING_SETTINGS["between_pages_delay"],
        help=f"페이지 간 간격 (초, 기본값: {CRAWLING_SETTINGS['between_pages_delay']})"
    )
    
    # RAG 관련 인자들 (새로 추가)
    rag_group = parser.add_mutually_exclusive_group()
    rag_group.add_argument(
        "--rag",
        action="store_true",
        help="RAG 최적화 JSON 생성 활성화"
    )
    
    rag_group.add_argument(
        "--no-rag",
        action="store_true", 
        help="RAG 기능 완전 비활성화"
    )
    
    parser.add_argument(
        "--rag-only",
        action="store_true",
        help="RAG JSON만 생성 (기본 Excel 파일 생략)"
    )
    
    # 기타 옵션들
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세한 로그 출력"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true", 
        help="실제 크롤링 없이 설정만 확인"
    )
    
    parser.add_argument(
        "--config-summary",
        action="store_true",
        help="설정 요약 출력 후 종료"
    )
    
    return parser.parse_args()

def determine_rag_settings(args):
    """RAG 설정 결정"""
    # 명령행 인자가 우선순위 최고
    if args.no_rag:
        return {
            "enable_rag": False,
            "excel_output": True,
            "reason": "명령행에서 --no-rag 지정"
        }
    
    if args.rag:
        return {
            "enable_rag": True,
            "excel_output": not args.rag_only,
            "reason": "명령행에서 --rag 지정"
        }
    
    if args.rag_only:
        return {
            "enable_rag": True,
            "excel_output": False,
            "reason": "명령행에서 --rag-only 지정"
        }
    
    # 설정 파일 기본값 사용
    return {
        "enable_rag": is_rag_enabled(),
        "excel_output": True,
        "reason": "config.py 설정 사용"
    }

def validate_arguments(args, logger):
    """인자 유효성 검사 (RAG 검사 추가)"""
    try:
        # 회사 설정 확인
        company_config = get_company_config(args.company)
        logger.info(f"회사 설정 확인: {company_config['name']}")
        
        # 페이지 수 확인
        pages = args.pages if args.pages else company_config["max_pages"]
        if pages <= 0 or pages > 1000:  # 안전을 위한 상한선
            raise ValueError(f"페이지 수는 1-1000 사이여야 합니다: {pages}")
        
        # 타임아웃 확인
        if args.timeout <= 0 or args.timeout > 60:
            raise ValueError(f"타임아웃은 1-60초 사이여야 합니다: {args.timeout}")
        
        # 지연시간 확인  
        if args.delay < 0 or args.delay > 60:
            raise ValueError(f"지연시간은 0-60초 사이여야 합니다: {args.delay}")
        
        # RAG 설정 확인
        rag_settings = determine_rag_settings(args)
        logger.info(f"RAG 설정: {rag_settings['reason']}")
        
        return company_config, pages, rag_settings
        
    except KeyError as e:
        from config import COMPANIES
        available_companies = list(COMPANIES.keys())
        logger.error(f"지원하지 않는 회사: {args.company}")
        logger.error(f"사용 가능한 회사: {', '.join(available_companies)}")
        raise
    except ValueError as e:
        logger.error(f"잘못된 인자: {e}")
        raise

def print_banner():
    """시작 배너 출력 (RAG 지원 표시)"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                   🔍 블라인드 크롤링 도구 v2.0                  ║
║                        (RAG 최적화 지원)                       ║
║                                                              ║
║  📊 기업 리뷰 데이터를 효율적으로 수집하는 도구입니다           ║
║  🤖 RAG 시스템을 위한 최적화된 JSON 생성 기능                  ║
║  ⚠️  사용 전 블라인드 로그인이 필요합니다                      ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_summary(company_config, pages, output_files, rag_settings, headless):
    """크롤링 설정 요약 출력 (RAG 정보 포함)"""
    print("\n크롤링 설정 요약:")
    print("─" * 60)
    print(f"회사명: {company_config['name']}")
    print(f"URL: {company_config['url']}")
    print(f"페이지 수: {pages}")
    print(f"브라우저: {'헤드리스 모드' if headless else '일반 모드'}")
    print(f"RAG 최적화: {'활성화' if rag_settings['enable_rag'] else '비활성화'}")
    
    # 출력 파일 정보
    print("\n생성될 파일:")
    if rag_settings['excel_output']:
        print(f"  - Excel: {output_files.get('excel', 'N/A')}")
    if rag_settings['enable_rag']:
        print(f"  - RAG JSON: {output_files.get('rag_json', 'N/A')}")
    
    if rag_settings['enable_rag']:
        rag_config = get_rag_settings()
        print(f"\nRAG 기능:")
        print(f"  - 키워드 추출: {'활성화' if rag_config['keyword_extraction']['enable'] else '비활성화'}")
        print(f"  - 감성 분석: {'활성화' if rag_config['sentiment_analysis']['enable'] else '비활성화'}")
        print(f"  - 의미적 콘텐츠: {'생성' if rag_config['content_generation']['semantic_content'] else '미생성'}")
        print(f"  - 인텔리전스 플래그: {'활성화' if rag_config['metadata_enrichment']['intelligence_flags'] else '비활성화'}")
    
    print("─" * 60)

def create_enhanced_crawler(args, rag_settings):
    """향상된 크롤러 인스턴스 생성"""
    
    # 기본 크롤러 생성
    crawler = BlindReviewCrawler(
        headless=args.headless,
        wait_timeout=args.timeout
    )
    
    # RAG 설정이 비활성화된 경우 기본 크롤러 반환
    if not rag_settings['enable_rag']:
        # RAG 프로세서 비활성화
        crawler.rag_processor = None
    
    return crawler

def execute_crawling(crawler, company_config, pages, company_key, rag_settings, output_files):
    """크롤링 실행 및 결과 저장"""
    
    # 크롤링 실행
    crawler.crawl_reviews(
        base_url=company_config["url"],
        last_page=pages,
        company_name=company_key
    )
    
    return True

def main():
    """메인 실행 함수"""
    # 배너 출력
    print_banner()
    
    # 로깅 설정
    logger = setup_logging()
    
    try:
        # 명령행 인자 파싱
        args = parse_arguments()
        
        # 설정 요약만 출력하고 종료
        if args.config_summary:
            print_config_summary()
            return
        
        # 상세 로깅 설정
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("상세 로깅 모드 활성화")
        
        # 설정 검증
        logger.info("설정 검증 중...")
        validate_config()
        ensure_directories()
        
        # 인자 유효성 검사
        company_config, pages, rag_settings = validate_arguments(args, logger)
        
        # 출력 파일 경로 설정
        output_files = {}
        
        if rag_settings['excel_output']:
            output_files['excel'] = get_output_filepath(
                args.company, 
                file_type="basic",
                custom_filename=args.output
            )
        
        if rag_settings['enable_rag']:
            output_files['rag_json'] = get_output_filepath(
                args.company,
                file_type="rag"
            )
        
        # 설정 요약 출력
        print_summary(company_config, pages, output_files, rag_settings, args.headless)
        
        # Dry run 모드인 경우 여기서 종료
        if args.dry_run:
            logger.info("Dry run 완료 - 실제 크롤링은 실행되지 않았습니다")
            print(f"\nRAG 설정 상세:")
            print(f"  - 활성화 이유: {rag_settings['reason']}")
            if rag_settings['enable_rag']:
                rag_config = get_rag_settings()
                print(f"  - 키워드 맥락 길이: {rag_config['keyword_extraction']['context_length']}")
                print(f"  - 감성 분석 방법: {rag_config['sentiment_analysis']['method']}")
            return
        
        # 사용자 확인
        if not args.headless:
            response = input("\n계속 진행하시겠습니까? (y/N): ").strip().lower()
            if response not in ['y', 'yes', '예']:
                logger.info("사용자가 크롤링을 취소했습니다")
                return
        
        # 크롤러 초기화
        logger.info("크롤러 초기화 중...")
        crawler = create_enhanced_crawler(args, rag_settings)
        
        # 페이지 간 간격 설정 (동적으로 업데이트)
        if hasattr(crawler, 'between_pages_delay'):
            crawler.between_pages_delay = args.delay
        
        logger.info("크롤링 시작...")
        
        # 크롤링 실행
        success = execute_crawling(
            crawler, company_config, pages, args.company, 
            rag_settings, output_files
        )
        
        if success:
            logger.info("크롤링 완료!")
            print(f"\n크롤링이 성공적으로 완료되었습니다!")
            
            # 생성된 파일 정보 출력
            print(f"\n생성된 파일:")
            if rag_settings['excel_output'] and 'excel' in output_files:
                print(f"  - 기본 Excel: {output_files['excel']}")
                
            if rag_settings['enable_rag'] and 'rag_json' in output_files:
                print(f"  - RAG JSON: {output_files['rag_json']}")
                print(f"    * 벡터 검색에 최적화된 구조")
                print(f"    * 청킹에 적합한 데이터 형태")
                print(f"    * 키워드 및 감성 분석 포함")
            
            # RAG 활용 가이드
            if rag_settings['enable_rag']:
                print(f"\nRAG 시스템 활용 가이드:")
                print(f"  1. JSON 파일을 벡터 DB에 직접 로드 가능")
                print(f"  2. content_semantic: 벡터 검색용")
                print(f"  3. content_keyword: 키워드 검색용") 
                print(f"  4. extracted_keywords: 추출된 키워드 목록")
                print(f"  5. intelligence_flags: 고급 분석 플래그")
        
    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단되었습니다")
        sys.exit(1)
    except Exception as e:
        logger.error(f"크롤링 실행 중 오류 발생: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        # 리소스 정리
        if 'crawler' in locals():
            crawler.close()
            logger.info("리소스 정리 완료")

if __name__ == "__main__":
    main()