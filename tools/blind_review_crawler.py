"""
업그레이드된 블라인드 크롤링 도구 v2.0 - 하이브리드 청크 방식
장점/단점 별도 청크 생성으로 정밀한 RAG 최적화

주요 업그레이드:
- 제목, 장점, 단점을 독립 청크로 분리
- 카테고리별 장점/단점 청크 생성  
- 벡터 DB 최적화된 메타데이터 구조
- kss 기반 정확한 문장 분할
"""

#blind_review_crawler.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from bs4 import BeautifulSoup
import time
import pandas as pd
import json
import re
import logging
import os
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict

# 새로운 모듈 import
from category_processor import CategorySpecificProcessor, VectorDBOptimizer
from keyword_dictionary import korean_keywords

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blind_crawler_hybrid_v2.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HybridBlindReviewCrawler:
    """하이브리드 청크 방식 블라인드 리뷰 크롤러"""
    
    def __init__(self, headless=False, wait_timeout=10, output_dir="./data/hybrid_vectordb"):
        self.wait_timeout = wait_timeout
        self.driver = None
        self.wait = None
        self.output_dir = output_dir
        self.is_logged_in = False
        
        # 하이브리드 프로세서들 초기화
        self.category_processor = CategorySpecificProcessor()
        self.vectordb_optimizer = VectorDBOptimizer()
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 처리 결과 추적
        self.processing_results = {
            "total_reviews_processed": 0,
            "total_chunks_created": 0,
            "chunk_breakdown": {
                "title_chunks": 0,
                "pros_chunks": 0, 
                "cons_chunks": 0
            },
            "category_breakdown": {
                "career_growth": 0,
                "salary_benefits": 0,
                "work_life_balance": 0,
                "company_culture": 0,
                "management": 0
            },
            "failed_reviews": 0
        }
        
        # Chrome 드라이버 초기화
        self._initialize_driver(headless)
    
    def _initialize_driver(self, headless):
        """Chrome 드라이버 초기화"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        
        # 기본 필수 설정
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # User-Agent 설정
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # 자동화 탐지 우회 스크립트
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined,
                    });
                '''
            })
            
            self.wait = WebDriverWait(self.driver, self.wait_timeout)
            logger.info("Chrome 드라이버 초기화 완료 (하이브리드 v2.0)")
            
        except Exception as e:
            logger.error(f"드라이버 초기화 실패: {e}")
            raise
    
    def login_wait(self, url):
        """로그인 대기 및 처리"""
        try:
            logger.info(f"페이지 접속 중: {url}")
            self.driver.get(url)
            time.sleep(5)
            
            # 로그인 버튼 클릭 시도
            try:
                login_btn = self.wait.until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "btn_signin"))
                )
                self.driver.execute_script("arguments[0].click();", login_btn)
                logger.info("로그인 버튼 클릭 완료")
            except Exception as e:
                logger.warning(f"로그인 버튼 클릭 실패: {e}")
            
            time.sleep(3)
            
            print("\n" + "="*60)
            print("🔐 블라인드 로그인이 필요합니다")
            print("1. 블라인드 앱에서 로그인하세요")
            print("2. '더보기' → '블라인드 웹 로그인' 클릭") 
            print("3. 인증번호를 입력하세요")
            print("4. 로그인 완료 후 아무 키나 눌러주세요...")
            print("="*60)
            
            input("로그인 완료 후 Enter 키를 눌러주세요: ")
            self.is_logged_in = True
            logger.info("사용자 로그인 완료 확인")
            
        except Exception as e:
            logger.error(f"로그인 과정 중 오류 발생: {e}")
            raise
    
    def extract_review_data(self, element):
        """개별 리뷰에서 데이터 추출 (기존 로직 유지)"""
        try:
            # 평점 정보 추출
            rating_element = element.find_element(By.CLASS_NAME, "rating")
            
            # 상세 평점 보기 클릭
            try:
                more_rating_btn = rating_element.find_element(By.CLASS_NAME, "more_rating")
                more_rating_btn.click()
                time.sleep(0.5)
            except NoSuchElementException:
                logger.warning("상세 평점 버튼을 찾을 수 없습니다")
            
            # 총점 추출
            try:
                score_element = rating_element.find_element(By.CLASS_NAME, "num")
                total_score = float(score_element.text.split("\n")[1])
            except (ValueError, IndexError):
                total_score = 0.0
            
            # 상세 점수 추출
            detail_scores = ["0"] * 5
            try:
                detail_elements = element.find_elements(
                    By.CSS_SELECTOR, 
                    "div.review_item_inr > div.rating > div.more_rating > span > span > span > div.ly_rating > div.rating_wp > span.desc > i.blind"
                )
                for i, detail_elem in enumerate(detail_elements[:5]):
                    detail_scores[i] = detail_elem.text
            except Exception as e:
                logger.warning(f"상세 점수 추출 실패: {e}")
            
            # 제목 추출
            try:
                title_element = element.find_element(By.CSS_SELECTOR, "div.review_item_inr > h3.rvtit > a")
                title = self.clean_text(title_element.text)
            except NoSuchElementException:
                title = "제목 없음"
            
            # 직원 유형 추출
            try:
                status_element = element.find_element(By.CSS_SELECTOR, "div.review_item_inr > div.auth")
                status_text = status_element.text.split("\n")
                status = status_text[1] if len(status_text) > 1 else "정보 없음"
            except (NoSuchElementException, IndexError):
                status = "정보 없음"
            
            # 장점/단점 추출
            pros = "정보 없음"
            cons = "정보 없음"
            
            try:
                parag_element = element.find_element(By.CSS_SELECTOR, ".parag")
                full_text = parag_element.text
                
                lines = full_text.split('\n')
                
                pros_started = False
                cons_started = False
                pros_lines = []
                cons_lines = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if '장점' in line and len(line) < 10:
                        pros_started = True
                        cons_started = False
                        continue
                    elif '단점' in line and len(line) < 10:
                        cons_started = True
                        pros_started = False
                        continue
                    elif any(keyword in line for keyword in ['추천', '평점', '회사에게', '경영진에게']):
                        pros_started = False
                        cons_started = False
                        continue
                    
                    if pros_started and line:
                        pros_lines.append(line)
                    elif cons_started and line:
                        cons_lines.append(line)
                
                if pros_lines:
                    pros = self.clean_text(' '.join(pros_lines))
                if cons_lines:
                    cons = self.clean_text(' '.join(cons_lines))
                    
            except Exception as e:
                logger.warning(f"장점/단점 추출 실패: {e}")
            
            return [
                total_score,
                detail_scores[0],
                detail_scores[1], 
                detail_scores[2],
                detail_scores[3],
                detail_scores[4],
                title,
                status,
                pros,
                cons
            ]
            
        except Exception as e:
            logger.error(f"리뷰 데이터 추출 중 오류: {e}")
            return [0.0, "0", "0", "0", "0", "0", "오류", "오류", "추출 실패", "추출 실패"]
    
    def clean_text(self, text):
        """텍스트 정제 함수"""
        if not text:
            return ""
        
        pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'
        text = re.sub(pattern=pattern, repl='', string=text)
        
        pattern = '<[^>]*>'
        text = re.sub(pattern=pattern, repl='', string=text)
        
        pattern = '[^\w\s가-힣]'
        text = re.sub(pattern=pattern, repl=' ', string=text)
        
        text = text.replace('\r', '').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_single_review(self, raw_review: List, company_name: str, review_index: int) -> Dict[str, List]:
        """단일 리뷰를 하이브리드 청크 방식으로 처리"""
        
        # 리뷰 데이터 구조화
        review_data = {
            "회사": company_name,
            "총점": raw_review[0],
            "커리어향상": raw_review[1], 
            "워라밸": raw_review[2],
            "급여복지": raw_review[3],
            "사내문화": raw_review[4],
            "경영진": raw_review[5],
            "제목": raw_review[6],
            "직원유형_원본": raw_review[7],
            "장점": raw_review[8],
            "단점": raw_review[9],
            "id": f"review_{company_name}_{review_index:04d}"
        }
        
        # 하이브리드 청크 처리
        try:
            all_chunks = self.category_processor.process_review_by_categories(review_data)
            
            # 통계 업데이트
            self.processing_results["total_reviews_processed"] += 1
            
            # 청크 유형별 통계 업데이트
            self.processing_results["chunk_breakdown"]["title_chunks"] += len(all_chunks.get("title", []))
            self.processing_results["chunk_breakdown"]["pros_chunks"] += len(all_chunks.get("pros", []))
            self.processing_results["chunk_breakdown"]["cons_chunks"] += len(all_chunks.get("cons", []))
            
            # 카테고리별 통계 업데이트
            for chunk in all_chunks.get("pros", []) + all_chunks.get("cons", []):
                category = chunk.category
                if category in self.processing_results["category_breakdown"]:
                    self.processing_results["category_breakdown"][category] += 1
            
            total_chunks = len(all_chunks.get("title", [])) + len(all_chunks.get("pros", [])) + len(all_chunks.get("cons", []))
            self.processing_results["total_chunks_created"] += total_chunks
            
            return all_chunks
            
        except Exception as e:
            logger.error(f"리뷰 {review_index} 하이브리드 처리 실패: {e}")
            self.processing_results["failed_reviews"] += 1
            return {"title": [], "pros": [], "cons": []}
    
    def crawl_company_reviews(self, company_code: str, pages: int = 25):
        """회사 리뷰 크롤링 및 하이브리드 청크 분석"""
        
        logger.info(f"🚀 {company_code} 크롤링 시작 (하이브리드 v2.0)")
        
        base_url = f"https://www.teamblind.com/kr/company/{company_code}/reviews"
        
        if not self.is_logged_in:
            self.login_wait(base_url)
        
        # 전체 리뷰 수집
        all_reviews = []
        
        for page in range(1, pages + 1):
            try:
                target_url = f"{base_url}?page={page}"
                logger.info(f"📄 페이지 {page}/{pages} 처리 중")
                
                self.driver.get(target_url)
                time.sleep(3)
                
                review_elements = self.driver.find_elements(By.CLASS_NAME, "review_item")
                
                if not review_elements:
                    logger.warning(f"페이지 {page}에서 리뷰를 찾을 수 없습니다")
                    continue
                
                logger.info(f"페이지 {page}에서 {len(review_elements)}개 리뷰 발견")
                
                for idx, element in enumerate(review_elements):
                    try:
                        review_data = self.extract_review_data(element)
                        all_reviews.append(review_data)
                        print(f"[{page}-{idx+1}] {review_data[6]} | 총점: {review_data[0]}")
                    except Exception as e:
                        logger.error(f"페이지 {page}, 리뷰 {idx+1} 추출 실패: {e}")
                        continue
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"페이지 {page} 처리 중 오류: {e}")
                continue
        
        if not all_reviews:
            logger.warning("추출된 리뷰 데이터가 없습니다")
            return False
        
        logger.info(f"📊 총 {len(all_reviews)}개 리뷰 수집 완료. 하이브리드 청크 분석 시작...")
        
        # 하이브리드 청크 분석 및 저장
        return self._process_and_save_hybrid_chunks(company_code, all_reviews)
    
    def _process_and_save_hybrid_chunks(self, company_code: str, all_reviews: List) -> bool:
        """하이브리드 청크 처리 및 저장"""
        
        # 모든 청크를 수집할 리스트
        all_processed_chunks = {
            "title": [],
            "pros": [],
            "cons": []
        }
        
        # 각 리뷰를 하이브리드 청크로 처리
        total_reviews = len(all_reviews)
        
        for idx, raw_review in enumerate(all_reviews):
            try:
                chunk_groups = self.process_single_review(raw_review, company_code, idx)
                
                # 청크 그룹별로 수집
                for chunk_type, chunks in chunk_groups.items():
                    all_processed_chunks[chunk_type].extend(chunks)
                
                # 진행상황 출력
                if (idx + 1) % 10 == 0:
                    progress = (idx + 1) / total_reviews * 100
                    logger.info(f"⚡ 하이브리드 처리 진행률: {progress:.1f}% ({idx + 1}/{total_reviews})")
                    
            except Exception as e:
                logger.error(f"리뷰 {idx} 하이브리드 처리 실패: {e}")
                continue
        
        logger.info(f"🎯 하이브리드 청크 분석 완료. 벡터 DB 최적화 시작...")
        
        # 벡터 DB 최적화 및 저장
        optimized_chunks = self.vectordb_optimizer.optimize_chunks_for_vectordb(all_processed_chunks)
        
        # 파일 저장
        file_path = self.vectordb_optimizer.save_vectordb_file(
            company_code, optimized_chunks, self.output_dir
        )
        
        # 결과 요약 출력
        self._print_hybrid_processing_summary(company_code, file_path)
        
        return True
    
    def _print_hybrid_processing_summary(self, company_code: str, file_path: str):
        """하이브리드 처리 결과 요약 출력"""
        
        print(f"\n{'='*80}")
        print(f"🎉 {company_code} 하이브리드 크롤링 및 분석 완료")
        print(f"{'='*80}")
        
        # 처리 통계
        stats = self.processing_results
        total_chunks = stats["total_chunks_created"]
        
        print(f"📊 처리 통계:")
        print(f"  - 처리된 리뷰: {stats['total_reviews_processed']}개")
        print(f"  - 실패한 리뷰: {stats['failed_reviews']}개")
        print(f"  - 생성된 총 청크: {total_chunks}개")
        
        # 청크 유형별 통계
        chunk_breakdown = stats["chunk_breakdown"]
        print(f"\n🏷️ 청크 유형별 통계:")
        print(f"  - 제목 청크: {chunk_breakdown['title_chunks']}개")
        print(f"  - 장점 청크: {chunk_breakdown['pros_chunks']}개")
        print(f"  - 단점 청크: {chunk_breakdown['cons_chunks']}개")
        
        # 카테고리별 통계
        category_breakdown = stats["category_breakdown"]
        print(f"\n📈 카테고리별 청크 수:")
        category_names_kr = {
            "career_growth": "커리어 향상",
            "salary_benefits": "급여 및 복지", 
            "work_life_balance": "업무와 삶의 균형",
            "company_culture": "사내 문화",
            "management": "경영진"
        }
        
        for category, count in category_breakdown.items():
            category_kr = category_names_kr.get(category, category)
            print(f"  - {category_kr}: {count}개")
        
        # 생성된 파일
        print(f"\n📁 생성된 벡터 DB 파일:")
        print(f"  - {os.path.basename(file_path)}")
        
        # 평균 통계
        avg_chunks_per_review = total_chunks / stats["total_reviews_processed"] if stats["total_reviews_processed"] > 0 else 0
        print(f"\n⭐ 통계 요약:")
        print(f"  - 리뷰당 평균 청크: {avg_chunks_per_review:.1f}개")
        print(f"  - 처리 성공률: {(stats['total_reviews_processed'] / (stats['total_reviews_processed'] + stats['failed_reviews']) * 100):.1f}%")
        
        print(f"\n💡 하이브리드 청크 방식으로 장점/단점이 명확히 분리되어")
        print(f"   AI Agent가 더 정확한 답변을 제공할 수 있습니다!")
        print(f"{'='*80}\n")
    
    def close(self):
        """드라이버 종료"""
        if self.driver:
            self.driver.quit()
            logger.info("드라이버 종료 완료")


def run_single_company_hybrid_crawl(company_code: str, pages: int = 25, headless: bool = False):
    """단일 기업 하이브리드 크롤링 실행"""
    
    print(f"\n블라인드 하이브리드 크롤러 v2.0")
    print(f"대상 기업: {company_code}")
    print(f"크롤링 페이지: {pages}")
    print(f"모드: {'헤드리스' if headless else '일반'}")
    print(f"청크 방식: 장점/단점 분리 하이브리드")
    
    crawler = HybridBlindReviewCrawler(
        headless=headless,
        output_dir="./data/hybrid_vectordb"
    )
    
    try:
        success = crawler.crawl_company_reviews(company_code, pages)
        
        if success:
            print(f"\n{company_code} 하이브리드 크롤링 완료!")
            print(f"출력 디렉토리: {crawler.output_dir}")
            print(f"장점/단점이 분리된 정밀 청크로 저장되었습니다.")
        else:
            print(f"{company_code} 크롤링 실패")
            
        return success
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 크롤링이 중단되었습니다.")
        return False
    except Exception as e:
        logger.error(f"크롤링 실행 중 오류: {e}")
        return False
    finally:
        crawler.close()


# 기존 코드에 추가할 함수들

def run_multiple_companies_hybrid_crawl(company_list: List[str], pages: int = 25, headless: bool = False, delay_between_companies: int = 30):
    """여러 기업을 순차적으로 하이브리드 크롤링 실행"""
    
    print(f"\n블라인드 다중 기업 하이브리드 크롤러 v2.0")
    print(f"대상 기업: {len(company_list)}개")
    print(f"크롤링 페이지: {pages}")
    print(f"모드: {'헤드리스' if headless else '일반'}")
    print(f"기업간 대기시간: {delay_between_companies}초")
    print(f"청크 방식: 장점/단점 분리 하이브리드")
    
    # 크롤링 결과 추적
    crawling_results = {
        "success": [],
        "failed": [],
        "total_companies": len(company_list),
        "start_time": datetime.now()
    }
    
    # 단일 크롤러 인스턴스 생성 (로그인 한번만 하기 위함)
    crawler = HybridBlindReviewCrawler(
        headless=headless,
        output_dir="./data/hybrid_vectordb"
    )
    
    try:
        print(f"\n{'='*80}")
        print(f"🚀 다중 기업 크롤링 시작 - 총 {len(company_list)}개 기업")
        print(f"{'='*80}")
        
        for idx, company_code in enumerate(company_list):
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{idx+1}/{len(company_list)}] {current_time} - {company_code} 크롤링 시작")
                
                # 개별 크롤링 통계 초기화
                crawler.processing_results = {
                    "total_reviews_processed": 0,
                    "total_chunks_created": 0,
                    "chunk_breakdown": {
                        "title_chunks": 0,
                        "pros_chunks": 0, 
                        "cons_chunks": 0
                    },
                    "category_breakdown": {
                        "career_growth": 0,
                        "salary_benefits": 0,
                        "work_life_balance": 0,
                        "company_culture": 0,
                        "management": 0
                    },
                    "failed_reviews": 0
                }
                
                success = crawler.crawl_company_reviews(company_code, pages)
                
                if success:
                    crawling_results["success"].append(company_code)
                    print(f"✅ {company_code} 크롤링 완료")
                else:
                    crawling_results["failed"].append(company_code)
                    print(f"❌ {company_code} 크롤링 실패")
                
                # 마지막 기업이 아니라면 대기
                if idx < len(company_list) - 1:
                    print(f"⏱️ 다음 기업 크롤링까지 {delay_between_companies}초 대기 중...")
                    time.sleep(delay_between_companies)
                    
            except Exception as e:
                logger.error(f"{company_code} 크롤링 중 오류: {e}")
                crawling_results["failed"].append(company_code)
                print(f"❌ {company_code} 크롤링 실패: {e}")
                
                # 오류 발생시에도 대기 (서버 부하 방지)
                if idx < len(company_list) - 1:
                    print(f"⏱️ 오류 발생으로 인한 복구 대기: {delay_between_companies}초")
                    time.sleep(delay_between_companies)
        
        # 전체 크롤링 결과 요약
        _print_multiple_crawling_summary(crawling_results)
        
        return crawling_results
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 다중 크롤링이 중단되었습니다.")
        _print_multiple_crawling_summary(crawling_results)
        return crawling_results
    except Exception as e:
        logger.error(f"다중 크롤링 실행 중 오류: {e}")
        return crawling_results
    finally:
        crawler.close()


def _print_multiple_crawling_summary(results: Dict):
    """다중 크롤링 결과 요약 출력"""
    
    end_time = datetime.now()
    total_time = end_time - results["start_time"]
    
    print(f"\n{'='*80}")
    print(f"🏁 다중 기업 하이브리드 크롤링 완료")
    print(f"{'='*80}")
    
    print(f"📊 전체 크롤링 결과:")
    print(f"  - 총 대상 기업: {results['total_companies']}개")
    print(f"  - 성공한 기업: {len(results['success'])}개")
    print(f"  - 실패한 기업: {len(results['failed'])}개")
    print(f"  - 성공률: {len(results['success'])/results['total_companies']*100:.1f}%")
    print(f"  - 총 소요시간: {total_time}")
    
    if results["success"]:
        print(f"\n✅ 성공한 기업 ({len(results['success'])}개):")
        for i, company in enumerate(results["success"]):
            print(f"  {i+1:2d}. {company}")
    
    if results["failed"]:
        print(f"\n❌ 실패한 기업 ({len(results['failed'])}개):")
        for i, company in enumerate(results["failed"]):
            print(f"  {i+1:2d}. {company}")
        
        print(f"\n💡 실패한 기업들은 다음과 같은 이유로 실패할 수 있습니다:")
        print(f"  - 해당 기업의 리뷰가 없거나 접근 불가")
        print(f"  - 기업 코드가 블라인드에서 다른 형태로 사용됨")
        print(f"  - 일시적인 네트워크 오류")
        print(f"  - 블라인드 서버 응답 지연")
    
    print(f"\n📁 생성된 파일들:")
    print(f"  - ./data/hybrid_vectordb/ 디렉토리에 각 기업별 벡터DB 파일 저장")
    print(f"  - 파일명 형식: {datetime.now().strftime('%Y%m%d')}_[기업명]_hybrid_vectordb.json")
    
    print(f"\n🎯 모든 기업의 하이브리드 청크가 생성되었습니다!")
    print(f"{'='*80}\n")


def run_top50_korean_companies_crawl(pages: int = 25, headless: bool = False):
    """한국 상위 50개 기업 크롤링 (시가총액 기준)"""
    
    top50_companies = [
        "삼성전자", "LG에너지솔루션", "SK하이닉스", "삼성바이오로직스", "NAVER",
        "LG화학", "현대차", "삼성SDI", "카카오", "기아",
        "POSCO홀딩스", "KB금융", "SK이노베이션", "셀트리온", "삼성물산",
        "신한지주", "현대모비스", "카카오뱅크", "SK", "LG전자",
        "한국전력", "S-Oil", "하나금융지주", "크래프톤", "삼성생명",
        "LG", "SK텔레콤", "KT&G", "카카오페이", "삼성전기",
        "삼성에스디에스", "우리금융지주", "고려아연", "LG생활건강", "포스코케이칼",
        "엔씨소프트", "KT", "삼성화재", "SK바이오사이언스", "하이브",
        "아모레퍼시픽", "기업은행", "SK아이이테크놀로지", "현대글로비스", "롯데케미칼",
        "넷마블", "한국조선해양", "SK바이오팜", "LG디스플레이", "한온시스템"
    ]
    
    print(f"\n🇰🇷 한국 상위 50개 기업 하이브리드 크롤링")
    print(f"시가총액 기준 선정된 주요 기업들을 순차 크롤링합니다")
    
    return run_multiple_companies_hybrid_crawl(
        company_list=top50_companies,
        pages=pages,
        headless=headless,
        delay_between_companies=30  # 30초 대기
    )


def main():
    """메인 실행 함수 - 선택지 추가"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           블라인드 크롤링 도구 v2.0 - 하이브리드 청크          ║
    ║                                                              ║
    ║  주요 특징:                                                   ║
    ║  • 제목, 장점, 단점을 독립 청크로 분리                        ║
    ║  • 카테고리별 장점/단점 청크 생성                             ║
    ║  • is_positive 메타데이터로 명확한 구분                       ║
    ║  • kss 기반 정확한 문장 분할                                 ║
    ║  • 벡터 DB 최적화된 독립 저장 구조                            ║
    ║                                                              ║
    ║  v2.0 신규 기능:                                             ║
    ║  • 다중 기업 순차 크롤링 지원                                ║
    ║  • 한국 상위 50개 기업 일괄 크롤링                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        print("\n크롤링 모드를 선택해주세요:")
        print("1. 단일 기업 크롤링")
        print("2. 여러 기업 크롤링 (직접 입력)")
        print("3. 한국 상위 50개 기업 일괄 크롤링")
        
        choice = input("\n선택 (1-3): ").strip()
        
        if choice == "1":
            # 기존 단일 기업 크롤링
            company = input("기업 코드 입력 (예: NAVER): ").strip()
            pages = int(input("크롤링할 페이지 수 (기본 25): ") or "25")
            headless = input("헤드리스 모드? (y/N): ").lower() in ['y', 'yes']
            
            run_single_company_hybrid_crawl(company, pages, headless)
            
        elif choice == "2":
            # 여러 기업 직접 입력
            print("\n기업 코드를 쉼표로 구분하여 입력하세요 (예: NAVER,삼성전자,카카오)")
            companies_input = input("기업 코드들: ").strip()
            company_list = [company.strip() for company in companies_input.split(',') if company.strip()]
            
            if not company_list:
                print("유효한 기업 코드가 입력되지 않았습니다.")
                return
            
            pages = int(input("크롤링할 페이지 수 (기본 25): ") or "25")
            headless = input("헤드리스 모드? (y/N): ").lower() in ['y', 'yes']
            delay = int(input("기업간 대기시간(초) (기본 30): ") or "30")
            
            run_multiple_companies_hybrid_crawl(company_list, pages, headless, delay)
            
        elif choice == "3":
            # 상위 50개 기업 일괄 크롤링
            pages = int(input("크롤링할 페이지 수 (기본 25): ") or "25")
            headless = input("헤드리스 모드? (y/N): ").lower() in ['y', 'yes']
            
            print(f"\n⚠️  주의사항:")
            print(f"• 50개 기업 크롤링은 상당한 시간이 소요됩니다 (예상: 3-5시간)")
            print(f"• 각 기업간 30초씩 대기하여 서버 부하를 방지합니다")
            print(f"• 중간에 Ctrl+C로 중단 가능하며, 그때까지의 결과는 저장됩니다")
            
            confirm = input("\n계속하시겠습니까? (y/N): ").lower()
            if confirm in ['y', 'yes']:
                run_top50_korean_companies_crawl(pages, headless)
            else:
                print("크롤링이 취소되었습니다.")
        
        else:
            print("잘못된 선택입니다.")
        
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"실행 중 오류 발생: {e}")


if __name__ == "__main__":
    main()