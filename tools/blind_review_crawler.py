# blind_review_crawler.py
"""
블라인드 크롤링 도구 v3.2 - 배치 처리 최적화 버전
- 크롤링과 AI 분류를 분리하여 효율성 향상
- 모든 청크를 수집한 후 대용량 배치로 AI 분류
- OpenAI API 호출 횟수 대폭 감소
"""

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
from dotenv import load_dotenv
from tqdm import tqdm

# 기존 모듈 import
from enhanced_category_processor import CategorySpecificProcessor, VectorDBOptimizer
from keyword_dictionary import korean_keywords

# Settings import (환경변수 설정 사용)
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from blindinsight.models.base import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

# 로깅 설정 (간소화)
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blind_crawler.log', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)

# 환경변수 로드
load_dotenv()

class BlindReviewCrawler:
    """개선된 블라인드 리뷰 크롤러 - 배치 처리 최적화"""
    
    def __init__(self, headless=False, wait_timeout=10, output_dir="./data/vectordb", 
                 use_ai_classification=True, openai_api_key=None, enable_spell_check=True):
        self.wait_timeout = wait_timeout
        self.driver = None
        self.wait = None
        self.output_dir = output_dir
        self.is_logged_in = False
        
        # 분류 방식 설정
        self.use_ai_classification = use_ai_classification
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.enable_spell_check = enable_spell_check
        
        # AI 분류를 원하지만 API 키가 없으면 키워드 분류로 자동 전환
        if self.use_ai_classification and not self.openai_api_key:
            print("⚠️ OpenAI API 키가 없어 키워드 분류로 전환합니다.")
            self.use_ai_classification = False
        
        # 프로세서 초기화
        self.category_processor = CategorySpecificProcessor(
            openai_api_key=self.openai_api_key if self.use_ai_classification else None,
            enable_spell_check=self.enable_spell_check if self.use_ai_classification else False
        )
        self.vectordb_optimizer = VectorDBOptimizer()
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 간소화된 처리 결과 추적
        self.results = {
            "reviews_processed": 0,
            "chunks_created": 0,
            "api_calls_saved": 0,
            "category_counts": {
                "career_growth": 0,
                "salary_benefits": 0,
                "work_life_balance": 0,
                "company_culture": 0,
                "management": 0
            }
        }
        
        # Chrome 드라이버 초기화
        self._initialize_driver(headless)
        
        # 분류 방식 출력
        classification_method = "AI 배치 분류" if self.use_ai_classification else "키워드 분류"
        print(f"🔍 분류 방식: {classification_method}")
    
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
            
        except Exception as e:
            print(f"❌ 드라이버 초기화 실패: {e}")
            raise
    
    def login_wait(self, url):
        """로그인 대기 및 처리"""
        try:
            print(f"🌐 페이지 접속 중: {url}")
            self.driver.get(url)
            time.sleep(5)
            
            # 로그인 버튼 클릭 시도
            try:
                login_btn = self.wait.until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "btn_signin"))
                )
                self.driver.execute_script("arguments[0].click();", login_btn)
            except Exception:
                pass
            
            time.sleep(3)
            
            print("\n" + "="*50)
            print("🔐 블라인드 로그인 필요")
            print("1. 블라인드 앱에서 로그인")
            print("2. '더보기' → '블라인드 웹 로그인' 클릭")
            print("3. 인증번호 입력")
            print("="*50)
            
            input("로그인 완료 후 Enter를 눌러주세요: ")
            self.is_logged_in = True
            print("✅ 로그인 완료")
            
        except Exception as e:
            print(f"❌ 로그인 과정 오류: {e}")
            raise
    
    def extract_review_data(self, element):
        """개별 리뷰에서 데이터 추출 - 직무/연도 정보 추가"""
        try:
            # 평점 정보 추출
            rating_element = element.find_element(By.CLASS_NAME, "rating")
            
            # 상세 평점 보기 클릭
            try:
                more_rating_btn = rating_element.find_element(By.CLASS_NAME, "more_rating")
                more_rating_btn.click()
                time.sleep(0.5)
            except NoSuchElementException:
                pass
            
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
            except Exception:
                pass
            
            # 제목 추출
            try:
                title_element = element.find_element(By.CSS_SELECTOR, "div.review_item_inr > h3.rvtit > a")
                title = self.clean_text(title_element.text)
            except NoSuchElementException:
                title = "제목 없음"
            
            # 직원 유형, 직무, 연도 추출 (auth 클래스에서)
            try:
                auth_element = element.find_element(By.CSS_SELECTOR, "div.review_item_inr > div.auth")
                auth_text = auth_element.text
                
                # 기본값 설정
                status = "정보 없음"
                position = "정보 없음"
                year = "정보 없음"
                
                # auth_text 형태 1: "현직원\n엔지니어링\n2020.03.10"
                # auth_text 형태 2: "현직원 · j********* · 하드웨어 엔지니어 - 2025.02.23"
                
                if '·' in auth_text:
                    # 형태 2: · 구분자로 파싱
                    auth_parts = [part.strip() for part in auth_text.split('·') if part.strip()]
                    
                    if len(auth_parts) >= 1:
                        # "Verified User\n현직원"에서 "현직원"만 추출
                        raw_status = auth_parts[0].strip()
                        if '\n' in raw_status:
                            status = raw_status.split('\n')[-1].strip()  # 마지막 부분만 가져오기
                        else:
                            status = raw_status
                    
                    if len(auth_parts) >= 3:
                        # 직무 정보는 마지막 부분에서 추출 (예: "생산엔지니어·생산관리 - 2024.02.06")
                        job_date_part = auth_parts[2].strip()
                        
                        # 날짜 부분 제거하고 직무만 추출
                        if '-' in job_date_part:
                            # 날짜 패턴을 찾아서 그 부분만 제거
                            date_match = re.search(r'\s*-\s*\d{4}\.\d{2}\.\d{2}', job_date_part)
                            if date_match:
                                position = job_date_part[:date_match.start()].strip()
                                # 연도 추출
                                year_match = re.search(r'(\d{4})', date_match.group())
                                if year_match:
                                    year = year_match.group(1)
                            else:
                                # 정규식으로 날짜 패턴 제거 (폴백 방식)
                                position = re.sub(r'\s*-\s*\d{4}\.?\d{0,2}\.?\d{0,2}.*', '', job_date_part).strip()
                                year_match = re.search(r'(\d{4})', job_date_part)
                                if year_match:
                                    year = year_match.group(1)
                        else:
                            position = job_date_part
                            
                else:
                    # 형태 1: 개행 구분자로 파싱
                    auth_lines = [line.strip() for line in auth_text.split('\n') if line.strip()]
                    
                    if len(auth_lines) >= 1:
                        # "Verified User"같은 부분을 제외하고 실제 직원 상태만 추출
                        raw_status = auth_lines[0]
                        if '현직원' in raw_status:
                            status = '현직원'
                        elif '전직원' in raw_status:
                            status = '전직원'
                        else:
                            status = raw_status
                    
                    if len(auth_lines) >= 2:
                        position = auth_lines[1]  # 직무
                    
                    if len(auth_lines) >= 3:
                        # 연도 추출 (2020.03.10 형태에서 2020만 추출)
                        date_text = auth_lines[2]
                        year_match = re.search(r'(\d{4})', date_text)
                        if year_match:
                            year = year_match.group(1)
                        
            except (NoSuchElementException, IndexError):
                status = "정보 없음"
                position = "정보 없음"
                year = "정보 없음"
            
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
                    
            except Exception:
                pass
            
            return [
                total_score,
                detail_scores[0],
                detail_scores[1], 
                detail_scores[2],
                detail_scores[3],
                detail_scores[4],
                title,
                status,
                position,      # 새로 추가: 직무
                year,          # 새로 추가: 연도
                pros,
                cons
            ]
            
        except Exception as e:
            return [0.0, "0", "0", "0", "0", "0", "오류", "오류", "오류", "오류", "추출 실패", "추출 실패"]
    
    def clean_text(self, text):
        """기본 텍스트 정제"""
        if not text:
            return ""
        
        # 한글 자모 제거
        pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'
        text = re.sub(pattern=pattern, repl='', string=text)
        
        # HTML 태그 제거
        pattern = '<[^>]*>'
        text = re.sub(pattern=pattern, repl='', string=text)
        
        # 기본 특수문자 정리
        pattern = '[^\w\s가-힣.,!?()-]'
        text = re.sub(pattern=pattern, repl=' ', string=text)
        
        # 줄바꿈을 공백으로
        text = text.replace('\r', '').replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def crawl_company_reviews(self, company_code: str, pages: int = 25):
        """회사 리뷰 크롤링 - 배치 처리 최적화"""
        
        print(f"\n🚀 {company_code} 크롤링 시작")
        
        base_url = f"https://www.teamblind.com/kr/company/{company_code}/reviews"
        
        if not self.is_logged_in:
            self.login_wait(base_url)
        
        # 1단계: 전체 리뷰 수집 (변경 없음)
        all_reviews = []
        
        print(f"📄 페이지 수집 중...")
        for page in range(1, pages + 1):
            try:
                target_url = f"{base_url}?page={page}"
                
                self.driver.get(target_url)
                time.sleep(3)
                
                review_elements = self.driver.find_elements(By.CLASS_NAME, "review_item")
                
                if not review_elements:
                    continue
                
                for idx, element in enumerate(review_elements):
                    try:
                        review_data = self.extract_review_data(element)
                        all_reviews.append(review_data)
                    except Exception:
                        continue
                
                time.sleep(2)
                
            except Exception:
                continue
        
        if not all_reviews:
            print("❌ 추출된 리뷰가 없습니다.")
            return False
        
        print(f"✅ 총 {len(all_reviews)}개 리뷰 수집 완료")
        
        # 2단계: 개선된 배치 처리
        return self._process_reviews_with_batch_optimization(company_code, all_reviews)
    
    def _process_reviews_with_batch_optimization(self, company_code: str, all_reviews: List) -> bool:
        """개선된 배치 처리 방식 - 직무/연도 필드 추가"""
        
        print(f"\n📊 리뷰 전처리 중...")
        
        # 1단계: 모든 리뷰에서 청크 생성 (분류 없이)
        all_chunk_data = []  # 분류 전 청크 정보 저장
        all_chunk_contents = []  # AI에 보낼 텍스트만 저장
        
        for idx, raw_review in enumerate(tqdm(all_reviews, desc="청크 생성", unit="리뷰")):
            try:
                # 리뷰 데이터 구조화 (필드 추가)
                review_data = {
                    "회사": company_code,
                    "총점": raw_review[0],
                    "커리어향상": raw_review[1], 
                    "워라밸": raw_review[2],
                    "급여복지": raw_review[3],
                    "사내문화": raw_review[4],
                    "경영진": raw_review[5],
                    "제목": raw_review[6],
                    "직원유형_원본": raw_review[7],
                    "직무": raw_review[8],        # 새로 추가
                    "연도": raw_review[9],        # 새로 추가
                    "직원유형": raw_review[7],  # 현직원/전직원 상태 추가
                    "장점": raw_review[10],       # 인덱스 조정
                    "단점": raw_review[11],       # 인덱스 조정
                    "id": f"review_{company_code}_{idx:04d}"
                }
                
                # 분류 없이 청크만 생성
                chunk_groups = self.category_processor.create_chunks_without_classification(review_data)
                
                # 청크 데이터와 내용 분리 저장
                for chunk_type, chunks_info in chunk_groups.items():
                    for chunk_info in chunks_info:
                        all_chunk_data.append(chunk_info)
                        all_chunk_contents.append(chunk_info['content'])
                
            except Exception:
                continue
        
        if not all_chunk_contents:
            print("❌ 생성된 청크가 없습니다.")
            return False
        
        print(f"📋 총 {len(all_chunk_contents)}개 청크 생성 완료")
        
        # 나머지 처리는 기존과 동일...
        # (분류, 매핑, 저장 로직은 변경 없음)
        
        # 2단계: 대용량 배치 분류 (AI 사용시에만)
        classification_results = []
        
        if self.use_ai_classification and self.category_processor.text_processor:
            try:
                print(f"🧠 AI 배치 분류 시작...")
                
                # 예상 API 호출 횟수 계산
                if SETTINGS_AVAILABLE:
                    batch_size = settings.ai_batch_size
                else:
                    batch_size = int(os.getenv("AI_BATCH_SIZE", "30"))
                    
                expected_api_calls = (len(all_chunk_contents) + batch_size - 1) // batch_size
                individual_calls_saved = len(all_reviews) - expected_api_calls
                
                print(f"   - 청크 수: {len(all_chunk_contents)}개")
                print(f"   - 예상 API 호출: {expected_api_calls}회")
                print(f"   - 절약된 API 호출: {individual_calls_saved}회")
                
                # 배치 분류 실행
                classification_results = self.category_processor.text_processor.process_chunks_batch(
                    all_chunk_contents, batch_size=batch_size
                )
                
                self.results["api_calls_saved"] = individual_calls_saved
                
            except Exception as e:
                print(f"⚠️ AI 분류 실패, 키워드 분류로 폴백: {e}")
                classification_results = [
                    self.category_processor._classify_with_keywords_fallback(content)
                    for content in all_chunk_contents
                ]
        else:
            # 키워드 분류
            print(f"🔤 키워드 분류 중...")
            classification_results = []
            for content in tqdm(all_chunk_contents, desc="키워드 분류", unit="청크"):
                result = self.category_processor._classify_with_keywords_fallback(content)
                classification_results.append(result)
        
        # 3단계: 분류 결과를 청크 데이터에 매핑
        print(f"🔗 결과 매핑 중...")
        final_chunks = self._map_classification_results_to_chunks(
            all_chunk_data, classification_results
        )
        
        # 4단계: 벡터 DB 최적화 및 저장
        print(f"💾 파일 저장 중...")
        optimized_chunks = self.vectordb_optimizer.optimize_chunks_for_vectordb(final_chunks)
        
        file_path = self.vectordb_optimizer.save_vectordb_file(
            company=company_code, 
            optimized_chunks=optimized_chunks, 
            output_dir=self.output_dir,
            category_processor=self.category_processor
        )
        
        # 통계 업데이트
        self.results["reviews_processed"] = len(all_reviews)
        self.results["chunks_created"] = len(final_chunks)
        
        # 카테고리별 통계
        for chunk in final_chunks:
            category = chunk.get('category', 'career_growth')
            if category in self.results["category_counts"]:
                self.results["category_counts"][category] += 1
        
        # 결과 출력
        self._print_batch_optimization_summary(company_code, file_path)
        
        return True
    
    def _map_classification_results_to_chunks(self, chunk_data_list, classification_results):
        """분류 결과를 청크 데이터에 매핑"""
        
        final_chunks = []
        
        for i, (chunk_data, classification) in enumerate(zip(chunk_data_list, classification_results)):
            # 1순위 카테고리로 청크 생성
            primary_chunk = self.category_processor.create_final_chunk(
                chunk_data, classification, "primary"
            )
            final_chunks.append(primary_chunk)
            
            # 2순위 카테고리 청크 생성 (신뢰도 조건 만족시)
            #if (classification.get("secondary_category") and 
             #   classification.get("secondary_confidence", 0) >= 0.3 and
              #  classification["secondary_category"] != classification["primary_category"]):
                
               # secondary_chunk = self.category_processor.create_final_chunk(
                #    chunk_data, classification, "secondary"
                #)
                #final_chunks.append(secondary_chunk)
        
        return final_chunks
    
    def _print_batch_optimization_summary(self, company_code: str, file_path: str):
        """배치 최적화 결과 출력"""
        
        print(f"\n{'='*50}")
        print(f"🎉 {company_code} 크롤링 완료 (배치 최적화)")
        print(f"{'='*50}")
        
        # 기본 통계
        print(f"📊 처리 결과:")
        print(f"  - 처리된 리뷰: {self.results['reviews_processed']}개")
        print(f"  - 생성된 청크: {self.results['chunks_created']}개")
        
        # 배치 최적화 효과
        if self.use_ai_classification and self.results['api_calls_saved'] > 0:
            print(f"  - 절약된 API 호출: {self.results['api_calls_saved']}회")
            efficiency_improvement = (self.results['api_calls_saved'] / self.results['reviews_processed']) * 100
            print(f"  - 효율성 개선: {efficiency_improvement:.1f}%")
        
        # 분류 방식 표시
        classification_method = "AI 대용량 배치 분류" if self.use_ai_classification else "키워드 기반 분류"
        print(f"  - 분류 방식: {classification_method}")
        
        # 카테고리별 청크 수
        category_names_kr = {
            "career_growth": "커리어 향상",
            "salary_benefits": "급여 및 복지", 
            "work_life_balance": "업무와 삶의 균형",
            "company_culture": "사내 문화",
            "management": "경영진"
        }
        
        print(f"\n📈 카테고리별 청크 수:")
        for category, count in self.results["category_counts"].items():
            category_kr = category_names_kr.get(category, category)
            print(f"  - {category_kr}: {count}개")
        
        # 파일 정보
        print(f"\n📁 저장된 파일:")
        print(f"  - {os.path.basename(file_path)}")
        
        print(f"\n✅ 배치 최적화 완료!")
        print(f"{'='*50}\n")
    
    def close(self):
        """드라이버 종료"""
        if self.driver:
            self.driver.quit()


def run_single_company_crawl(company_code: str, pages: int = 25, headless: bool = False, 
                            use_ai_classification: bool = True, openai_api_key: str = None, 
                            enable_spell_check: bool = True):
    """단일 기업 크롤링 실행"""
    
    print(f"\n블라인드 크롤러 v3.2 - 배치 최적화")
    print(f"🎯 대상 기업: {company_code}")
    print(f"📄 크롤링 페이지: {pages}")
    
    crawler = BlindReviewCrawler(
        headless=headless,
        output_dir="./data/vectordb",
        use_ai_classification=use_ai_classification,
        openai_api_key=openai_api_key,
        enable_spell_check=enable_spell_check
    )
    
    try:
        success = crawler.crawl_company_reviews(company_code, pages)
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        return False
    except Exception as e:
        print(f"❌ 크롤링 실행 중 오류: {e}")
        return False
    finally:
        crawler.close()


def run_multiple_companies_crawl(company_list: List[str], pages: int = 25, 
                                headless: bool = False, delay_between_companies: int = 30,
                                use_ai_classification: bool = True, openai_api_key: str = None, 
                                enable_spell_check: bool = True):
    """여러 기업 순차 크롤링 실행"""
    
    print(f"\n블라인드 다중 기업 크롤러 v3.2 - 배치 최적화")
    print(f"🎯 대상 기업: {len(company_list)}개")
    print(f"📄 크롤링 페이지: {pages}")
    
    # 크롤링 결과 추적
    results = {
        "success": [],
        "failed": [],
        "total_companies": len(company_list),
        "total_api_calls_saved": 0
    }
    
    # 단일 크롤러 인스턴스 생성
    crawler = BlindReviewCrawler(
        headless=headless,
        output_dir="./data/vectordb",
        use_ai_classification=use_ai_classification,
        openai_api_key=openai_api_key,
        enable_spell_check=enable_spell_check
    )
    
    try:
        print(f"\n{'='*50}")
        print(f"🚀 배치 최적화 다중 기업 크롤링 시작")
        print(f"{'='*50}")
        
        for idx, company_code in enumerate(company_list):
            try:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{idx+1}/{len(company_list)}] {current_time} - {company_code}")
                
                # 크롤링 실행
                success = crawler.crawl_company_reviews(company_code, pages)
                
                if success:
                    results["success"].append(company_code)
                    results["total_api_calls_saved"] += crawler.results.get("api_calls_saved", 0)
                else:
                    results["failed"].append(company_code)
                
                # 대기 (마지막 기업이 아닌 경우)
                if idx < len(company_list) - 1:
                    print(f"⏱️ {delay_between_companies}초 대기 중...")
                    time.sleep(delay_between_companies)
                    
            except Exception as e:
                print(f"❌ {company_code} 크롤링 실패: {e}")
                results["failed"].append(company_code)
                
                if idx < len(company_list) - 1:
                    time.sleep(delay_between_companies)
        
        # 전체 결과 요약
        _print_multiple_crawling_summary_optimized(results)
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        _print_multiple_crawling_summary_optimized(results)
        return results
    except Exception as e:
        print(f"❌ 다중 크롤링 실행 중 오류: {e}")
        return results
    finally:
        crawler.close()


def _print_multiple_crawling_summary_optimized(results: Dict):
    """배치 최적화된 다중 크롤링 결과 요약 출력"""
    
    print(f"\n{'='*50}")
    print(f"🏁 배치 최적화 다중 기업 크롤링 완료")
    print(f"{'='*50}")
    
    print(f"📊 전체 결과:")
    print(f"  - 총 대상 기업: {results['total_companies']}개")
    print(f"  - 성공: {len(results['success'])}개")
    print(f"  - 실패: {len(results['failed'])}개")
    print(f"  - 성공률: {len(results['success'])/results['total_companies']*100:.1f}%")
    
    # 배치 최적화 효과
    if results.get("total_api_calls_saved", 0) > 0:
        print(f"  - 총 절약된 API 호출: {results['total_api_calls_saved']}회")
        print(f"  - 예상 비용 절약: ${results['total_api_calls_saved'] * 0.002:.2f}")
    
    if results["success"]:
        print(f"\n✅ 성공한 기업:")
        for i, company in enumerate(results["success"]):
            print(f"  {i+1:2d}. {company}")
    
    if results["failed"]:
        print(f"\n❌ 실패한 기업:")
        for i, company in enumerate(results["failed"]):
            print(f"  {i+1:2d}. {company}")
    
    print(f"\n📁 생성된 파일들:")
    print(f"  - ./data/vectordb/ 디렉토리에 각 기업별 파일 저장")
    print(f"{'='*50}\n")


def main():
    """메인 실행 함수"""
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                    블라인드 크롤링 도구 v3.2                     ║
    ║                     배치 처리 최적화 버전                        ║
    ║                                                                ║
    ║  • 키워드 기반 분류 (기본)                                       ║
    ║  • AI 기반 대용량 배치 분류 (OpenAI API 키 필요)                 ║  
    ║  • API 호출 횟수 최대 90% 절약                                   ║
    ║  • 간소화된 출력과 진행률 표시                                    ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        print("크롤링 모드를 선택해주세요:")
        print("1. 단일 기업 크롤링")
        print("2. 여러 기업 크롤링 (직접 입력)")
        print("3. 한국 상위 50개 기업 일괄 크롤링")
        
        choice = input("\n선택 (1-3): ").strip()
        
        # 분류 방식 선택
        print("\n분류 방식을 선택해주세요:")
        print("1. 키워드 분류 (빠름, API 비용 없음)")
        print("2. AI 배치 분류 (정확함, OpenAI API 키 필요, 비용 최적화)")
        
        classification_choice = input("선택 (1-2, 기본값: 1): ").strip()
        use_ai_classification = classification_choice == "2"
        
        # API 키 설정 (AI 분류 선택시)
        openai_api_key = None
        enable_spell_check = False
        
        if use_ai_classification:
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                print("\n⚠️ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
                api_input = input("API 키를 입력하거나 Enter로 키워드 분류 사용: ").strip()
                if api_input:
                    openai_api_key = api_input
                else:
                    use_ai_classification = False
            
            if use_ai_classification:
                enable_spell_check = input("맞춤법 검사 사용? (y/N): ").lower() in ['y', 'yes']
                print("\n💡 배치 최적화 효과:")
                print("  - 개별 리뷰 처리 대비 API 호출 90% 절약")
                print("  - 대용량 배치로 처리 속도 향상")
                print("  - 비용 효율적인 AI 분류")
        
        if choice == "1":
            # 단일 기업 크롤링
            company = input("\n기업 코드 입력 (예: NAVER): ").strip()
            pages = int(input("크롤링할 페이지 수 (기본 25): ") or "25")
            headless = input("헤드리스 모드? (y/N): ").lower() in ['y', 'yes']
            
            run_single_company_crawl(
                company_code=company, 
                pages=pages, 
                headless=headless,
                use_ai_classification=use_ai_classification,
                openai_api_key=openai_api_key,
                enable_spell_check=enable_spell_check
            )
            
        elif choice == "2":
            # 여러 기업 직접 입력
            print("\n기업 코드를 쉼표로 구분하여 입력하세요 (예: NAVER,삼성전자,카카오)")
            companies_input = input("기업 코드들: ").strip()
            company_list = [company.strip() for company in companies_input.split(',') if company.strip()]
            
            if not company_list:
                print("❌ 유효한 기업 코드가 입력되지 않았습니다.")
                return
            
            pages = int(input("크롤링할 페이지 수 (기본 25): ") or "25")
            headless = input("헤드리스 모드? (y/N): ").lower() in ['y', 'yes']
            delay = int(input("기업간 대기시간(초) (기본 30): ") or "30")
            
            run_multiple_companies_crawl(
                company_list=company_list, 
                pages=pages, 
                headless=headless,
                delay_between_companies=delay,
                use_ai_classification=use_ai_classification,
                openai_api_key=openai_api_key,
                enable_spell_check=enable_spell_check
            )
            
        elif choice == "3":
            # 상위 50개 기업 일괄 크롤링
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
            
            pages = int(input("\n크롤링할 페이지 수 (기본 25): ") or "25")
            headless = input("헤드리스 모드? (y/N): ").lower() in ['y', 'yes']
            
            print(f"\n⚠️ 주의사항:")
            print(f"• 50개 기업 크롤링은 상당한 시간이 소요됩니다 (예상: 3-5시간)")
            if use_ai_classification:
                estimated_cost = len(top50_companies) * pages * 0.1  # 배치 최적화로 대폭 절약
                print(f"• 배치 최적화로 OpenAI API 비용 절약 (예상: ${estimated_cost:.2f}, 기존 대비 90% 절약)")
            print(f"• 각 기업간 30초씩 대기하여 서버 부하를 방지합니다")
            print(f"• 중간에 Ctrl+C로 중단 가능하며, 그때까지의 결과는 저장됩니다")
            
            confirm = input("\n계속하시겠습니까? (y/N): ").lower()
            if confirm in ['y', 'yes']:
                run_multiple_companies_crawl(
                    company_list=top50_companies,
                    pages=pages, 
                    headless=headless,
                    delay_between_companies=30,
                    use_ai_classification=use_ai_classification,
                    openai_api_key=openai_api_key,
                    enable_spell_check=enable_spell_check
                )
            else:
                print("❌ 크롤링이 취소되었습니다.")
        
        else:
            print("❌ 잘못된 선택입니다.")
        
    except KeyboardInterrupt:
        print("\n⚠️ 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    main()