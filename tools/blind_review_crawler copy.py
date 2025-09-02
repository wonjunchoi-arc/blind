# integrated_blind_crawler.py
"""
통합 블라인드 크롤링 도구

주요 기능:
- 블라인드 웹사이트에서 기업 리뷰 데이터 수집
- 배치 크롤링: 여러 기업을 한 번 로그인으로 순차 처리
- 회사 통계 데이터 추출 (홈페이지 표시용)
- RAG 시스템을 위한 최적화된 JSON 구조로 저장
- 청킹에 적합한 데이터 전처리 및 메타데이터 생성
- 키워드 추출 및 감성 분석 기능

사용법:
1. 단일 기업: crawler = BlindReviewCrawler(); crawler.crawl_single_company("NAVER", 25)
2. 배치 크롤링: crawler.crawl_multiple_companies(["NAVER", "KAKAO", "SAMSUNG"], 25)
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
from typing import List, Dict, Any

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blind_crawler.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGDataProcessor:
    """RAG 최적화를 위한 데이터 처리기"""
    
    def __init__(self):
        # 키워드 카테고리 사전
        self.keyword_categories = {
            "worklife": ["야근", "잔업", "오버타임", "워라밸", "퇴근시간", "휴가", "눈치", "밤샘"],
            "salary": ["연봉", "급여", "월급", "기본급", "상여금", "보너스", "인센티브"],
            "culture": ["분위기", "문화", "동료", "상사", "갑질", "회식", "조직문화"],
            "growth": ["성장", "승진", "교육", "경력", "기회", "발전", "커리어"],
            "benefits": ["복지", "보험", "스톡옵션", "주식", "식대", "교통비"],
            "management": ["경영진", "임원", "대표", "리더십", "비전"]
        }
        
        # 감성 키워드
        self.positive_keywords = ["좋", "만족", "훌륭", "최고", "추천", "성장", "기회"]
        self.negative_keywords = ["힘들", "아쉬운", "문제", "스트레스", "갑질", "눈치"]
        
        # 인텔리전스 패턴
        self.intelligence_patterns = {
            "overtime_indicators": ["야근", "잔업", "늦게까지", "밤늦게"],
            "toxic_culture_indicators": ["갑질", "눈치", "강압적", "일방적"],
            "salary_issues_indicators": ["연봉삭감", "급여문제", "인센티브없음"],
            "work_pressure_indicators": ["압박", "스트레스", "힘들", "무리한"]
        }
    
    def parse_employee_info(self, employee_raw_text: str) -> Dict[str, str]:
        """직원 정보를 파싱하여 재직 상태, 직무, 작성날짜로 구분"""
        import re
        
        parsed_info = {
            "employment_status": "정보 없음",
            "position": "정보 없음", 
            "review_date": "정보 없음",
            "raw_text": employee_raw_text
        }
        
        if not employee_raw_text:
            return parsed_info
        
        try:
            # 날짜 패턴 추출 (YYYY.MM.DD 형태)
            date_pattern = r'(\d{4}\.\d{2}\.\d{2})'
            date_match = re.search(date_pattern, employee_raw_text)
            if date_match:
                parsed_info["review_date"] = date_match.group(1)
        except:
            pass
        
        try:
            # " - " 기준으로 날짜 부분 제거
            text_without_date = re.sub(r'\s*-\s*\d{4}\.\d{2}\.\d{2}.*$', '', employee_raw_text)
            
            # " · " 기준으로 분할
            parts = [part.strip() for part in text_without_date.split('·')]
            
            if len(parts) >= 1:
                employment_status = parts[0].strip()
                if employment_status:
                    parsed_info["employment_status"] = employment_status
            
            if len(parts) >= 3:
                position = parts[2].strip()
                if position:
                    parsed_info["position"] = position
            elif len(parts) == 2:
                second_part = parts[1].strip()
                if not re.match(r'^[A-Z]\*+$', second_part):
                    parsed_info["position"] = second_part
                    
        except Exception as e:
            logger.warning(f"직원 정보 파싱 실패: {e}")
        
        return parsed_info
    
    def extract_keywords_with_context(self, text: str) -> List[Dict[str, Any]]:
        """키워드 추출 및 맥락 포함"""
        if not text:
            return []
        
        found_keywords = []
        
        for category, keywords in self.keyword_categories.items():
            for keyword in keywords:
                if keyword in text:
                    context = self._extract_context(text, keyword, 15)
                    found_keywords.append({
                        "keyword": keyword,
                        "category": category,
                        "context": context,
                        "frequency": text.count(keyword)
                    })
        
        return found_keywords
    
    def _extract_context(self, text: str, keyword: str, context_length: int = 15) -> str:
        """키워드 주변 맥락 추출"""
        try:
            index = text.find(keyword)
            if index == -1:
                return ""
            
            start = max(0, index - context_length)
            end = min(len(text), index + len(keyword) + context_length)
            return text[start:end]
        except:
            return ""
    
    def create_semantic_content(self, review_data: Dict) -> str:
        """벡터 검색용 의미적 콘텐츠 생성"""
        rating_text = self._rating_to_text(review_data["총점"])
        employee_info = self.parse_employee_info(review_data.get("직원유형_원본", ""))
        
        semantic_content = f"""
{review_data["제목"]}

{review_data["회사"]}에 대한 {employee_info["employment_status"]} 평가입니다.
직무는 {employee_info["position"]}이며, {employee_info["review_date"]}에 작성된 리뷰입니다.
총점 {review_data["총점"]}점으로 {rating_text} 수준의 만족도를 보입니다.

평가 세부사항:
- 커리어 향상: {review_data["커리어향상"]}점
- 업무와 삶의 균형: {review_data["워라밸"]}점  
- 급여 및 복지: {review_data["급여복지"]}점
- 사내 문화: {review_data["사내문화"]}점
- 경영진: {review_data["경영진"]}점

주요 장점: {review_data["장점"]}

주요 단점: {review_data["단점"]}

이 리뷰는 {employee_info["employment_status"]}이 {employee_info["position"]} 직무의 관점에서 회사의 전반적인 근무 환경과 조직 문화를 평가한 내용입니다.
        """.strip()
        
        return semantic_content
    
    def create_keyword_content(self, review_data: Dict) -> str:
        """키워드 검색용 구조화된 콘텐츠 생성"""
        employee_info = self.parse_employee_info(review_data.get("직원유형_원본", ""))
        
        keyword_content = f"""
회사: {review_data["회사"]}
재직상태: {employee_info["employment_status"]}
직무: {employee_info["position"]}
작성날짜: {employee_info["review_date"]}
총점: {review_data["총점"]}점
커리어향상: {review_data["커리어향상"]}점
워라밸: {review_data["워라밸"]}점
급여복지: {review_data["급여복지"]}점
사내문화: {review_data["사내문화"]}점
경영진: {review_data["경영진"]}점

제목: {review_data["제목"]}
장점: {review_data["장점"]}
단점: {review_data["단점"]}
        """.strip()
        
        return keyword_content
    
    def _rating_to_text(self, score: float) -> str:
        """평점을 자연어로 변환"""
        if score >= 4.5:
            return "매우 높은"
        elif score >= 4.0:
            return "높은"
        elif score >= 3.5:
            return "보통 이상의"
        elif score >= 3.0:
            return "보통"
        else:
            return "아쉬운"
    
    def analyze_sentiment(self, pros_text: str, cons_text: str) -> Dict[str, Any]:
        """감성 분석"""
        positive_count = sum(1 for word in self.positive_keywords if word in pros_text)
        negative_count = sum(1 for word in self.negative_keywords if word in cons_text)
        
        total_words = len((pros_text + cons_text).split())
        if total_words == 0:
            return {"score": 0.0, "positive_signals": 0, "negative_signals": 0, "overall": "neutral"}
        
        sentiment_score = (positive_count - negative_count) / total_words
        
        if sentiment_score > 0.1:
            overall = "positive"
        elif sentiment_score < -0.1:
            overall = "negative"
        else:
            overall = "neutral"
        
        return {
            "score": round(sentiment_score, 3),
            "positive_signals": positive_count,
            "negative_signals": negative_count,
            "overall": overall
        }
    
    def create_search_metadata(self, review_data: Dict, keywords: List[Dict]) -> Dict[str, Any]:
        """검색 메타데이터 생성"""
        employee_info = self.parse_employee_info(review_data.get("직원유형_원본", ""))
        
        category_flags = {}
        for category in self.keyword_categories.keys():
            category_flags[f"has_{category}_mention"] = any(
                kw["category"] == category for kw in keywords
            )
        
        metadata = {
            "company": review_data["회사"],
            "employment_status": employee_info["employment_status"],
            "position": employee_info["position"],
            "review_date": employee_info["review_date"],
            "rating_level": self._categorize_rating(review_data["총점"]),
            "rating_numeric": review_data["총점"],
            **category_flags,
            "total_keywords_count": len(keywords),
            "review_length": len(f"{review_data['제목']} {review_data['장점']} {review_data['단점']}")
        }
        
        return metadata
    
    def _categorize_rating(self, score: float) -> str:
        """평점 카테고리화"""
        if score >= 4.0:
            return "high"
        elif score >= 3.0:
            return "medium"
        else:
            return "low"
    
    def create_intelligence_flags(self, review_data: Dict) -> Dict[str, bool]:
        """인텔리전스 플래그 생성"""
        full_text = f"{review_data['제목']} {review_data['장점']} {review_data['단점']}"
        
        flags = {}
        
        for flag_name, indicators in self.intelligence_patterns.items():
            flags[flag_name.replace("_indicators", "")] = any(
                indicator in full_text for indicator in indicators
            )
        
        flags["has_negative_worklife"] = (
            flags.get("overtime", False) and 
            any(neg in full_text for neg in ["힘들", "스트레스", "문제"])
        )
        
        flags["has_positive_growth"] = (
            any(growth in full_text for growth in ["성장", "기회", "발전"]) and
            any(pos in full_text for pos in ["좋", "만족", "추천"])
        )
        
        return flags
    
    def create_rag_optimized_data(self, raw_review: List, company_name: str, review_index: int) -> Dict[str, Any]:
        """RAG 최적화 데이터 생성"""
        
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
            "단점": raw_review[9]
        }
        
        employee_info = self.parse_employee_info(raw_review[7])
        
        keywords = self.extract_keywords_with_context(
            f"{review_data['제목']} {review_data['장점']} {review_data['단점']}"
        )
        
        semantic_content = self.create_semantic_content(review_data)
        keyword_content = self.create_keyword_content(review_data)
        search_metadata = self.create_search_metadata(review_data, keywords)
        intelligence_flags = self.create_intelligence_flags(review_data)
        sentiment = self.analyze_sentiment(
            review_data['장점'] or "", 
            review_data['단점'] or ""
        )
        
        return {
            "id": f"review_{company_name}_{review_index:04d}",
            "raw_data": {
                "총점": raw_review[0],
                "커리어향상": raw_review[1],
                "워라밸": raw_review[2],
                "급여복지": raw_review[3],
                "사내문화": raw_review[4],
                "경영진": raw_review[5],
                "제목": raw_review[6],
                "직원유형_원본": raw_review[7],
                "재직상태": employee_info["employment_status"],
                "직무": employee_info["position"],
                "작성날짜": employee_info["review_date"],
                "장점": raw_review[8],
                "단점": raw_review[9]
            },
            "processed_data": {
                "content_semantic": semantic_content,
                "content_keyword": keyword_content,
                "extracted_keywords": keywords,
                "sentiment_analysis": sentiment,
                "rating_category": self._categorize_rating(raw_review[0])
            },
            "search_metadata": search_metadata,
            "intelligence_flags": intelligence_flags
        }


class CompanyStatisticsExtractor:
    """회사별 통계 데이터 추출 및 저장 클래스"""
    
    def __init__(self):
        self.company_stats = {}
    
    def extract_company_overview_stats(self, driver, company_name: str) -> Dict[str, Any]:
        """회사 개요 페이지에서 총괄 평점 정보 추출"""
        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import NoSuchElementException
        
        stats = {
            "company_name": company_name,
            "extraction_date": datetime.now().isoformat(),
            "overall_rating": 0.0,
            "total_reviews": 0,
            "category_ratings": {
                "career_growth": 0.0,
                "work_life_balance": 0.0,
                "salary_benefits": 0.0,
                "company_culture": 0.0,
                "management": 0.0
            },
            "rating_distribution": {
                "5_star": 0,
                "4_star": 0, 
                "3_star": 0,
                "2_star": 0,
                "1_star": 0
            },
            "percentage_distribution": {
                "5_star": 0.0,
                "4_star": 0.0,
                "3_star": 0.0,
                "2_star": 0.0,
                "1_star": 0.0
            }
        }
        
        try:
            # 전체 평점 추출
            overall_element = driver.find_element(By.CSS_SELECTOR, "strong.rate span.blind")
            if overall_element:
                stats["overall_rating"] = float(overall_element.text)
        except (NoSuchElementException, ValueError) as e:
            logger.warning(f"전체 평점 추출 실패: {e}")
        
        try:
            # 총 리뷰 수 추출 
            count_element = driver.find_element(By.CSS_SELECTOR, "span.count")
            if count_element:
                count_text = count_element.text.replace(',', '').replace('개', '').replace('리뷰', '').strip()
                stats["total_reviews"] = int(count_text)
        except (NoSuchElementException, ValueError) as e:
            logger.warning(f"리뷰 수 추출 실패: {e}")
        
        try:
            # 카테고리별 평점 추출
            category_elements = driver.find_elements(By.CSS_SELECTOR, "ul li i.blind")
            
            if len(category_elements) >= 5:
                stats["category_ratings"]["career_growth"] = float(category_elements[0].text)
                stats["category_ratings"]["work_life_balance"] = float(category_elements[1].text) 
                stats["category_ratings"]["salary_benefits"] = float(category_elements[2].text)
                stats["category_ratings"]["company_culture"] = float(category_elements[3].text)
                stats["category_ratings"]["management"] = float(category_elements[4].text)
        except (IndexError, ValueError) as e:
            logger.warning(f"카테고리별 평점 추출 실패: {e}")
        
        try:
            # 평점 분포도 추출
            distribution_elements = driver.find_elements(By.CSS_SELECTOR, "div.rating_bar")
            
            for i, element in enumerate(distribution_elements):
                try:
                    bar_text = element.text.strip()
                    if bar_text and bar_text.isdigit():
                        star_key = f"{5-i}_star"
                        if star_key in stats["rating_distribution"]:
                            stats["rating_distribution"][star_key] = int(bar_text)
                except:
                    continue
        except Exception as e:
            logger.warning(f"평점 분포 추출 실패: {e}")
        
        # 백분율 계산
        if stats["total_reviews"] > 0:
            for star_level, count in stats["rating_distribution"].items():
                percentage = (count / stats["total_reviews"]) * 100
                stats["percentage_distribution"][star_level] = round(percentage, 1)
        
        return stats
    
    def calculate_review_based_stats(self, reviews_data: List, company_name: str) -> Dict[str, Any]:
        """수집된 리뷰 데이터를 기반으로 통계 계산"""
        if not reviews_data:
            return self._empty_stats(company_name)
        
        total_reviews = len(reviews_data)
        overall_ratings = [review[0] for review in reviews_data if isinstance(review[0], (int, float))]
        
        if not overall_ratings:
            return self._empty_stats(company_name)
        
        avg_overall = sum(overall_ratings) / len(overall_ratings)
        
        # 카테고리별 평균 계산
        category_averages = {}
        category_names = ["career_growth", "work_life_balance", "salary_benefits", "company_culture", "management"]
        
        for i, category in enumerate(category_names, 1):
            try:
                category_scores = []
                for review in reviews_data:
                    if len(review) > i and isinstance(review[i], (int, float, str)):
                        if isinstance(review[i], str):
                            score = float(review[i]) if review[i].replace('.', '').isdigit() else 0
                        else:
                            score = float(review[i])
                        category_scores.append(score)
                
                if category_scores:
                    category_averages[category] = round(sum(category_scores) / len(category_scores), 1)
                else:
                    category_averages[category] = 0.0
            except:
                category_averages[category] = 0.0
        
        # 평점 분포 계산
        distribution = {"5_star": 0, "4_star": 0, "3_star": 0, "2_star": 0, "1_star": 0}
        for rating in overall_ratings:
            if rating >= 4.5:
                distribution["5_star"] += 1
            elif rating >= 3.5:
                distribution["4_star"] += 1
            elif rating >= 2.5:
                distribution["3_star"] += 1
            elif rating >= 1.5:
                distribution["2_star"] += 1
            else:
                distribution["1_star"] += 1
        
        # 백분율 계산
        percentages = {}
        for star_level, count in distribution.items():
            percentages[star_level] = round((count / total_reviews) * 100, 1)
        
        return {
            "company_name": company_name,
            "extraction_date": datetime.now().isoformat(),
            "data_source": "crawled_reviews",
            "overall_rating": round(avg_overall, 1),
            "total_reviews": total_reviews,
            "category_ratings": category_averages,
            "rating_distribution": distribution,
            "percentage_distribution": percentages
        }
    
    def _empty_stats(self, company_name: str) -> Dict[str, Any]:
        """빈 통계 데이터 반환"""
        return {
            "company_name": company_name,
            "extraction_date": datetime.now().isoformat(),
            "overall_rating": 0.0,
            "total_reviews": 0,
            "category_ratings": {
                "career_growth": 0.0,
                "work_life_balance": 0.0,
                "salary_benefits": 0.0,
                "company_culture": 0.0,
                "management": 0.0
            },
            "rating_distribution": {"5_star": 0, "4_star": 0, "3_star": 0, "2_star": 0, "1_star": 0},
            "percentage_distribution": {"5_star": 0.0, "4_star": 0.0, "3_star": 0.0, "2_star": 0.0, "1_star": 0.0}
        }
    
    def save_company_stats(self, stats_data: Dict, company_name: str, output_dir: str = "."):
        """회사 통계 데이터를 JSON 파일로 저장"""
        filename = f"{company_name}_company_stats.json"
        filepath = os.path.join(output_dir, filename)
        
        enhanced_stats = {
            **stats_data,
            "file_info": {
                "created_at": datetime.now().isoformat(),
                "file_version": "1.0",
                "purpose": "homepage_display",
                "data_type": "company_overview_statistics"
            }
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(enhanced_stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"회사 통계 저장 완료: {filepath}")
        return filepath


class BlindReviewCrawler:
    """통합 블라인드 기업 리뷰 크롤링 클래스 (배치 처리 + 회사 통계 포함)"""
    
    def __init__(self, headless=False, wait_timeout=10, save_stats=True, output_dir="./data/reviews"):
        self.wait_timeout = wait_timeout
        self.driver = None
        self.wait = None
        self.rag_processor = RAGDataProcessor()
        self.stats_extractor = CompanyStatisticsExtractor()
        self.save_stats = save_stats
        self.output_dir = output_dir
        self.is_logged_in = False
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 배치 크롤링 결과 추적
        self.results_summary = []
        self.failed_companies = []
        
        # Chrome 옵션 설정
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        
        # 기본 필수 설정
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # CSP 및 보안 정책 우회
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-ipc-flooding-protection")
        chrome_options.add_argument("--disable-renderer-backgrounding")
        chrome_options.add_argument("--disable-backgrounding-occluded-windows")
        chrome_options.add_argument("--disable-client-side-phishing-detection")
        
        # Mixed Content 허용
        chrome_options.add_argument("--allow-running-insecure-content")
        chrome_options.add_argument("--ignore-certificate-errors")
        chrome_options.add_argument("--ignore-ssl-errors")
        chrome_options.add_argument("--ignore-certificate-errors-spki-list")
        
        # 자동화 탐지 우회 강화
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # 추가 우회 옵션
        chrome_options.add_argument("--disable-extensions-file-access-check")
        chrome_options.add_argument("--disable-extensions-http-throttling")
        chrome_options.add_argument("--aggressive-cache-discard")
        
        # User-Agent 설정
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
        
        # 창 크기
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--start-maximized")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # 자동화 탐지 우회 스크립트들
            self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
                'source': '''
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined,
                    });
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5],
                    });
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['ko-KR', 'ko', 'en-US', 'en'],
                    });
                    window.chrome = {
                        runtime: {},
                    };
                    Object.defineProperty(navigator, 'permissions', {
                        get: () => ({
                            query: () => Promise.resolve({ state: 'granted' }),
                        }),
                    });
                '''
            })
            
            self.wait = WebDriverWait(self.driver, wait_timeout)
            logger.info("Chrome 드라이버 초기화 완료")
            
        except Exception as e:
            logger.error(f"드라이버 초기화 실패: {e}")
            raise
    
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
    
    def login_wait(self, url):
        """로그인 대기 및 처리 - CSP 문제 해결 버전"""
        try:
            logger.info(f"페이지 접속 중: {url}")
            self.driver.get(url)
            
            # 페이지 완전 로딩 대기
            time.sleep(5)
            
            # 여러 방법으로 로그인 버튼 찾기 시도
            login_clicked = False
            
            # 1. 일반적인 방법
            try:
                login_btn = self.wait.until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "btn_signin"))
                )
                
                # DOM 조작으로 강제 클릭
                self.driver.execute_script("""
                    var element = arguments[0];
                    var event = new MouseEvent('click', {
                        view: window,
                        bubbles: true,
                        cancelable: true
                    });
                    element.dispatchEvent(event);
                """, login_btn)
                
                login_clicked = True
                logger.info("로그인 버튼 클릭 완료 (DOM 이벤트)")
                
            except Exception as e:
                logger.warning(f"DOM 이벤트 클릭 실패: {e}")
            
            # 2. CSS 셀렉터로 재시도
            if not login_clicked:
                try:
                    self.driver.execute_script("""
                        var loginBtn = document.querySelector('.btn_signin');
                        if (loginBtn) {
                            loginBtn.click();
                            return true;
                        }
                        return false;
                    """)
                    login_clicked = True
                    logger.info("로그인 버튼 클릭 완료 (직접 쿼리셀렉터)")
                except Exception as e:
                    logger.warning(f"쿼리셀렉터 클릭 실패: {e}")
            
            # 3. 텍스트 기반으로 찾기
            if not login_clicked:
                try:
                    result = self.driver.execute_script("""
                        var buttons = document.querySelectorAll('button, a');
                        for (var i = 0; i < buttons.length; i++) {
                            if (buttons[i].textContent.includes('로그인')) {
                                buttons[i].click();
                                return true;
                            }
                        }
                        return false;
                    """)
                    if result:
                        login_clicked = True
                        logger.info("로그인 버튼 클릭 완료 (텍스트 검색)")
                except Exception as e:
                    logger.warning(f"텍스트 검색 클릭 실패: {e}")
            
            if not login_clicked:
                logger.error("모든 로그인 시도 실패")
                raise Exception("로그인 버튼 클릭 실패")
            
            time.sleep(3)
            
            print("\n" + "="*60)
            print("블라인드 로그인이 필요합니다")
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
        """개별 리뷰에서 데이터 추출"""
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
                logger.warning("총점 추출 실패")
            
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
                logger.warning("제목 추출 실패")
            
            # 직원 유형 추출
            try:
                status_element = element.find_element(By.CSS_SELECTOR, "div.review_item_inr > div.auth")
                status_text = status_element.text.split("\n")
                status = status_text[1] if len(status_text) > 1 else "정보 없음"
            except (NoSuchElementException, IndexError):
                status = "정보 없음"
                logger.warning("직원 유형 추출 실패")
            
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
            
            # 결과 데이터 구성
            review_data = [
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
            
            return review_data
            
        except Exception as e:
            logger.error(f"리뷰 데이터 추출 중 오류: {e}")
            return [0.0, "0", "0", "0", "0", "0", "오류", "오류", "추출 실패", "추출 실패"]
    
    def save_rag_json(self, results: List, company_name: str, output_dir: str = None):
        """RAG 최적화 JSON 저장"""
        if not results:
            return None
        
        if output_dir is None:
            output_dir = self.output_dir
        
        # 전체 통계 계산
        avg_rating = sum(r[0] for r in results) / len(results)
        rating_distribution = {
            "high (4.0+)": sum(1 for r in results if r[0] >= 4.0),
            "medium (3.0-3.9)": sum(1 for r in results if 3.0 <= r[0] < 4.0),
            "low (<3.0)": sum(1 for r in results if r[0] < 3.0)
        }
        
        # RAG 데이터 생성
        rag_data = {
            "metadata": {
                "company": company_name,
                "crawl_timestamp": datetime.now().isoformat(),
                "crawler_version": "integrated_v1.0",
                "total_reviews": len(results),
                "avg_rating": round(avg_rating, 2),
                "rating_distribution": rating_distribution,
                "features": {
                    "semantic_content": True,
                    "keyword_content": True,
                    "advanced_keywords": True,
                    "search_metadata": True,
                    "intelligence_flags": True
                }
            },
            "reviews": []
        }
        
        # 각 리뷰를 RAG 형태로 변환
        for idx, review in enumerate(results):
            rag_review = self.rag_processor.create_rag_optimized_data(
                review, company_name, idx
            )
            rag_data["reviews"].append(rag_review)
        
        # JSON 파일로 저장
        json_filename = f"{company_name}_rag_optimized.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"RAG 최적화 JSON 저장 완료: {json_filepath}")
        return json_filename
    
    def crawl_single_company(self, company_code: str, pages: int = 25):
        """단일 기업 크롤링 (통계 포함)"""
        try:
            base_url = f"https://www.teamblind.com/kr/company/{company_code}/reviews"
            
            if not self.is_logged_in:
                self.login_wait(base_url)
            
            # 회사 통계 추출 시도
            company_stats = None
            if self.save_stats:
                try:
                    logger.info("회사 개요 통계 추출 시도...")
                    company_stats = self.stats_extractor.extract_company_overview_stats(
                        self.driver, company_code
                    )
                    logger.info(f"개요 통계 추출 성공: 평점 {company_stats['overall_rating']}, 리뷰 {company_stats['total_reviews']}개")
                except Exception as e:
                    logger.warning(f"개요 통계 추출 실패, 리뷰 기반 계산으로 대체: {e}")
            
            # 리뷰 데이터 수집
            results = []
            column_names = [
                "총점", "커리어 향상", "업무와 삶의 균형", "급여 및 복지", 
                "사내 문화", "경영진", "제목", "직원유형", "장점", "단점"
            ]
            
            logger.info(f"리뷰 크롤링 시작: 총 {pages}페이지")
            
            for page in range(1, pages + 1):
                try:
                    target_url = f"{base_url}?page={page}"
                    logger.info(f"페이지 {page}/{pages} 처리 중")
                    
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
                            results.append(review_data)
                            print(f"[{page}-{idx+1}] {review_data[6]} | 총점: {review_data[0]}")
                        except Exception as e:
                            logger.error(f"페이지 {page}, 리뷰 {idx+1} 추출 실패: {e}")
                            continue
                    
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"페이지 {page} 처리 중 오류: {e}")
                    continue
            
            # 결과 저장
            if results:
                # # 1. 기본 Excel 저장
                # df = pd.DataFrame(results, columns=column_names)
                # excel_filename = f"{company_code}_basic.xlsx"
                # excel_filepath = os.path.join(self.output_dir, excel_filename)
                # df.to_excel(excel_filepath, index=False)
                
                # 2. RAG 최적화 JSON 저장
                json_filename = self.save_rag_json(results, company_code)
                
                # 3. 회사 통계 저장
                if self.save_stats:
                    if company_stats is None or company_stats["total_reviews"] == 0:
                        logger.info("리뷰 데이터 기반으로 통계 계산...")
                        company_stats = self.stats_extractor.calculate_review_based_stats(
                            results, company_code
                        )
                    
                    stats_filepath = self.stats_extractor.save_company_stats(
                        company_stats, company_code, self.output_dir
                    )
                    
                    # 통계 요약 출력
                    print(f"\n회사 통계 정보:")
                    print(f"- 전체 평점: {company_stats['overall_rating']}/5.0")
                    print(f"- 총 리뷰 수: {company_stats['total_reviews']}개")
                    print(f"- 카테고리별 평점:")
                    for category, rating in company_stats['category_ratings'].items():
                        category_kr = self._translate_category(category)
                        print(f"  * {category_kr}: {rating}")
                
                # 4. 결과 요약
                print(f"\n크롤링 결과:")
                print(f"- 총 리뷰 수: {len(results)}개")
                print(f"- 평균 총점: {df['총점'].mean():.2f}")
                print(f"- 높은 평점 (4.0+): {sum(1 for r in results if r[0] >= 4.0)}개")
                # print(f"- Excel 파일: {excel_filename}")
                print(f"- RAG JSON: {json_filename}")
                
                return True
            else:
                logger.warning("추출된 리뷰 데이터가 없습니다")
                return False
                
        except Exception as e:
            logger.error(f"크롤링 중 치명적 오류: {e}")
            raise
    
    def crawl_multiple_companies(self, company_codes: List[str], pages_per_company: int = 25, 
                               delay_between_companies: int = 30):
        """배치 크롤링: 여러 기업을 순차적으로 크롤링"""
        
        total_companies = len(company_codes)
        logger.info(f"배치 크롤링 시작: {total_companies}개 기업")
        print(f"\n{'='*60}")
        print(f"배치 크롤링 시작: {total_companies}개 기업")
        print(f"기업 코드: {', '.join(company_codes)}")
        print(f"기업당 페이지: {pages_per_company}")
        print(f"{'='*60}\n")
        
        try:
            for idx, company_code in enumerate(company_codes, 1):
                logger.info(f"\n[{idx}/{total_companies}] {company_code} 크롤링 시작")
                print(f"\n[{idx}/{total_companies}] {company_code} 크롤링 시작")
                
                try:
                    start_time = datetime.now()
                    
                    # 크롤링 실행
                    success = self.crawl_single_company(company_code, pages_per_company)
                    
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    if success:
                        # 성공 기록
                        self.results_summary.append({
                            "company": company_code,
                            "status": "success",
                            "duration": duration,
                            "pages": pages_per_company,
                            "timestamp": end_time.isoformat()
                        })
                        
                        print(f"{company_code} 완료 ({duration:.1f}초)")
                    else:
                        self.failed_companies.append({
                            "company": company_code,
                            "error": "데이터 없음",
                            "timestamp": end_time.isoformat()
                        })
                    
                    # 다음 기업 전 대기 (마지막 기업 제외)
                    if idx < total_companies:
                        print(f"다음 기업까지 {delay_between_companies}초 대기...")
                        time.sleep(delay_between_companies)
                
                except Exception as e:
                    logger.error(f"{company_code} 크롤링 실패: {e}")
                    print(f"{company_code} 실패: {e}")
                    
                    self.failed_companies.append({
                        "company": company_code,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # 에러 발생 시에도 계속 진행
                    print(f"에러 무시하고 다음 기업 진행...")
                    continue
        
        finally:
            # 통합 통계 파일 생성
            if self.save_stats:
                self._create_combined_stats_file()
            
            # 최종 결과 출력
            self._print_batch_summary()
    
    def _create_combined_stats_file(self):
        """모든 기업의 통계를 하나의 파일로 통합"""
        try:
            # 개별 통계 파일들 수집
            stats_files = glob.glob(os.path.join(self.output_dir, "*_company_stats.json"))
            combined_stats = {
                "generated_at": datetime.now().isoformat(),
                "total_companies": len(stats_files),
                "companies": {}
            }
            
            for stats_file in stats_files:
                try:
                    with open(stats_file, "r", encoding="utf-8") as f:
                        company_data = json.load(f)
                        company_name = company_data.get("company_name", "unknown")
                        combined_stats["companies"][company_name] = company_data
                except Exception as e:
                    logger.warning(f"통계 파일 {stats_file} 읽기 실패: {e}")
            
            # 통합 파일 저장
            combined_file = os.path.join(self.output_dir, "all_companies_stats.json")
            with open(combined_file, "w", encoding="utf-8") as f:
                json.dump(combined_stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"통합 통계 파일 생성: {combined_file}")
            print(f"통합 통계 파일: {os.path.basename(combined_file)}")
            
        except Exception as e:
            logger.error(f"통합 통계 파일 생성 실패: {e}")
    
    def _print_batch_summary(self):
        """배치 크롤링 결과 요약"""
        print(f"\n{'='*60}")
        print("배치 크롤링 완료")
        print(f"{'='*60}")
        
        if self.results_summary:
            print(f"성공: {len(self.results_summary)}개 기업")
            total_pages = 0
            total_time = 0
            for result in self.results_summary:
                print(f"  - {result['company']}: {result['pages']}페이지 ({result['duration']:.1f}초)")
                total_pages += result['pages']
                total_time += result['duration']
            
            print(f"\n전체 통계:")
            print(f"  - 총 페이지: {total_pages}페이지")
            print(f"  - 총 소요시간: {total_time/3600:.1f}시간")
            print(f"  - 평균 속도: {total_pages/total_time*60:.1f}페이지/분")
        
        if self.failed_companies:
            print(f"\n실패: {len(self.failed_companies)}개 기업")
            for failed in self.failed_companies:
                print(f"  - {failed['company']}: {failed['error']}")
        
        if self.results_summary or self.failed_companies:
            success_rate = len(self.results_summary) / (len(self.results_summary) + len(self.failed_companies)) * 100
            print(f"\n성공률: {success_rate:.1f}%")
        
        print(f"\n저장 위치: {self.output_dir}")
        print(f"{'='*60}")
    
    def _translate_category(self, category_en: str) -> str:
        """카테고리 영문명을 한글로 변환"""
        translations = {
            "career_growth": "커리어 향상",
            "work_life_balance": "워라밸", 
            "salary_benefits": "급여/복지",
            "company_culture": "사내 문화",
            "management": "경영진"
        }
        return translations.get(category_en, category_en)
    
    def close(self):
        """드라이버 종료"""
        if self.driver:
            self.driver.quit()
            logger.info("드라이버 종료 완료")


# 실행 함수들


def run_batch_crawling():
    """배치 크롤링 실행 함수 - 기본 기업 리스트"""
    
    # 크롤링할 기업 코드 리스트
    company_codes = [
        "삼성전자", "SK하이닉스", "LG전자", "현대자동차", "기아",
        "네이버", "카카오", "쿠팡", "엔씨소프트", "넷마블",
        "KB금융지주", "신한지주", "하나금융지주", "포스코", "SK텔레콤"
    ]
    
    # 설정
    pages_per_company = 25  # 기업당 크롤링할 페이지 수
    delay_between_companies = 30  # 기업 간 대기 시간 (초)
    
    print("배치 크롤링 설정:")
    print(f"- 기업 수: {len(company_codes)}개")
    print(f"- 기업 코드: {', '.join(company_codes[:5])}... (총 {len(company_codes)}개)")
    print(f"- 기업당 페이지: {pages_per_company}")
    print(f"- 기업간 대기시간: {delay_between_companies}초")
    print(f"- 예상 소요시간: {len(company_codes) * pages_per_company * 10 / 3600:.1f}시간")
    
    # 사용자 확인
    if input("\n계속 진행하시겠습니까? (y/N): ").lower() not in ['y', 'yes']:
        print("배치 크롤링이 취소되었습니다.")
        return
    
    # 배치 크롤링 실행
    crawler = BlindReviewCrawler(
        headless=False,
        save_stats=True,
        output_dir="./data/reviews"
    )
    
    try:
        crawler.crawl_multiple_companies(
            company_codes=company_codes,
            pages_per_company=pages_per_company,
            delay_between_companies=delay_between_companies
        )
    except KeyboardInterrupt:
        print("\n사용자에 의해 배치 크롤링이 중단되었습니다.")
    except Exception as e:
        print(f"\n배치 크롤링 중 오류 발생: {e}")
    finally:
        crawler.close()

def run_custom_batch_crawl(company_codes: List[str], pages: int = 25, delay: int = 30):
    """커스텀 배치 크롤링 함수"""
    crawler = BlindReviewCrawler(
        headless=False,
        save_stats=True,
        output_dir="./data/reviews"
    )
    
    try:
        crawler.crawl_multiple_companies(
            company_codes=company_codes,
            pages_per_company=pages,
            delay_between_companies=delay
        )
    finally:
        crawler.close()


# 메인 실행부
def main():
    """메인 실행 함수"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                 통합 블라인드 크롤링 도구                      ║
    ║              (배치 처리 + 회사 통계 + RAG 지원)                ║
    ║                                                              ║
    ║  1. 단일 기업 크롤링                                           ║
    ║  2. 배치 크롤링 (기본 기업 리스트)                             ║
    ║  3. 커스텀 배치 크롤링                                         ║
    ║                                                              ║
    ║  생성 파일:                                                   ║
    ║  - [회사명]_basic.xlsx (상세 리뷰 데이터)                      ║
    ║  - [회사명]_rag_optimized.json (RAG용 최적화 데이터)           ║
    ║  - [회사명]_company_stats.json (홈페이지용 통계)               ║
    ║  - all_companies_stats.json (통합 통계)                       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
 
    # 배치 크롤링 (기본 리스트)
    print("\n기본 기업 리스트로 배치 크롤링을 시작합니다...")
    run_batch_crawling()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n\n프로그램 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()