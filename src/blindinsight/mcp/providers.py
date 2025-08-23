"""
외부 데이터 제공자 클래스들

다양한 외부 API로부터 회사 정보, 채용 데이터, 연봉 정보를 수집하는
데이터 제공자들을 정의합니다.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import json
import re

from .client import APIEndpoint
from ..models.company import BlindPost, CompanyReview, SalaryInfo
from ..models.base import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataProviderConfig:
    """데이터 제공자 설정 정보"""
    
    name: str                        # 제공자 이름
    base_url: str                    # 기본 URL
    api_key: Optional[str] = None    # API 키
    rate_limit: int = 100            # 분당 요청 제한
    timeout: int = 30                # 타임아웃 (초)
    retry_attempts: int = 3          # 재시도 횟수
    cache_ttl: int = 3600           # 캐시 유지 시간 (초)
    enabled: bool = True             # 활성화 여부


class BaseDataProvider(ABC):
    """
    기본 데이터 제공자 추상 클래스
    
    모든 데이터 제공자가 구현해야 하는 공통 인터페이스를 정의합니다.
    """
    
    def __init__(self, config: DataProviderConfig, mcp_client):
        """
        기본 데이터 제공자 초기화
        
        Args:
            config: 제공자 설정
            mcp_client: MCP 클라이언트
        """
        self.config = config
        self.mcp_client = mcp_client
        self.last_fetch_time: Optional[datetime] = None
        self.fetch_count = 0
        self.error_count = 0
        
        # API 엔드포인트 등록
        self._register_endpoints()
        
        logger.info(f"{self.config.name} 데이터 제공자 초기화 완료")
    
    @abstractmethod
    def _register_endpoints(self):
        """API 엔드포인트들을 등록하는 추상 메소드"""
        pass
    
    @abstractmethod
    async def fetch_company_data(self, company_name: str) -> List[Dict]:
        """회사 데이터를 가져오는 추상 메소드"""
        pass
    
    def _clean_text(self, text: str) -> str:
        """
        텍스트 정리 및 정규화
        
        Args:
            text: 원본 텍스트
            
        Returns:
            정리된 텍스트
        """
        if not text:
            return ""
        
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수 문자 정리
        text = re.sub(r'[^\w\s가-힣.,!?-]', '', text)
        
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def _validate_company_name(self, company_name: str) -> bool:
        """
        회사명 유효성 검증
        
        Args:
            company_name: 검증할 회사명
            
        Returns:
            유효성 여부
        """
        if not company_name or len(company_name.strip()) < 2:
            return False
        
        # 금지된 키워드 확인
        forbidden_keywords = ["test", "테스트", "example", "예시"]
        for keyword in forbidden_keywords:
            if keyword.lower() in company_name.lower():
                return False
        
        return True
    
    async def _make_request(
        self, 
        endpoint_name: str, 
        params: Dict = None
    ) -> Optional[Dict]:
        """
        API 요청 수행 (에러 처리 포함)
        
        Args:
            endpoint_name: 엔드포인트 이름
            params: 요청 매개변수
            
        Returns:
            응답 데이터 또는 None
        """
        try:
            self.fetch_count += 1
            result = await self.mcp_client.make_request(endpoint_name, params)
            
            if result:
                self.last_fetch_time = datetime.now()
                return result
            else:
                self.error_count += 1
                return None
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"{self.config.name} API 요청 실패: {str(e)}")
            return None
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """제공자 통계 정보 반환"""
        return {
            "name": self.config.name,
            "enabled": self.config.enabled,
            "fetch_count": self.fetch_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.fetch_count, 1),
            "last_fetch_time": self.last_fetch_time.isoformat() if self.last_fetch_time else None
        }


class BlindDataProvider(BaseDataProvider):
    """
    Blind 앱 데이터 제공자
    
    Blind 플랫폼에서 익명 회사 리뷰와 토론 데이터를 수집합니다.
    실제 Blind API는 제한적이므로 시뮬레이션 데이터를 제공합니다.
    """
    
    def _register_endpoints(self):
        """Blind API 엔드포인트 등록"""
        
        # 회사 리뷰 조회 엔드포인트
        company_reviews_endpoint = APIEndpoint(
            name="blind_company_reviews",
            url=f"{self.config.base_url}/api/v1/companies/{{company_id}}/reviews",
            method="GET",
            headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
            rate_limit=self.config.rate_limit,
            timeout=self.config.timeout,
            cache_ttl=self.config.cache_ttl
        )
        
        # 연봉 정보 조회 엔드포인트
        salary_info_endpoint = APIEndpoint(
            name="blind_salary_info",
            url=f"{self.config.base_url}/api/v1/companies/{{company_id}}/salaries",
            method="GET",
            headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
            rate_limit=self.config.rate_limit,
            timeout=self.config.timeout,
            cache_ttl=self.config.cache_ttl
        )
        
        # 회사 토론 조회 엔드포인트
        discussions_endpoint = APIEndpoint(
            name="blind_discussions",
            url=f"{self.config.base_url}/api/v1/companies/{{company_id}}/discussions",
            method="GET",
            headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
            rate_limit=self.config.rate_limit,
            timeout=self.config.timeout,
            cache_ttl=self.config.cache_ttl
        )
        
        # 엔드포인트들을 MCP 클라이언트에 등록
        self.mcp_client.register_endpoint(company_reviews_endpoint)
        self.mcp_client.register_endpoint(salary_info_endpoint)
        self.mcp_client.register_endpoint(discussions_endpoint)
    
    async def fetch_company_data(self, company_name: str) -> List[Dict]:
        """
        회사의 Blind 데이터 수집
        
        Args:
            company_name: 회사명
            
        Returns:
            수집된 데이터 리스트
        """
        if not self._validate_company_name(company_name):
            logger.warning(f"유효하지 않은 회사명: {company_name}")
            return []
        
        all_data = []
        
        try:
            # 실제 API 대신 시뮬레이션 데이터 생성
            # 실제 환경에서는 진짜 Blind API를 호출해야 합니다
            
            # 회사 리뷰 데이터 시뮬레이션
            reviews_data = await self._simulate_company_reviews(company_name)
            all_data.extend(reviews_data)
            
            # 연봉 정보 데이터 시뮬레이션
            salary_data = await self._simulate_salary_info(company_name)
            all_data.extend(salary_data)
            
            # 토론 데이터 시뮬레이션
            discussion_data = await self._simulate_discussions(company_name)
            all_data.extend(discussion_data)
            
            logger.info(f"{company_name} Blind 데이터 수집 완료: {len(all_data)}개 항목")
            return all_data
            
        except Exception as e:
            logger.error(f"{company_name} Blind 데이터 수집 실패: {str(e)}")
            return []
    
    async def _simulate_company_reviews(self, company_name: str) -> List[Dict]:
        """회사 리뷰 시뮬레이션 데이터 생성"""
        
        # 실제 환경에서는 이 부분을 실제 API 호출로 대체
        sample_reviews = [
            {
                "type": "company_review",
                "company_name": company_name,
                "title": f"{company_name} 문화와 워라밸에 대한 솔직한 후기",
                "content": f"{company_name}에서 2년 근무한 경험을 공유합니다. 전반적으로 자유로운 분위기이고 워라밸이 좋은 편입니다. 동료들도 친절하고 서로 도와주는 문화가 있습니다.",
                "rating": 4.2,
                "pros": ["자유로운 분위기", "좋은 워라밸", "친절한 동료"],
                "cons": ["승진 기회 부족", "연봉 상승폭 아쉬움"],
                "department": "개발팀",
                "position": "시니어 개발자",
                "employment_status": "현직",
                "work_years": 2,
                "created_at": datetime.now() - timedelta(days=30),
                "anonymous_id": "user_001"
            },
            {
                "type": "company_review", 
                "company_name": company_name,
                "title": f"{company_name} 성장 기회와 복지에 대해",
                "content": f"{company_name}는 빠르게 성장하는 회사로 다양한 프로젝트 경험을 할 수 있습니다. 복지제도도 나쁘지 않고 교육 지원도 있습니다.",
                "rating": 3.8,
                "pros": ["다양한 프로젝트", "교육 지원", "성장 기회"],
                "cons": ["업무 강도 높음", "출퇴근 시간 자유도 부족"],
                "department": "기획팀",
                "position": "매니저",
                "employment_status": "전직",
                "work_years": 3,
                "created_at": datetime.now() - timedelta(days=15),
                "anonymous_id": "user_002"
            }
        ]
        
        return sample_reviews
    
    async def _simulate_salary_info(self, company_name: str) -> List[Dict]:
        """연봉 정보 시뮬레이션 데이터 생성"""
        
        sample_salaries = [
            {
                "type": "salary_info",
                "company_name": company_name,
                "position": "시니어 개발자",
                "department": "개발팀",
                "total_compensation": 85000000,  # 8500만원
                "base_salary": 70000000,
                "bonus": 10000000,
                "stock_options": 5000000,
                "experience_years": 5,
                "education": "대학교 졸업",
                "location": "서울",
                "employment_type": "정규직",
                "created_at": datetime.now() - timedelta(days=20),
                "anonymous_id": "salary_001"
            },
            {
                "type": "salary_info",
                "company_name": company_name,
                "position": "프로덕트 매니저",
                "department": "기획팀",
                "total_compensation": 95000000,  # 9500만원
                "base_salary": 80000000,
                "bonus": 12000000,
                "stock_options": 3000000,
                "experience_years": 7,
                "education": "대학원 졸업",
                "location": "서울",
                "employment_type": "정규직",
                "created_at": datetime.now() - timedelta(days=10),
                "anonymous_id": "salary_002"
            }
        ]
        
        return sample_salaries
    
    async def _simulate_discussions(self, company_name: str) -> List[Dict]:
        """토론 시뮬레이션 데이터 생성"""
        
        sample_discussions = [
            {
                "type": "blind_post",
                "company_name": company_name,
                "title": f"{company_name} 면접 준비 팁 공유",
                "content": f"{company_name} 면접을 준비하시는 분들을 위해 팁을 공유합니다. 기술 면접보다는 문화 적합성을 많이 봅니다.",
                "category": "면접",
                "views": 1250,
                "likes": 45,
                "comments": 23,
                "created_at": datetime.now() - timedelta(days=5),
                "anonymous_id": "post_001"
            },
            {
                "type": "blind_post",
                "company_name": company_name,
                "title": f"{company_name} 신입 개발자 현실",
                "content": f"{company_name}에 신입으로 입사한 지 6개월이 되어 현실적인 이야기를 해드리고 싶습니다. 멘토링 시스템이 잘 되어 있어 도움이 많이 됩니다.",
                "category": "커리어",
                "views": 2100,
                "likes": 78,
                "comments": 42,
                "created_at": datetime.now() - timedelta(days=2),
                "anonymous_id": "post_002"
            }
        ]
        
        return sample_discussions


class JobSiteProvider(BaseDataProvider):
    """
    채용 사이트 데이터 제공자
    
    원티드, 잡코리아, 사람인 등의 채용 사이트에서
    채용 공고와 회사 정보를 수집합니다.
    """
    
    def _register_endpoints(self):
        """채용 사이트 API 엔드포인트 등록"""
        
        # 채용 공고 조회 엔드포인트
        job_postings_endpoint = APIEndpoint(
            name="job_postings",
            url=f"{self.config.base_url}/api/v1/jobs",
            method="GET",
            headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
            rate_limit=self.config.rate_limit,
            timeout=self.config.timeout,
            cache_ttl=self.config.cache_ttl
        )
        
        # 회사 정보 조회 엔드포인트
        company_info_endpoint = APIEndpoint(
            name="company_info",
            url=f"{self.config.base_url}/api/v1/companies/{{company_id}}",
            method="GET",
            headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
            rate_limit=self.config.rate_limit,
            timeout=self.config.timeout,
            cache_ttl=self.config.cache_ttl
        )
        
        self.mcp_client.register_endpoint(job_postings_endpoint)
        self.mcp_client.register_endpoint(company_info_endpoint)
    
    async def fetch_company_data(self, company_name: str) -> List[Dict]:
        """
        회사의 채용 관련 데이터 수집
        
        Args:
            company_name: 회사명
            
        Returns:
            수집된 채용 데이터 리스트
        """
        if not self._validate_company_name(company_name):
            return []
        
        try:
            # 시뮬레이션 데이터 (실제로는 API 호출)
            job_data = await self._simulate_job_postings(company_name)
            company_info = await self._simulate_company_info(company_name)
            
            all_data = job_data + company_info
            
            logger.info(f"{company_name} 채용 데이터 수집 완료: {len(all_data)}개 항목")
            return all_data
            
        except Exception as e:
            logger.error(f"{company_name} 채용 데이터 수집 실패: {str(e)}")
            return []
    
    async def _simulate_job_postings(self, company_name: str) -> List[Dict]:
        """채용 공고 시뮬레이션 데이터"""
        
        sample_jobs = [
            {
                "type": "job_posting",
                "company_name": company_name,
                "title": "시니어 백엔드 개발자",
                "description": f"{company_name}에서 확장성 있는 백엔드 시스템을 개발할 시니어 개발자를 찾습니다. Python, Django, AWS 경험이 필요합니다.",
                "requirements": ["Python", "Django", "AWS", "5년 이상 경험"],
                "benefits": ["자율 출퇴근", "맥북 지급", "교육비 지원", "건강검진"],
                "salary_range": "6000만원 - 9000만원",
                "location": "서울 강남구",
                "employment_type": "정규직",
                "experience_level": "시니어",
                "department": "개발팀",
                "posted_at": datetime.now() - timedelta(days=7),
                "deadline": datetime.now() + timedelta(days=23)
            },
            {
                "type": "job_posting",
                "company_name": company_name,
                "title": "프론트엔드 개발자",
                "description": f"{company_name}의 사용자 경험을 혁신할 프론트엔드 개발자를 모집합니다. React, TypeScript 경험 우대합니다.",
                "requirements": ["React", "TypeScript", "3년 이상 경험"],
                "benefits": ["유연근무", "간식 제공", "성과급", "스톡옵션"],
                "salary_range": "5000만원 - 7500만원",
                "location": "서울 서초구",
                "employment_type": "정규직",
                "experience_level": "미들",
                "department": "개발팀",
                "posted_at": datetime.now() - timedelta(days=3),
                "deadline": datetime.now() + timedelta(days=27)
            }
        ]
        
        return sample_jobs
    
    async def _simulate_company_info(self, company_name: str) -> List[Dict]:
        """회사 정보 시뮬레이션 데이터"""
        
        company_info = [{
            "type": "company_info",
            "company_name": company_name,
            "industry": "IT/소프트웨어",
            "size": "중견기업 (100-300명)",
            "founded_year": 2015,
            "description": f"{company_name}는 혁신적인 기술로 사용자 경험을 개선하는 IT 회사입니다.",
            "location": "서울특별시",
            "website": f"https://www.{company_name.lower()}.com",
            "benefits": [
                "4대보험", "퇴직금", "연차", "반차", "교육비 지원",
                "건강검진", "경조사 지원", "야근시 식대 지원"
            ],
            "culture_keywords": ["자율성", "성장", "협업", "혁신", "워라밸"],
            "rating": 4.1,
            "updated_at": datetime.now()
        }]
        
        return company_info


class SalaryDataProvider(BaseDataProvider):
    """
    연봉 정보 전문 데이터 제공자
    
    잡플래닛, 크레딧잡 등의 연봉 정보 사이트에서
    상세한 보상 패키지 정보를 수집합니다.
    """
    
    def _register_endpoints(self):
        """연봉 정보 API 엔드포인트 등록"""
        
        salary_endpoint = APIEndpoint(
            name="salary_data",
            url=f"{self.config.base_url}/api/v1/salaries",
            method="GET",
            headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
            rate_limit=self.config.rate_limit,
            timeout=self.config.timeout,
            cache_ttl=self.config.cache_ttl
        )
        
        compensation_endpoint = APIEndpoint(
            name="compensation_data",
            url=f"{self.config.base_url}/api/v1/compensation",
            method="GET",
            headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
            rate_limit=self.config.rate_limit,
            timeout=self.config.timeout,
            cache_ttl=self.config.cache_ttl
        )
        
        self.mcp_client.register_endpoint(salary_endpoint)
        self.mcp_client.register_endpoint(compensation_endpoint)
    
    async def fetch_company_data(self, company_name: str) -> List[Dict]:
        """회사의 연봉 정보 수집"""
        
        if not self._validate_company_name(company_name):
            return []
        
        try:
            # 시뮬레이션 데이터
            salary_data = await self._simulate_detailed_salary_info(company_name)
            
            logger.info(f"{company_name} 연봉 데이터 수집 완료: {len(salary_data)}개 항목")
            return salary_data
            
        except Exception as e:
            logger.error(f"{company_name} 연봉 데이터 수집 실패: {str(e)}")
            return []
    
    async def _simulate_detailed_salary_info(self, company_name: str) -> List[Dict]:
        """상세 연봉 정보 시뮬레이션"""
        
        positions = [
            "신입 개발자", "주니어 개발자", "시니어 개발자", "팀 리드",
            "프로덕트 매니저", "디자이너", "마케터", "데이터 분석가"
        ]
        
        salary_data = []
        
        for i, position in enumerate(positions):
            data = {
                "type": "detailed_salary",
                "company_name": company_name,
                "position": position,
                "level": f"Level {i+1}",
                "base_salary": 45000000 + (i * 15000000),  # 기본급
                "variable_pay": 5000000 + (i * 5000000),   # 변동급
                "stock_value": i * 10000000,               # 스톡 가치
                "benefits_value": 5000000,                 # 복리후생 가치
                "total_compensation": 55000000 + (i * 30000000),
                "experience_range": f"{i}-{i+2}년",
                "education_requirement": "대학교 졸업" if i < 4 else "대학원 졸업",
                "skills_required": ["Python", "Java", "SQL"] if "개발자" in position else ["기획", "분석", "소통"],
                "location": "서울",
                "last_updated": datetime.now() - timedelta(days=i*2),
                "sample_size": 15 + i*5  # 샘플 수
            }
            salary_data.append(data)
        
        return salary_data


class CompanyNewsProvider(BaseDataProvider):
    """
    회사 뉴스 및 미디어 데이터 제공자
    
    뉴스 사이트, 프레스 릴리스, 소셜 미디어에서
    회사 관련 최신 뉴스와 여론을 수집합니다.
    """
    
    def _register_endpoints(self):
        """뉴스 API 엔드포인트 등록"""
        
        news_endpoint = APIEndpoint(
            name="company_news",
            url=f"{self.config.base_url}/api/v1/news",
            method="GET",
            headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
            rate_limit=self.config.rate_limit,
            timeout=self.config.timeout,
            cache_ttl=1800  # 30분 캐시 (뉴스는 자주 업데이트)
        )
        
        social_media_endpoint = APIEndpoint(
            name="social_mentions",
            url=f"{self.config.base_url}/api/v1/social",
            method="GET",
            headers={"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {},
            rate_limit=self.config.rate_limit,
            timeout=self.config.timeout,
            cache_ttl=900  # 15분 캐시
        )
        
        self.mcp_client.register_endpoint(news_endpoint)
        self.mcp_client.register_endpoint(social_media_endpoint)
    
    async def fetch_company_data(self, company_name: str) -> List[Dict]:
        """회사 뉴스 및 미디어 데이터 수집"""
        
        if not self._validate_company_name(company_name):
            return []
        
        try:
            news_data = await self._simulate_news_data(company_name)
            social_data = await self._simulate_social_mentions(company_name)
            
            all_data = news_data + social_data
            
            logger.info(f"{company_name} 뉴스/미디어 데이터 수집 완료: {len(all_data)}개 항목")
            return all_data
            
        except Exception as e:
            logger.error(f"{company_name} 뉴스/미디어 데이터 수집 실패: {str(e)}")
            return []
    
    async def _simulate_news_data(self, company_name: str) -> List[Dict]:
        """뉴스 데이터 시뮬레이션"""
        
        news_articles = [
            {
                "type": "news_article",
                "company_name": company_name,
                "title": f"{company_name}, 신규 서비스 출시로 매출 20% 증가",
                "content": f"{company_name}가 새로운 AI 기반 서비스를 출시하며 전년 대비 매출이 20% 증가했다고 발표했습니다.",
                "source": "테크뉴스",
                "author": "김기자",
                "published_at": datetime.now() - timedelta(days=1),
                "sentiment": "positive",
                "category": "비즈니스",
                "url": f"https://technews.com/{company_name}-revenue-growth"
            },
            {
                "type": "news_article",
                "company_name": company_name,
                "title": f"{company_name}, 개발자 복지 확대로 인재 영입 가속화",
                "content": f"{company_name}가 개발자 대상 복리후생을 대폭 확대하며 우수 인재 영입에 박차를 가하고 있습니다.",
                "source": "IT타임즈",
                "author": "박기자",
                "published_at": datetime.now() - timedelta(days=3),
                "sentiment": "positive",
                "category": "인사",
                "url": f"https://ittimes.com/{company_name}-developer-benefits"
            }
        ]
        
        return news_articles
    
    async def _simulate_social_mentions(self, company_name: str) -> List[Dict]:
        """소셜 미디어 언급 시뮬레이션"""
        
        social_mentions = [
            {
                "type": "social_mention",
                "company_name": company_name,
                "platform": "트위터",
                "content": f"{company_name} 면접 봤는데 분위기가 정말 좋더라. 질문도 적절하고 직원들이 친절했음",
                "sentiment": "positive",
                "engagement": 25,  # 좋아요, 리트윗 등
                "posted_at": datetime.now() - timedelta(hours=12),
                "influence_score": 0.7
            },
            {
                "type": "social_mention",
                "company_name": company_name,
                "platform": "링크드인",
                "content": f"{company_name}에서 근무하며 많이 성장할 수 있었습니다. 좋은 동료들과 함께 일할 수 있어 감사했습니다.",
                "sentiment": "positive",
                "engagement": 45,
                "posted_at": datetime.now() - timedelta(days=1),
                "influence_score": 0.8
            }
        ]
        
        return social_mentions