"""
MCP 데이터 처리 핸들러

외부에서 수집된 다양한 형태의 데이터를 BlindInsight AI의
내부 데이터 모델로 변환하고 처리하는 핸들러들을 정의합니다.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

from ..models.company import BlindPost, CompanyReview, SalaryInfo
from ..models.base import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """데이터 처리 결과를 담는 데이터 클래스"""
    
    success: bool                    # 처리 성공 여부
    processed_count: int             # 처리된 항목 수
    failed_count: int               # 실패한 항목 수
    created_objects: List[Any]      # 생성된 객체들
    errors: List[str]               # 오류 메시지들
    processing_time: float          # 처리 시간 (초)
    metadata: Dict[str, Any]        # 추가 메타데이터


class DataTransformer:
    """
    데이터 변환 유틸리티 클래스
    
    외부 API 데이터를 내부 모델 형식으로 변환하는 정적 메서드들을 제공합니다.
    """
    
    @staticmethod
    def normalize_company_name(company_name: str) -> str:
        """
        회사명 정규화
        
        Args:
            company_name: 원본 회사명
            
        Returns:
            정규화된 회사명
        """
        if not company_name:
            return ""
        
        # 공백 제거 및 대소문자 정리
        normalized = company_name.strip()
        
        # 괄호 안의 내용 제거 (예: "네이버(주)" -> "네이버")
        import re
        normalized = re.sub(r'\([^)]*\)', '', normalized).strip()
        
        # 회사 접미사 정리
        suffixes = ["주식회사", "(주)", "㈜", "Corp.", "Inc.", "Co.", "Ltd."]
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        return normalized
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """
        텍스트에서 키워드 추출
        
        Args:
            text: 대상 텍스트
            max_keywords: 최대 키워드 수
            
        Returns:
            키워드 리스트
        """
        if not text:
            return []
        
        # 한국어 업무 관련 키워드들
        important_keywords = [
            "워라밸", "야근", "복지", "연봉", "승진", "성장", "분위기", "문화",
            "동료", "상사", "프로젝트", "개발", "기술", "교육", "자율", "자유",
            "스트레스", "압박", "만족", "추천", "이직", "경력", "신입", "경험"
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in important_keywords:
            if keyword in text_lower and keyword not in found_keywords:
                found_keywords.append(keyword)
                if len(found_keywords) >= max_keywords:
                    break
        
        return found_keywords
    
    @staticmethod
    def calculate_sentiment_score(text: str) -> float:
        """
        텍스트의 감정 점수 계산 (간단한 키워드 기반)
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            감정 점수 (-1.0 ~ 1.0, 음수: 부정, 양수: 긍정)
        """
        if not text:
            return 0.0
        
        positive_words = [
            "좋다", "만족", "추천", "훌륭", "우수", "성장", "기회", "자유",
            "친절", "도움", "발전", "혁신", "즐거", "행복", "보람", "가치"
        ]
        
        negative_words = [
            "나쁘다", "불만", "힘들다", "스트레스", "과로", "문제", "어려움",
            "실망", "불안", "걱정", "압박", "부족", "아쉬움", "한계", "제약"
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # 정규화된 감정 점수 계산
        sentiment_score = (positive_count - negative_count) / max(total_words, 1)
        
        # -1.0 ~ 1.0 범위로 제한
        return max(-1.0, min(1.0, sentiment_score * 10))
    
    @staticmethod
    def parse_salary_amount(salary_text: str) -> Optional[int]:
        """
        연봉 텍스트에서 금액 추출
        
        Args:
            salary_text: 연봉 관련 텍스트
            
        Returns:
            연봉 금액 (원) 또는 None
        """
        if not salary_text:
            return None
        
        import re
        
        # 숫자와 단위 추출
        # "5000만원", "50,000,000원", "5천만원" 등 다양한 형태 지원
        patterns = [
            r'(\d{1,3}(?:,\d{3})*)\s*만원',      # "5,000만원"
            r'(\d{1,3}(?:,\d{3})*)\s*원',        # "50,000,000원"
            r'(\d+)\s*천만원',                   # "5천만원"
            r'(\d+)\s*억',                       # "1억"
            r'(\d+\.?\d*)\s*만원'                # "5000.0만원"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, salary_text)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount = float(amount_str)
                    
                    if '만원' in salary_text:
                        return int(amount * 10000)
                    elif '천만원' in salary_text:
                        return int(amount * 10000000)
                    elif '억' in salary_text:
                        return int(amount * 100000000)
                    else:
                        return int(amount)
                        
                except ValueError:
                    continue
        
        return None


class BaseDataHandler(ABC):
    """
    기본 데이터 핸들러 추상 클래스
    
    모든 데이터 핸들러가 구현해야 하는 공통 인터페이스를 정의합니다.
    """
    
    def __init__(self, knowledge_base=None):
        """
        기본 데이터 핸들러 초기화
        
        Args:
            knowledge_base: 지식 베이스 인스턴스
        """
        self.knowledge_base = knowledge_base
        self.transformer = DataTransformer()
        self.processing_stats = {
            "total_processed": 0,
            "total_failed": 0,
            "last_processing_time": None
        }
    
    @abstractmethod
    async def process_data(self, raw_data: List[Dict]) -> ProcessingResult:
        """원시 데이터를 처리하는 추상 메소드"""
        pass
    
    def _validate_required_fields(self, data: Dict, required_fields: List[str]) -> bool:
        """
        필수 필드 검증
        
        Args:
            data: 검증할 데이터
            required_fields: 필수 필드 리스트
            
        Returns:
            검증 통과 여부
        """
        for field in required_fields:
            if field not in data or not data[field]:
                return False
        return True
    
    def _clean_and_validate_text(self, text: str, max_length: int = 5000) -> str:
        """
        텍스트 정리 및 검증
        
        Args:
            text: 원본 텍스트
            max_length: 최대 길이
            
        Returns:
            정리된 텍스트
        """
        if not text:
            return ""
        
        # HTML 태그 제거
        import re
        text = re.sub(r'<[^>]+>', '', text)
        
        # 특수 문자 정리 (한글, 영문, 숫자, 기본 구두점만 유지)
        text = re.sub(r'[^\w\s가-힣.,!?\-()[\]{}"]', '', text)
        
        # 연속된 공백 정리
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 길이 제한
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 정보 반환"""
        return self.processing_stats.copy()


class BlindDataHandler(BaseDataHandler):
    """
    Blind 데이터 전용 핸들러
    
    Blind에서 수집된 원시 데이터를 BlindPost, CompanyReview, SalaryInfo 객체로 변환합니다.
    """
    
    async def process_data(self, raw_data: List[Dict]) -> ProcessingResult:
        """
        Blind 원시 데이터를 처리
        
        Args:
            raw_data: Blind에서 수집된 원시 데이터 리스트
            
        Returns:
            처리 결과
        """
        start_time = datetime.now()
        created_objects = []
        errors = []
        processed_count = 0
        failed_count = 0
        
        try:
            for raw_item in raw_data:
                try:
                    # 데이터 타입에 따라 처리
                    data_type = raw_item.get("type", "")
                    
                    if data_type == "blind_post":
                        obj = await self._process_blind_post(raw_item)
                    elif data_type == "company_review":
                        obj = await self._process_company_review(raw_item)
                    elif data_type == "salary_info":
                        obj = await self._process_salary_info(raw_item)
                    else:
                        logger.warning(f"알 수 없는 데이터 타입: {data_type}")
                        failed_count += 1
                        continue
                    
                    if obj:
                        created_objects.append(obj)
                        processed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    error_msg = f"데이터 처리 실패: {str(e)}"
                    errors.append(error_msg)
                    failed_count += 1
                    logger.error(error_msg)
            
            # 통계 업데이트
            self.processing_stats["total_processed"] += processed_count
            self.processing_stats["total_failed"] += failed_count
            self.processing_stats["last_processing_time"] = datetime.now()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ProcessingResult(
                success=processed_count > 0,
                processed_count=processed_count,
                failed_count=failed_count,
                created_objects=created_objects,
                errors=errors,
                processing_time=processing_time,
                metadata={
                    "data_source": "blind",
                    "handler_type": "BlindDataHandler"
                }
            )
            
            logger.info(f"Blind 데이터 처리 완료: {processed_count}개 성공, {failed_count}개 실패")
            return result
            
        except Exception as e:
            logger.error(f"Blind 데이터 처리 중 오류: {str(e)}")
            return ProcessingResult(
                success=False,
                processed_count=0,
                failed_count=len(raw_data),
                created_objects=[],
                errors=[str(e)],
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": "processing_failed"}
            )
    
    async def _process_blind_post(self, raw_data: Dict) -> Optional[BlindPost]:
        """Blind 포스트 데이터 처리"""
        
        # 필수 필드 검증
        required_fields = ["company_name", "title", "content"]
        if not self._validate_required_fields(raw_data, required_fields):
            return None
        
        try:
            # BlindPost 객체 생성
            blind_post = BlindPost(
                id=raw_data.get("id", f"blind_{hash(str(raw_data))}"),
                company_name=self.transformer.normalize_company_name(
                    raw_data["company_name"]
                ),
                title=self._clean_and_validate_text(raw_data["title"], 200),
                content=self._clean_and_validate_text(raw_data["content"]),
                category=raw_data.get("category", "일반"),
                author_info={
                    "anonymous_id": raw_data.get("anonymous_id", "unknown"),
                    "department": raw_data.get("department"),
                    "position": raw_data.get("position"),
                    "work_years": raw_data.get("work_years")
                },
                engagement_metrics={
                    "views": raw_data.get("views", 0),
                    "likes": raw_data.get("likes", 0),
                    "comments": raw_data.get("comments", 0)
                },
                tags=self.transformer.extract_keywords(raw_data["content"]),
                sentiment_score=self.transformer.calculate_sentiment_score(
                    raw_data["content"]
                ),
                created_at=raw_data.get("created_at", datetime.now()),
                processed_at=datetime.now()
            )
            
            return blind_post
            
        except Exception as e:
            logger.error(f"BlindPost 생성 실패: {str(e)}")
            return None
    
    async def _process_company_review(self, raw_data: Dict) -> Optional[CompanyReview]:
        """회사 리뷰 데이터 처리"""
        
        required_fields = ["company_name", "title", "content"]
        if not self._validate_required_fields(raw_data, required_fields):
            return None
        
        try:
            company_review = CompanyReview(
                id=raw_data.get("id", f"review_{hash(str(raw_data))}"),
                company_name=self.transformer.normalize_company_name(
                    raw_data["company_name"]
                ),
                title=self._clean_and_validate_text(raw_data["title"], 200),
                content=self._clean_and_validate_text(raw_data["content"]),
                rating=float(raw_data.get("rating", 3.0)),
                pros=raw_data.get("pros", []),
                cons=raw_data.get("cons", []),
                reviewer_info={
                    "department": raw_data.get("department"),
                    "position": raw_data.get("position"),
                    "employment_status": raw_data.get("employment_status"),
                    "work_years": raw_data.get("work_years"),
                    "anonymous_id": raw_data.get("anonymous_id", "unknown")
                },
                categories={
                    "culture": self.transformer.calculate_sentiment_score(
                        raw_data.get("content", "")
                    ),
                    "compensation": 0.0,  # 별도 분석 필요
                    "work_life_balance": 0.0,  # 별도 분석 필요
                    "management": 0.0,  # 별도 분석 필요
                    "career_growth": 0.0  # 별도 분석 필요
                },
                tags=self.transformer.extract_keywords(raw_data["content"]),
                created_at=raw_data.get("created_at", datetime.now()),
                processed_at=datetime.now()
            )
            
            return company_review
            
        except Exception as e:
            logger.error(f"CompanyReview 생성 실패: {str(e)}")
            return None
    
    async def _process_salary_info(self, raw_data: Dict) -> Optional[SalaryInfo]:
        """연봉 정보 데이터 처리"""
        
        required_fields = ["company_name", "position"]
        if not self._validate_required_fields(raw_data, required_fields):
            return None
        
        try:
            # 연봉 정보 파싱
            total_compensation = raw_data.get("total_compensation")
            if isinstance(total_compensation, str):
                total_compensation = self.transformer.parse_salary_amount(total_compensation)
            
            base_salary = raw_data.get("base_salary")
            if isinstance(base_salary, str):
                base_salary = self.transformer.parse_salary_amount(base_salary)
            
            salary_info = SalaryInfo(
                id=raw_data.get("id", f"salary_{hash(str(raw_data))}"),
                company_name=self.transformer.normalize_company_name(
                    raw_data["company_name"]
                ),
                position=raw_data["position"],
                department=raw_data.get("department"),
                compensation_details={
                    "base_salary": base_salary or 0,
                    "bonus": raw_data.get("bonus", 0),
                    "stock_options": raw_data.get("stock_options", 0),
                    "benefits_value": raw_data.get("benefits_value", 0),
                    "total_compensation": total_compensation or 0
                },
                experience_info={
                    "years": raw_data.get("experience_years", 0),
                    "level": raw_data.get("level"),
                    "education": raw_data.get("education"),
                    "skills": raw_data.get("skills_required", [])
                },
                location=raw_data.get("location", "서울"),
                employment_type=raw_data.get("employment_type", "정규직"),
                reported_date=raw_data.get("created_at", datetime.now()),
                verification_status="unverified",  # 기본값
                tags=["연봉정보", raw_data.get("position", "")],
                metadata={
                    "source": "blind",
                    "anonymous_id": raw_data.get("anonymous_id"),
                    "sample_size": raw_data.get("sample_size", 1)
                },
                processed_at=datetime.now()
            )
            
            return salary_info
            
        except Exception as e:
            logger.error(f"SalaryInfo 생성 실패: {str(e)}")
            return None


class JobSiteDataHandler(BaseDataHandler):
    """
    채용 사이트 데이터 전용 핸들러
    
    채용 사이트에서 수집된 채용 공고와 회사 정보를 처리합니다.
    """
    
    async def process_data(self, raw_data: List[Dict]) -> ProcessingResult:
        """채용 사이트 원시 데이터 처리"""
        
        start_time = datetime.now()
        created_objects = []
        errors = []
        processed_count = 0
        failed_count = 0
        
        try:
            for raw_item in raw_data:
                try:
                    data_type = raw_item.get("type", "")
                    
                    if data_type == "job_posting":
                        # 채용 공고는 BlindPost로 변환하여 처리
                        obj = await self._process_job_posting(raw_item)
                    elif data_type == "company_info":
                        # 회사 정보는 CompanyReview로 변환하여 처리
                        obj = await self._process_company_info(raw_item)
                    else:
                        logger.warning(f"지원하지 않는 채용 데이터 타입: {data_type}")
                        failed_count += 1
                        continue
                    
                    if obj:
                        created_objects.append(obj)
                        processed_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    error_msg = f"채용 데이터 처리 실패: {str(e)}"
                    errors.append(error_msg)
                    failed_count += 1
                    logger.error(error_msg)
            
            # 통계 업데이트
            self.processing_stats["total_processed"] += processed_count
            self.processing_stats["total_failed"] += failed_count
            self.processing_stats["last_processing_time"] = datetime.now()
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ProcessingResult(
                success=processed_count > 0,
                processed_count=processed_count,
                failed_count=failed_count,
                created_objects=created_objects,
                errors=errors,
                processing_time=processing_time,
                metadata={
                    "data_source": "job_sites",
                    "handler_type": "JobSiteDataHandler"
                }
            )
            
            logger.info(f"채용 사이트 데이터 처리 완료: {processed_count}개 성공, {failed_count}개 실패")
            return result
            
        except Exception as e:
            logger.error(f"채용 사이트 데이터 처리 중 오류: {str(e)}")
            return ProcessingResult(
                success=False,
                processed_count=0,
                failed_count=len(raw_data),
                created_objects=[],
                errors=[str(e)],
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": "processing_failed"}
            )
    
    async def _process_job_posting(self, raw_data: Dict) -> Optional[BlindPost]:
        """채용 공고를 BlindPost로 변환"""
        
        required_fields = ["company_name", "title", "description"]
        if not self._validate_required_fields(raw_data, required_fields):
            return None
        
        try:
            # 채용 공고 내용 구성
            content_parts = [
                f"포지션: {raw_data['title']}",
                f"설명: {raw_data['description']}",
                f"요구사항: {', '.join(raw_data.get('requirements', []))}",
                f"혜택: {', '.join(raw_data.get('benefits', []))}",
                f"연봉: {raw_data.get('salary_range', '협의')}",
                f"위치: {raw_data.get('location', '서울')}",
                f"고용형태: {raw_data.get('employment_type', '정규직')}"
            ]
            
            content = "\n".join(content_parts)
            
            blind_post = BlindPost(
                id=f"job_{hash(str(raw_data))}",
                company_name=self.transformer.normalize_company_name(
                    raw_data["company_name"]
                ),
                title=f"[채용정보] {raw_data['title']}",
                content=self._clean_and_validate_text(content),
                category="채용",
                author_info={
                    "anonymous_id": "hr_team",
                    "department": "인사팀",
                    "position": "채용담당자"
                },
                engagement_metrics={"views": 0, "likes": 0, "comments": 0},
                tags=["채용", raw_data.get("department", ""), raw_data.get("experience_level", "")],
                sentiment_score=0.5,  # 중립적 감정
                created_at=raw_data.get("posted_at", datetime.now()),
                processed_at=datetime.now()
            )
            
            return blind_post
            
        except Exception as e:
            logger.error(f"채용 공고 처리 실패: {str(e)}")
            return None
    
    async def _process_company_info(self, raw_data: Dict) -> Optional[CompanyReview]:
        """회사 정보를 CompanyReview로 변환"""
        
        required_fields = ["company_name", "description"]
        if not self._validate_required_fields(raw_data, required_fields):
            return None
        
        try:
            company_review = CompanyReview(
                id=f"info_{hash(str(raw_data))}",
                company_name=self.transformer.normalize_company_name(
                    raw_data["company_name"]
                ),
                title=f"{raw_data['company_name']} 회사 정보",
                content=self._clean_and_validate_text(raw_data["description"]),
                rating=raw_data.get("rating", 4.0),
                pros=raw_data.get("benefits", []),
                cons=[],  # 회사 정보에는 단점이 없으므로 빈 리스트
                reviewer_info={
                    "department": "공식정보",
                    "position": "회사정보",
                    "employment_status": "공식",
                    "anonymous_id": "official"
                },
                categories={
                    "culture": self.transformer.calculate_sentiment_score(
                        " ".join(raw_data.get("culture_keywords", []))
                    ),
                    "compensation": 0.5,
                    "work_life_balance": 0.5,
                    "management": 0.5,
                    "career_growth": 0.5
                },
                tags=["회사정보"] + raw_data.get("culture_keywords", []),
                created_at=raw_data.get("updated_at", datetime.now()),
                processed_at=datetime.now()
            )
            
            return company_review
            
        except Exception as e:
            logger.error(f"회사 정보 처리 실패: {str(e)}")
            return None


class DataHandler:
    """
    통합 데이터 핸들러
    
    다양한 소스의 데이터를 통합 처리하고 지식 베이스에 저장하는 
    중앙 집중식 핸들러입니다.
    """
    
    def __init__(self, knowledge_base=None):
        """
        통합 데이터 핸들러 초기화
        
        Args:
            knowledge_base: 지식 베이스 인스턴스
        """
        self.knowledge_base = knowledge_base
        
        # 전용 핸들러들 초기화
        self.handlers = {
            "blind": BlindDataHandler(knowledge_base),
            "job_sites": JobSiteDataHandler(knowledge_base)
        }
        
        # 처리 통계
        self.total_stats = {
            "total_processed": 0,
            "total_failed": 0,
            "processing_sessions": 0,
            "last_processing_time": None
        }
        
        logger.info("통합 데이터 핸들러 초기화 완료")
    
    async def process_provider_data(
        self, 
        provider_name: str, 
        raw_data: List[Dict]
    ) -> ProcessingResult:
        """
        특정 제공자의 데이터 처리
        
        Args:
            provider_name: 데이터 제공자 이름
            raw_data: 원시 데이터 리스트
            
        Returns:
            처리 결과
        """
        if not raw_data:
            return ProcessingResult(
                success=True,
                processed_count=0,
                failed_count=0,
                created_objects=[],
                errors=[],
                processing_time=0.0,
                metadata={"message": "no_data_to_process"}
            )
        
        try:
            # 제공자에 따른 핸들러 선택
            if provider_name in ["blind", "blind_data"]:
                handler = self.handlers["blind"]
            elif provider_name in ["job_sites", "wanted", "jobkorea"]:
                handler = self.handlers["job_sites"]
            else:
                # 알 수 없는 제공자는 기본적으로 Blind 핸들러 사용
                logger.warning(f"알 수 없는 제공자: {provider_name}, Blind 핸들러 사용")
                handler = self.handlers["blind"]
            
            # 데이터 처리
            result = await handler.process_data(raw_data)
            
            # 지식 베이스에 저장
            if result.success and result.created_objects and self.knowledge_base:
                await self._save_to_knowledge_base(result.created_objects, provider_name)
            
            # 전체 통계 업데이트
            self.total_stats["total_processed"] += result.processed_count
            self.total_stats["total_failed"] += result.failed_count
            self.total_stats["processing_sessions"] += 1
            self.total_stats["last_processing_time"] = datetime.now()
            
            return result
            
        except Exception as e:
            logger.error(f"데이터 처리 중 오류: {str(e)}")
            return ProcessingResult(
                success=False,
                processed_count=0,
                failed_count=len(raw_data),
                created_objects=[],
                errors=[str(e)],
                processing_time=0.0,
                metadata={"error": "handler_error"}
            )
    
    async def _save_to_knowledge_base(
        self, 
        objects: List[Any], 
        provider_name: str
    ):
        """
        처리된 객체들을 지식 베이스에 저장
        
        Args:
            objects: 저장할 객체 리스트
            provider_name: 데이터 제공자 이름
        """
        try:
            # 객체 타입별로 분류
            blind_posts = [obj for obj in objects if isinstance(obj, BlindPost)]
            company_reviews = [obj for obj in objects if isinstance(obj, CompanyReview)]
            salary_infos = [obj for obj in objects if isinstance(obj, SalaryInfo)]
            
            # 각 타입별로 적절한 컬렉션에 저장
            save_tasks = []
            
            if blind_posts:
                if "채용" in str(blind_posts[0].category):
                    collection = "career_advice"
                else:
                    collection = "culture_reviews"
                save_tasks.append(
                    self.knowledge_base.add_documents(blind_posts, collection)
                )
            
            if company_reviews:
                save_tasks.append(
                    self.knowledge_base.add_documents(company_reviews, "culture_reviews")
                )
            
            if salary_infos:
                save_tasks.append(
                    self.knowledge_base.add_documents(salary_infos, "salary_discussions")
                )
            
            # 병렬로 저장 실행
            if save_tasks:
                results = await asyncio.gather(*save_tasks, return_exceptions=True)
                
                success_count = sum(1 for result in results if result is True)
                logger.info(
                    f"지식 베이스 저장 완료: {success_count}/{len(save_tasks)} 성공 "
                    f"(제공자: {provider_name})"
                )
            
        except Exception as e:
            logger.error(f"지식 베이스 저장 실패: {str(e)}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """종합 처리 통계 정보 반환"""
        handler_stats = {}
        for name, handler in self.handlers.items():
            handler_stats[name] = handler.get_processing_stats()
        
        return {
            "total_stats": self.total_stats,
            "handler_stats": handler_stats,
            "success_rate": (
                self.total_stats["total_processed"] / 
                max(
                    self.total_stats["total_processed"] + self.total_stats["total_failed"], 
                    1
                )
            )
        }