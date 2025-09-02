"""
문서 처리 및 청킹 시스템

블라인드 게시글과 리뷰 데이터를 효과적으로 처리하고
검색에 최적화된 청크로 분할하는 시스템입니다.
"""

import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ..models.base import BaseModel, settings
from ..models.company import BlindPost, CompanyReview, SalaryInfo


@dataclass
class ChunkMetadata:
    """청크 메타데이터를 담는 데이터 클래스"""
    
    source_id: str                # 원본 문서 ID
    chunk_index: int              # 청크 순서 번호
    chunk_type: str               # 청크 타입 (title, content, summary)
    company: str                  # 회사명
    category: str                 # 카테고리 (culture, salary, career)
    created_at: datetime          # 생성 시간
    language: str = "ko"          # 언어 (기본: 한국어)
    word_count: int = 0           # 단어 수
    char_count: int = 0           # 글자 수
    sentiment: Optional[float] = None  # 감정 점수 (-1 ~ 1)


class TextChunker:
    """
    텍스트를 검색에 최적화된 청크로 분할하는 클래스
    
    한국어 특성을 고려한 문장 단위 분할과
    의미적 일관성을 유지하는 청킹을 제공합니다.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        텍스트 청커 초기화
        
        Args:
            chunk_size: 청크 최대 크기 (문자 단위)
            chunk_overlap: 청크 간 중복 크기
            separators: 분할 기준 문자들
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 한국어에 최적화된 분할 기준
        self.separators = separators or [
            "\n\n",      # 문단 구분
            "\n",        # 줄바꿈
            "。",        # 한국어 문장 끝
            ".",         # 영어 문장 끝
            "!",         # 감탄문
            "?",         # 의문문
            ";",         # 세미콜론
            ",",         # 쉼표
            " ",         # 공백
            ""           # 문자 단위
        ]
        
        # LangChain의 RecursiveCharacterTextSplitter 사용
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len,  # 문자 수 기준
            is_separator_regex=False
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, ChunkMetadata]]:
        """
        텍스트를 청크로 분할
        
        Args:
            text: 분할할 텍스트
            metadata: 청크에 포함할 메타데이터
            
        Returns:
            (청크_텍스트, 청크_메타데이터) 튜플 리스트
        """
        if not text or not text.strip():
            return []
        
        # 텍스트 전처리
        cleaned_text = self._preprocess_text(text)
        
        # 텍스트 분할
        chunks = self.text_splitter.split_text(cleaned_text)
        
        # 청크 메타데이터 생성
        chunk_list = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = ChunkMetadata(
                source_id=metadata.get("source_id", "unknown"),
                chunk_index=i,
                chunk_type="content",
                company=metadata.get("company", ""),
                category=metadata.get("category", "general"),
                created_at=datetime.now(),
                word_count=len(chunk.split()),
                char_count=len(chunk),
                sentiment=self._analyze_sentiment(chunk)
            )
            
            chunk_list.append((chunk, chunk_metadata))
        
        return chunk_list
    
    def _preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리 (정리 및 정규화)
        
        Args:
            text: 원본 텍스트
            
        Returns:
            전처리된 텍스트
        """
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        # 불필요한 특수문자 제거
        text = re.sub(r'[^\w\s\.\!\?\,\;\:\(\)\[\]\-]', '', text)
        
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        간단한 감정 분석 (키워드 기반)
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            감정 점수 (-1: 부정적, 0: 중립, 1: 긍정적)
        """
        # 긍정적 키워드
        positive_keywords = [
            "좋은", "훌륭한", "만족", "추천", "성장", "기회", "복지",
            "자유로운", "수평적", "혁신적", "도전적", "소통", "배려"
        ]
        
        # 부정적 키워드
        negative_keywords = [
            "나쁜", "최악", "불만", "스트레스", "야근", "수직적", "경직된",
            "갑질", "차별", "과로", "압박", "번아웃", "힘들다"
        ]
        
        # 키워드 카운트
        positive_count = sum(1 for keyword in positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in negative_keywords if keyword in text)
        
        # 감정 점수 계산
        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return 0.0  # 중립
        
        return (positive_count - negative_count) / total_keywords


class DocumentProcessor:
    """
    다양한 형태의 문서를 처리하여 검색 가능한 형태로 변환하는 클래스
    
    블라인드 포스트, 회사 리뷰, 연봉 정보 등을 처리하여
    구조화된 문서 형태로 변환합니다.
    """
    
    def __init__(self, chunker: Optional[TextChunker] = None):
        """
        문서 처리기 초기화
        
        Args:
            chunker: 텍스트 청킹을 위한 TextChunker 인스턴스
        """
        self.chunker = chunker or TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
    
    def process_blind_post(self, post: BlindPost) -> List[Document]:
        """
        블라인드 게시글을 처리하여 Document 객체들로 변환
        
        Args:
            post: 블라인드 게시글 객체
            
        Returns:
            처리된 Document 객체 리스트
        """
        documents = []
        
        # 기본 메타데이터 구성
        base_metadata = {
            "source_id": post.post_id,
            "source_type": "blind_post",
            "company": post.company,
            "category": post.category,
            "department": post.department,
            "position": post.position,
            "experience_years": post.experience_years,
            "created_at": post.created_at.isoformat(),
            "sentiment_score": post.sentiment_score,
            "credibility_score": post.credibility_score,
            "tags": post.tags
        }
        
        # 제목과 내용을 결합한 전체 텍스트
        full_content = f"제목: {post.title}\n\n내용: {post.content}"
        
        # 텍스트 청킹
        chunks = self.chunker.chunk_text(full_content, base_metadata)
        
        # Document 객체들 생성
        for chunk_text, chunk_metadata in chunks:
            # 메타데이터 병합
            metadata = {**base_metadata}
            metadata.update({
                "chunk_index": chunk_metadata.chunk_index,
                "chunk_type": chunk_metadata.chunk_type,
                "word_count": chunk_metadata.word_count,
                "char_count": chunk_metadata.char_count,
                "chunk_sentiment": chunk_metadata.sentiment
            })
            
            document = Document(
                page_content=chunk_text,
                metadata=metadata
            )
            documents.append(document)
        
        return documents
    
    def process_company_review(self, review: CompanyReview) -> List[Document]:
        """
        회사 리뷰를 처리하여 Document 객체들로 변환
        
        Args:
            review: 회사 리뷰 객체
            
        Returns:
            처리된 Document 객체 리스트
        """
        documents = []
        
        # 기본 메타데이터 구성 (새로운 카테고리 체계 적용)
        base_metadata = {
            "source_id": review.review_id,
            "source_type": "company_review",
            "company": review.company,
            "category": "company_culture",  # 새로운 카테고리 체계 적용
            "overall_rating": review.overall_rating,
            "work_life_balance": review.work_life_balance,
            "culture_values": review.culture_values,
            "career_opportunities": review.career_opportunities,
            "compensation_benefits": review.compensation_benefits,
            "senior_management": review.senior_management,
            "employment_status": review.employment_status,
            "position_category": review.position_category,
            "tenure_months": review.tenure_months,
            "created_at": review.created_at.isoformat(),
            "verified": review.verified
        }
        
        # 장점 섹션 처리
        if review.pros:
            pros_content = f"장점: {review.pros}"
            pros_chunks = self.chunker.chunk_text(pros_content, {
                **base_metadata,
                "review_section": "pros"
            })
            
            for chunk_text, chunk_metadata in pros_chunks:
                metadata = {**base_metadata}
                metadata.update({
                    "review_section": "pros",
                    "chunk_index": chunk_metadata.chunk_index,
                    "chunk_sentiment": chunk_metadata.sentiment
                })
                
                documents.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))
        
        # 단점 섹션 처리
        if review.cons:
            cons_content = f"단점: {review.cons}"
            cons_chunks = self.chunker.chunk_text(cons_content, {
                **base_metadata,
                "review_section": "cons"
            })
            
            for chunk_text, chunk_metadata in cons_chunks:
                metadata = {**base_metadata}
                metadata.update({
                    "review_section": "cons",
                    "chunk_index": chunk_metadata.chunk_index,
                    "chunk_sentiment": chunk_metadata.sentiment
                })
                
                documents.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))
        
        # 경영진 리뷰 처리 (있는 경우)
        if review.management_review:
            mgmt_content = f"경영진 평가: {review.management_review}"
            mgmt_chunks = self.chunker.chunk_text(mgmt_content, {
                **base_metadata,
                "review_section": "management"
            })
            
            for chunk_text, chunk_metadata in mgmt_chunks:
                metadata = {**base_metadata}
                metadata.update({
                    "review_section": "management",
                    "chunk_index": chunk_metadata.chunk_index,
                    "chunk_sentiment": chunk_metadata.sentiment
                })
                
                documents.append(Document(
                    page_content=chunk_text,
                    metadata=metadata
                ))
        
        return documents
    
    def process_salary_info(self, salary: SalaryInfo) -> List[Document]:
        """
        연봉 정보를 처리하여 Document 객체들로 변환
        
        Args:
            salary: 연봉 정보 객체
            
        Returns:
            처리된 Document 객체 리스트
        """
        # 연봉 정보를 텍스트 형태로 구성
        content_parts = [
            f"회사: {salary.company}",
            f"직무: {salary.position}",
            f"레벨: {salary.level or '미지정'}",
            f"부서: {salary.department or '미지정'}",
            f"기본연봉: {salary.base_salary}만원",
            f"총 연봉: {salary.total_compensation}만원",
            f"경력: {salary.experience_years}년",
            f"근무지: {salary.location}",
            f"보고일: {salary.salary_date.strftime('%Y년 %m월')}"
        ]
        
        if salary.bonus:
            content_parts.append(f"보너스: {salary.bonus}만원")
        
        if salary.stock_options:
            content_parts.append(f"스톡옵션: {salary.stock_options}")
        
        if salary.other_benefits:
            content_parts.append(f"기타혜택: {', '.join(salary.other_benefits)}")
        
        full_content = "\n".join(content_parts)
        
        # 메타데이터 구성 (새로운 카테고리 체계 적용)
        metadata = {
            "source_id": salary.salary_id,
            "source_type": "salary_info",
            "company": salary.company,
            "category": "salary_benefits",  # 새로운 카테고리 체계 적용
            "position": salary.position,
            "level": salary.level,
            "department": salary.department,
            "base_salary": salary.base_salary,
            "total_compensation": salary.total_compensation,
            "bonus": salary.bonus,
            "experience_years": salary.experience_years,
            "education_level": salary.education_level,
            "location": salary.location,
            "salary_date": salary.salary_date.isoformat(),
            "reported_at": salary.reported_at.isoformat(),
            "verified": salary.verified,
            "currency": salary.currency
        }
        
        # 단일 문서로 반환 (연봉 정보는 보통 짧아서 청킹이 불필요)
        return [Document(
            page_content=full_content,
            metadata=metadata
        )]


class DocumentLoader:
    """
    다양한 소스에서 문서를 로드하는 클래스
    
    JSON 파일, CSV 파일, 데이터베이스 등에서
    문서를 로드하여 처리 가능한 형태로 변환합니다.
    """
    
    def __init__(self, processor: Optional[DocumentProcessor] = None):
        """
        문서 로더 초기화
        
        Args:
            processor: 문서 처리를 위한 DocumentProcessor 인스턴스
        """
        self.processor = processor or DocumentProcessor()
    
    def load_from_json_file(self, file_path: str) -> List[Document]:
        """
        JSON 파일에서 문서들을 로드
        
        Args:
            file_path: JSON 파일 경로
            
        Returns:
            로드된 Document 객체 리스트
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            # 데이터 타입에 따른 처리
            if isinstance(data, list):
                for item in data:
                    docs = self._process_json_item(item)
                    documents.extend(docs)
            elif isinstance(data, dict):
                docs = self._process_json_item(data)
                documents.extend(docs)
            
            print(f"{file_path}에서 {len(documents)}개 문서 로드 완료")
            return documents
            
        except Exception as e:
            print(f"JSON 파일 로드 실패 ({file_path}): {str(e)}")
            return []
    
    def _process_json_item(self, item: Dict[str, Any]) -> List[Document]:
        """JSON 아이템을 문서로 변환"""
        documents = []
        
        try:
            # 데이터 타입 판별 및 처리
            source_type = item.get("type", "unknown")
            
            if source_type == "blind_post" or "post_id" in item:
                # 블라인드 포스트로 처리
                post = BlindPost(**item)
                documents = self.processor.process_blind_post(post)
                
            elif source_type == "company_review" or "review_id" in item:
                # 회사 리뷰로 처리
                review = CompanyReview(**item)
                documents = self.processor.process_company_review(review)
                
            elif source_type == "salary_info" or "salary_id" in item:
                # 연봉 정보로 처리
                salary = SalaryInfo(**item)
                documents = self.processor.process_salary_info(salary)
                
            else:
                # 일반 문서로 처리
                content = item.get("content", "")
                if content:
                    metadata = {k: v for k, v in item.items() if k != "content"}
                    documents = [Document(page_content=content, metadata=metadata)]
        
        except Exception as e:
            print(f"JSON 아이템 처리 실패: {str(e)}")
        
        return documents
    
    def load_mock_data(self) -> List[Document]:
        """
        테스트용 목업 데이터 생성
        
        Returns:
            생성된 목업 Document 객체 리스트
        """
        mock_documents = []
        
        # 목업 블라인드 포스트들
        mock_posts = [
            BlindPost(
                post_id="mock_post_001",
                company="네이버",
                title="네이버 개발자 워라밸 후기",
                content="네이버에서 2년째 근무 중인 개발자입니다. 전반적으로 워라밸이 좋은 편이고, 자유로운 분위기에서 일할 수 있어 만족합니다. 기술 스택도 다양하고 성장할 기회가 많아요.",
                category="culture",
                department="개발",
                position="소프트웨어 엔지니어",
                experience_years=2,
                sentiment_score=0.7,
                credibility_score=0.8,
                tags=["워라밸", "개발자", "성장기회"],
                created_at=datetime.now()
            ),
            BlindPost(
                post_id="mock_post_002", 
                company="카카오",
                title="카카오 연봉 정보 공유",
                content="카카오 3년차 백엔드 개발자 연봉 정보 공유합니다. 기본연봉 7500만원, 보너스 포함 총 9000만원 정도 받고 있어요. 복지도 괜찮은 편입니다.",
                category="salary",
                department="개발",
                position="백엔드 개발자",
                experience_years=3,
                sentiment_score=0.5,
                credibility_score=0.9,
                tags=["연봉", "백엔드", "복지"],
                created_at=datetime.now()
            )
        ]
        
        # 목업 회사 리뷰들
        mock_reviews = [
            CompanyReview(
                review_id="mock_review_001",
                company="토스",
                overall_rating=4.2,
                work_life_balance=4,
                culture_values=5,
                career_opportunities=4,
                compensation_benefits=4,
                senior_management=3,
                pros="혁신적인 문화, 빠른 의사결정, 좋은 동료들",
                cons="빠른 변화로 인한 스트레스, 높은 성과 기대치",
                employment_status="current",
                position_category="개발",
                tenure_months=18,
                created_at=datetime.now()
            )
        ]
        
        # 목업 연봉 정보들
        mock_salaries = [
            SalaryInfo(
                salary_id="mock_salary_001",
                company="네이버",
                position="AI 엔지니어",
                level="시니어",
                base_salary=8500,
                total_compensation=11000,
                bonus=2000,
                experience_years=5,
                location="서울",
                salary_date=datetime.now(),
                reported_at=datetime.now()
            )
        ]
        
        # 각 타입별로 문서 처리
        for post in mock_posts:
            documents = self.processor.process_blind_post(post)
            mock_documents.extend(documents)
        
        for review in mock_reviews:
            documents = self.processor.process_company_review(review)
            mock_documents.extend(documents)
        
        for salary in mock_salaries:
            documents = self.processor.process_salary_info(salary)
            mock_documents.extend(documents)
        
        print(f"목업 데이터 생성 완료: {len(mock_documents)}개 문서")
        return mock_documents