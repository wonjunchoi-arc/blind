"""
통합 지식 베이스 관리 시스템

BlindInsight AI의 중앙 집중식 지식 관리 시스템으로,
문서 처리, 벡터 임베딩, 검색 기능을 통합하여 제공합니다.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .embeddings import EmbeddingManager, VectorStore, DocumentEmbedding
from .document_processor import DocumentProcessor, ChunkMetadata
from .retriever import RAGRetriever, SearchResult
from ..models.base import settings
from ..models.company import BlindPost, CompanyReview, SalaryInfo

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentIndex:
    """문서 인덱스 정보를 관리하는 데이터 클래스"""
    
    document_id: str              # 문서 고유 식별자
    source_type: str              # 문서 유형 (blind_post, company_review, salary_info)
    company_name: str             # 회사명
    category: str                 # 카테고리 (culture, salary, career, interview)
    file_path: Optional[str]      # 원본 파일 경로
    processed_at: datetime        # 처리 시간
    chunk_count: int              # 청크 개수
    embedding_model: str          # 사용된 임베딩 모델
    metadata: Dict[str, Any]      # 추가 메타데이터
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['processed_at'] = self.processed_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentIndex':
        """딕셔너리에서 객체 생성"""
        data['processed_at'] = datetime.fromisoformat(data['processed_at'])
        return cls(**data)


class QueryEngine:
    """
    쿼리 엔진 클래스
    
    사용자 질의를 처리하고 최적화된 검색 결과를 제공합니다.
    """
    
    def __init__(self, retriever: RAGRetriever):
        """
        쿼리 엔진 초기화
        
        Args:
            retriever: RAG 검색기 인스턴스
        """
        self.retriever = retriever
        
        # 쿼리 최적화를 위한 키워드 매핑
        self.query_keywords = {
            "문화": ["문화", "분위기", "환경", "워라밸", "복지"],
            "연봉": ["연봉", "급여", "월급", "보상", "인센티브", "스톡옵션"],
            "커리어": ["승진", "성장", "커리어", "발전", "기회"],
            "면접": ["면접", "채용", "입사", "지원", "선발"]
        }
    
    def _optimize_query(self, query: str) -> str:
        """
        쿼리 최적화
        
        Args:
            query: 원본 쿼리
            
        Returns:
            최적화된 쿼리
        """
        # 동의어 확장
        optimized_query = query
        for category, keywords in self.query_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    # 관련 키워드들을 쿼리에 추가
                    other_keywords = [k for k in keywords if k != keyword]
                    optimized_query += f" {' '.join(other_keywords[:2])}"
                    break
        
        return optimized_query
    
    def _categorize_query(self, query: str) -> str:
        """
        쿼리 카테고리 분류
        
        Args:
            query: 검색 쿼리
            
        Returns:
            카테고리명
        """
        for category, keywords in self.query_keywords.items():
            if any(keyword in query for keyword in keywords):
                return category
        return "general"
    
    async def search(
        self,
        query: str,
        company_name: Optional[str] = None,
        category: Optional[str] = None,
        k: int = 10,
        use_reranking: bool = True
    ) -> List[SearchResult]:
        """
        통합 검색 수행
        
        Args:
            query: 검색 쿼리
            company_name: 특정 회사로 필터링
            category: 특정 카테고리로 필터링
            k: 반환할 결과 수
            use_reranking: 재랭킹 사용 여부
            
        Returns:
            검색 결과 리스트
        """
        # 쿼리 최적화
        optimized_query = self._optimize_query(query)
        
        # 카테고리 자동 감지
        if not category:
            category = self._categorize_query(query)
        
        # 검색 수행
        results = await self.retriever.search(
            query=optimized_query,
            company_filter=company_name,
            category_filter=category if category != "general" else None,
            k=k,
            use_reranking=use_reranking
        )
        
        return results
    
    async def get_company_summary(
        self, 
        company_name: str,
        aspect: str = "overall"
    ) -> Dict[str, Any]:
        """
        회사 요약 정보 조회
        
        Args:
            company_name: 회사명
            aspect: 분석 관점 (overall, culture, salary, growth, career)
            
        Returns:
            요약 정보
        """
        summary = {
            "company_name": company_name,
            "aspect": aspect,
            "categories": {}
        }
        
        # 카테고리별 대표 문서 조회
        categories = ["culture_reviews", "salary_discussions", "career_advice", "interview_reviews"]
        
        for category in categories:
            results = await self.retriever.search(
                query=f"{company_name} {aspect}",
                company_filter=company_name,
                collection_name=category,
                k=3,
                use_reranking=True
            )
            
            if results:
                summary["categories"][category] = {
                    "count": len(results),
                    "top_documents": [
                        {
                            "content": result.content[:200] + "...",
                            "score": result.score,
                            "metadata": result.metadata
                        }
                        for result in results
                    ]
                }
        
        return summary


class KnowledgeBase:
    """
    중앙 집중식 지식 베이스 관리 클래스
    
    BlindInsight AI의 모든 지식 관리 기능을 통합하여 제공합니다.
    문서 처리, 벡터 임베딩, 검색, 인덱싱을 포함합니다.
    """
    
    def __init__(self, data_dir: str = None):
        """
        지식 베이스 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir or settings.data_directory)
        self.index_file = self.data_dir / "document_index.json"
        
        # 구성 요소 초기화
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore()
        self.document_processor = DocumentProcessor()
        self.retriever = RAGRetriever(self.vector_store)
        self.query_engine = QueryEngine(self.retriever)
        
        # 문서 인덱스 로드
        self.document_index: Dict[str, DocumentIndex] = {}
        self._load_index()
        
        # 성능 메트릭
        self.performance_stats = {
            "documents_processed": 0,
            "total_chunks": 0,
            "search_queries": 0,
            "average_search_time": 0.0,
            "cache_hit_rate": 0.0
        }
    
    def _load_index(self):
        """문서 인덱스 로드"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                
                for doc_id, data in index_data.items():
                    self.document_index[doc_id] = DocumentIndex.from_dict(data)
                
                logger.info(f"{len(self.document_index)}개 문서 인덱스 로드 완료")
                
            except Exception as e:
                logger.error(f"인덱스 로드 실패: {str(e)}")
    
    def _save_index(self):
        """문서 인덱스 저장"""
        try:
            # 디렉토리 생성
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # 인덱스 데이터 준비
            index_data = {
                doc_id: index.to_dict()
                for doc_id, index in self.document_index.items()
            }
            
            # JSON 파일로 저장
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"문서 인덱스 저장 완료: {len(self.document_index)}개 문서")
            
        except Exception as e:
            logger.error(f"인덱스 저장 실패: {str(e)}")
    
    async def add_documents(
        self,
        documents: List[Union[BlindPost, CompanyReview, SalaryInfo]],
        collection_name: str = None
    ) -> bool:
        """
        문서들을 지식 베이스에 추가
        
        Args:
            documents: 추가할 문서 리스트
            collection_name: 저장할 컬렉션명
            
        Returns:
            성공 여부
        """
        try:
            logger.info(f"{len(documents)}개 문서 처리 시작...")
            
            # 문서 처리 및 청킹
            all_chunks = []
            for doc in documents:
                # 문서 유형에 따라 처리
                if isinstance(doc, BlindPost):
                    chunks = await self.document_processor.process_blind_post(doc)
                elif isinstance(doc, CompanyReview):
                    chunks = await self.document_processor.process_company_review(doc)
                elif isinstance(doc, SalaryInfo):
                    chunks = await self.document_processor.process_salary_info(doc)
                else:
                    logger.warning(f"지원하지 않는 문서 유형: {type(doc)}")
                    continue
                
                all_chunks.extend(chunks)
                
                # 문서 인덱스 업데이트
                doc_id = getattr(doc, 'id', str(hash(str(doc))))
                self.document_index[doc_id] = DocumentIndex(
                    document_id=doc_id,
                    source_type=type(doc).__name__.lower(),
                    company_name=getattr(doc, 'company_name', 'unknown'),
                    category=getattr(doc, 'category', 'general'),
                    file_path=None,
                    processed_at=datetime.now(),
                    chunk_count=len(chunks),
                    embedding_model=self.embedding_manager.embedding_model.model,
                    metadata={"original_doc": str(doc)[:100]}
                )
            
            logger.info(f"총 {len(all_chunks)}개 청크 생성됨")
            
            # 임베딩 생성 및 벡터 저장소에 추가
            document_embeddings = []
            for chunk in all_chunks:
                # DocumentEmbedding 객체 생성
                doc_embedding = self.embedding_manager.create_document_embedding(
                    content=chunk.content,
                    metadata=chunk.metadata.__dict__
                )
                
                # 임베딩 생성
                doc_embedding.embedding = await self.embedding_manager.create_embedding(
                    chunk.content
                )
                
                document_embeddings.append(doc_embedding)
            
            # 컬렉션명 결정
            if not collection_name:
                collection_name = "company_general"
            
            # 벡터 저장소에 추가
            success = await self.vector_store.add_documents(
                document_embeddings, 
                collection_name
            )
            
            if success:
                # 성능 통계 업데이트
                self.performance_stats["documents_processed"] += len(documents)
                self.performance_stats["total_chunks"] += len(all_chunks)
                
                # 인덱스 저장
                self._save_index()
                
                logger.info(f"문서 추가 완료: {len(documents)}개 문서, {len(all_chunks)}개 청크")
                return True
            else:
                logger.error("벡터 저장소 추가 실패")
                return False
                
        except Exception as e:
            logger.error(f"문서 추가 중 오류 발생: {str(e)}")
            return False
    
    async def search(
        self,
        query: str,
        company_name: Optional[str] = None,
        category: Optional[str] = None,
        k: int = 10,
        use_reranking: bool = True
    ) -> List[SearchResult]:
        """
        지식 베이스 검색
        
        Args:
            query: 검색 쿼리
            company_name: 회사명 필터
            category: 카테고리 필터
            k: 반환할 결과 수
            use_reranking: 재랭킹 사용 여부
            
        Returns:
            검색 결과 리스트
        """
        import time
        
        start_time = time.time()
        
        try:
            # 쿼리 엔진을 통한 검색
            results = await self.query_engine.search(
                query=query,
                company_name=company_name,
                category=category,
                k=k,
                use_reranking=use_reranking
            )
            
            # 검색 시간 측정
            search_time = time.time() - start_time
            
            # 성능 통계 업데이트
            self.performance_stats["search_queries"] += 1
            current_avg = self.performance_stats["average_search_time"]
            query_count = self.performance_stats["search_queries"]
            self.performance_stats["average_search_time"] = (
                (current_avg * (query_count - 1) + search_time) / query_count
            )
            
            logger.info(f"검색 완료: {len(results)}개 결과, {search_time:.3f}초")
            return results
            
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {str(e)}")
            return []
    
    async def get_company_analysis(
        self, 
        company_name: str, 
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        회사 종합 분석 수행
        
        Args:
            company_name: 분석할 회사명
            analysis_type: 분석 유형 (comprehensive, culture, salary, growth, career)
            
        Returns:
            분석 결과
        """
        try:
            analysis_result = {
                "company_name": company_name,
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat(),
                "categories": {}
            }
            
            # 분석 유형에 따른 카테고리 설정
            if analysis_type == "comprehensive":
                categories = ["culture_reviews", "salary_discussions", "career_advice", "interview_reviews"]
            elif analysis_type == "culture":
                categories = ["culture_reviews"]
            elif analysis_type == "salary":
                categories = ["salary_discussions"]
            elif analysis_type == "growth":
                categories = ["career_advice"]
            elif analysis_type == "career":
                categories = ["career_advice", "interview_reviews"]
            else:
                categories = ["company_general"]
            
            # 각 카테고리별 분석
            for category in categories:
                # 카테고리별 검색
                search_results = await self.retriever.search(
                    query=f"{company_name}",
                    company_filter=company_name,
                    collection_name=category,
                    k=20,
                    use_reranking=True
                )
                
                if search_results:
                    # 카테고리 분석 결과
                    category_analysis = {
                        "document_count": len(search_results),
                        "average_score": sum(r.score for r in search_results) / len(search_results),
                        "key_insights": [
                            {
                                "content": result.content[:300] + "...",
                                "score": result.score,
                                "metadata": result.metadata
                            }
                            for result in search_results[:5]
                        ],
                        "sentiment_distribution": self._analyze_sentiment(search_results),
                        "common_keywords": self._extract_keywords(search_results)
                    }
                    
                    analysis_result["categories"][category] = category_analysis
            
            logger.info(f"회사 분석 완료: {company_name} ({analysis_type})")
            return analysis_result
            
        except Exception as e:
            logger.error(f"회사 분석 중 오류 발생: {str(e)}")
            return {}
    
    def _analyze_sentiment(self, search_results: List[SearchResult]) -> Dict[str, int]:
        """
        검색 결과의 감정 분석
        
        Args:
            search_results: 검색 결과 리스트
            
        Returns:
            감정 분포
        """
        # 간단한 키워드 기반 감정 분석
        positive_keywords = ["좋다", "만족", "추천", "훌륭", "우수", "성장", "기회"]
        negative_keywords = ["나쁘다", "불만", "힘들다", "스트레스", "과로", "문제"]
        
        sentiment_count = {"positive": 0, "negative": 0, "neutral": 0}
        
        for result in search_results:
            content = result.content.lower()
            positive_score = sum(1 for keyword in positive_keywords if keyword in content)
            negative_score = sum(1 for keyword in negative_keywords if keyword in content)
            
            if positive_score > negative_score:
                sentiment_count["positive"] += 1
            elif negative_score > positive_score:
                sentiment_count["negative"] += 1
            else:
                sentiment_count["neutral"] += 1
        
        return sentiment_count
    
    def _extract_keywords(self, search_results: List[SearchResult]) -> List[str]:
        """
        검색 결과에서 주요 키워드 추출
        
        Args:
            search_results: 검색 결과 리스트
            
        Returns:
            주요 키워드 리스트
        """
        # 간단한 빈도 기반 키워드 추출
        all_text = " ".join(result.content for result in search_results)
        
        # 주요 키워드 후보
        important_keywords = [
            "워라밸", "야근", "복지", "연봉", "승진", "성장", "분위기", 
            "동료", "상사", "프로젝트", "개발", "기술", "교육", "문화"
        ]
        
        keyword_count = {}
        for keyword in important_keywords:
            count = all_text.lower().count(keyword)
            if count > 0:
                keyword_count[keyword] = count
        
        # 빈도순으로 정렬하여 상위 10개 반환
        sorted_keywords = sorted(keyword_count.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, count in sorted_keywords[:10]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        지식 베이스 통계 정보 반환
        
        Returns:
            통계 정보
        """
        # 컬렉션별 통계
        collection_stats = {}
        for collection_name in ["culture_reviews", "salary_discussions", "career_advice", "interview_reviews", "company_general"]:
            stats = self.vector_store.get_collection_stats(collection_name)
            if stats:
                collection_stats[collection_name] = stats
        
        # 전체 통계
        total_documents = sum(stats.get("document_count", 0) for stats in collection_stats.values())
        
        # 임베딩 캐시 통계
        cache_stats = self.embedding_manager.get_cache_stats()
        
        return {
            "knowledge_base": {
                "total_documents": total_documents,
                "total_indexed": len(self.document_index),
                "collections": collection_stats
            },
            "performance": self.performance_stats,
            "embedding_cache": cache_stats,
            "system_info": {
                "embedding_model": self.embedding_manager.embedding_model.model,
                "vector_db_path": str(self.vector_store.persist_directory),
                "data_directory": str(self.data_dir)
            }
        }
    
    async def backup_knowledge_base(self, backup_path: str) -> bool:
        """
        지식 베이스 백업
        
        Args:
            backup_path: 백업 저장 경로
            
        Returns:
            성공 여부
        """
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 각 컬렉션 백업
            success_count = 0
            for collection_name in ["culture_reviews", "salary_discussions", "career_advice", "interview_reviews", "company_general"]:
                backup_file = backup_dir / f"{collection_name}_backup.json"
                if self.vector_store.backup_collection(collection_name, str(backup_file)):
                    success_count += 1
            
            # 문서 인덱스 백업
            index_backup = backup_dir / "document_index_backup.json"
            with open(index_backup, 'w', encoding='utf-8') as f:
                index_data = {
                    doc_id: index.to_dict()
                    for doc_id, index in self.document_index.items()
                }
                json.dump(index_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"지식 베이스 백업 완료: {success_count}개 컬렉션, 백업 경로: {backup_path}")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"백업 중 오류 발생: {str(e)}")
            return False
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            # 인덱스 저장
            self._save_index()
            
            # 캐시 정리
            self.embedding_manager._embedding_cache.clear()
            
            logger.info("지식 베이스 정리 완료")
            
        except Exception as e:
            logger.error(f"정리 중 오류 발생: {str(e)}")