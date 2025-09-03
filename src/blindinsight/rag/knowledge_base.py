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
from .json_processor import ChunkDataProcessor, ChunkDataLoader
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
        categories = ["company_culture", "work_life_balance", "management", "salary_benefits", "career_growth", "general"]
        
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
    
    def __init__(self, data_dir: str = None, vector_db_path: str = None):
        """
        지식 베이스 초기화
        
        Args:
            data_dir: 데이터 저장 디렉토리
            vector_db_path: 벡터 데이터베이스 저장 경로
        """
        self.data_dir = Path(data_dir or settings.data_directory)
        self.vector_db_path = vector_db_path or settings.vector_db_path
        self.index_file = self.data_dir / "document_index.json"
        
        # 구성 요소 초기화
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStore(persist_directory=self.vector_db_path)
        self.document_processor = DocumentProcessor()
        self.chunk_processor = ChunkDataProcessor()
        self.retriever = RAGRetriever(self.vector_store, keyword_threshold=0.005)
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
    
    async def initialize(self):
        """
        지식베이스 초기화
        
        벡터 스토어와 모든 구성 요소를 초기화합니다.
        """
        try:
            logger.info("지식베이스 초기화 중...")
            
            # 디렉토리 생성
            os.makedirs(self.vector_db_path, exist_ok=True)
            os.makedirs(self.data_dir, exist_ok=True)
            
            # 벡터 스토어 및 컬렉션 초기화 확인
            if hasattr(self.vector_store, '_collections'):
                logger.info(f"벡터 스토어 컬렉션 수: {len(self.vector_store._collections)}")
            
            # 통계 확인
            stats = self.get_statistics()
            total_docs = stats.get("knowledge_base", {}).get("total_documents", 0)
            logger.info(f"지식베이스 초기화 완료 - 총 문서 수: {total_docs}")
            
        except Exception as e:
            logger.error(f"지식베이스 초기화 실패: {str(e)}")
            raise
    
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
            
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {str(e)}")
            return []
    
    async def load_chunk_data(
        self, 
        data_dir: str = "tools/data",
        company_filter: Optional[List[str]] = None
    ) -> bool:
        """
        JSON 청크 데이터를 지식 베이스에 로드
        
        Args:
            data_dir: 청크 JSON 파일들이 있는 디렉토리
            company_filter: 로드할 회사 리스트 (None이면 모든 회사)
            
        Returns:
            성공 여부
        """
        try:
            logger.info(f"JSON 청크 데이터 로드 시작: {data_dir}")
            
            # ChunkDataLoader를 통해 데이터 로드
            chunk_loader = ChunkDataLoader(data_dir)
            success = await chunk_loader.load_all_chunks(company_filter)
            
            if success:
                # 통계 업데이트
                stats = chunk_loader.get_stats()
                self.performance_stats["documents_processed"] += stats["documents_created"]
                self.performance_stats["total_chunks"] += stats["documents_created"]
                
                # 인덱스에 추가 (간단화된 버전)
                for company in stats["companies_processed"]:
                    doc_id = f"chunks_{company}_{datetime.now().strftime('%Y%m%d')}"
                    self.document_index[doc_id] = DocumentIndex(
                        document_id=doc_id,
                        source_type="json_chunks",
                        company_name=company,
                        category="chunks",
                        file_path=f"{data_dir}/hybrid_vectordb/{company}_chunk_first_vectordb.json",
                        processed_at=datetime.now(),
                        chunk_count=stats["documents_created"] // len(stats["companies_processed"]),
                        embedding_model="text-embedding-3-small",
                        metadata={"loader_stats": stats}
                    )
                
                # 인덱스 저장
                self._save_index()
                
                logger.info(f"청크 데이터 로드 완료: {stats}")
                return True
            else:
                logger.error("청크 데이터 로드 실패")
                return False
                
        except Exception as e:
            logger.error(f"청크 데이터 로드 중 오류: {str(e)}")
            return False
    
    async def search_reviews(
        self,
        query: str,
        company_name: Optional[str] = None,
        review_aspects: Optional[List[str]] = None,
        k: int = 10
    ) -> List[SearchResult]:
        """
        리뷰 데이터 전용 검색
        
        Args:
            query: 검색 쿼리
            company_name: 특정 회사로 필터링
            review_aspects: 검색할 리뷰 측면 (culture, salary, growth 등)
            k: 반환할 결과 수
            
        Returns:
            검색 결과 리스트
        """
        try:
            # 검색 필터 구성
            filters = {}
            if company_name:
                filters["company"] = company_name
            
            # 리뷰 측면에 따른 컬렉션 선택 (새로운 6개 카테고리 체계 적용)
            collections_to_search = []
            if review_aspects:
                collection_mapping = {
                    "culture": "company_culture",
                    "company_culture": "company_culture",
                    "work_life_balance": "work_life_balance",
                    "management": "management",
                    "salary": "salary_benefits",
                    "salary_benefits": "salary_benefits",
                    "benefits": "salary_benefits", 
                    "growth": "career_growth",
                    "career_growth": "career_growth",
                    "career": "career_growth",
                    "general": "general"
                }
                for aspect in review_aspects:
                    if aspect in collection_mapping:
                        collections_to_search.append(collection_mapping[aspect])
            else:
                collections_to_search = [
                    "company_culture", 
                    "work_life_balance",
                    "management", 
                    "salary_benefits", 
                    "career_growth", 
                    "general"
                ]
            
            # 각 컬렉션에서 검색 수행
            all_results = []
            for collection in collections_to_search:
                try:
                    results = await self.retriever.search(
                        query=query,
                        collection_name=collection,
                        k=k // len(collections_to_search) + 2,  # 컬렉션당 결과 수 조정
                        filters=filters,
                        search_type="ensemble"
                    )
                    all_results.extend(results)
                except Exception as e:
                    logger.warning(f"컬렉션 {collection} 검색 실패: {str(e)}")
                    continue
            
            # 중복 제거 및 상위 k개 반환
            unique_results = []
            seen_contents = set()
            
            for result in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
                content_hash = hash(result.document.page_content[:100])  # 내용의 처음 100자로 중복 체크
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    unique_results.append(result)
                    
                    if len(unique_results) >= k:
                        break
            
            return unique_results
            
        except Exception as e:
            logger.error(f"리뷰 검색 중 오류: {str(e)}")
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
                categories = ["company_culture", "work_life_balance", "management", "salary_benefits", "career_growth", "general"]
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
    
    async def get_available_companies(self) -> List[str]:
        """
        SQLite 메타데이터 데이터베이스에서 사용 가능한 회사 목록 조회
        
        Returns:
            회사명 리스트
        """
        try:
            # SQLite 메타데이터 DB에서 직접 조회 (빠르고 효율적)
            companies = self.vector_store.get_companies_from_metadata()
            logger.info(f"SQLite DB에서 회사 {len(companies)}개 조회됨")
            return companies
            
        except Exception as e:
            logger.error(f"SQLite DB 회사 목록 조회 실패: {str(e)}")
            return []
    
    async def get_available_positions(self, company_name: Optional[str] = None) -> List[str]:
        """
        SQLite 메타데이터 데이터베이스에서 직무 목록 조회 (회사별 필터링)
        
        Args:
            company_name: 특정 회사로 필터링 (None이면 모든 회사의 직무)
            
        Returns:
            직무 리스트
        """
        try:
            # SQLite 메타데이터 DB에서 직접 조회 (빠르고 효율적)
            positions = self.vector_store.get_positions_from_metadata(company_name)
            
            if company_name:
                logger.info(f"SQLite DB에서 {company_name} 직무 {len(positions)}개 조회됨")
            else:
                logger.info(f"SQLite DB에서 전체 직무 {len(positions)}개 조회됨")
                
            return positions
            
        except Exception as e:
            logger.error(f"SQLite DB 직무 목록 조회 실패: {str(e)}")
            return []
    
    async def get_available_years(self, company_name: Optional[str] = None) -> List[str]:
        """
        SQLite 메타데이터 데이터베이스에서 연도 목록 조회 (회사별 필터링)
        
        Args:
            company_name: 특정 회사로 필터링 (None이면 모든 회사의 연도)
            
        Returns:
            연도 리스트 (문자열, 내림차순 정렬)
        """
        try:
            # SQLite 메타데이터 DB에서 직접 조회 (빠르고 효율적)
            year_integers = self.vector_store.get_years_from_metadata(company_name)
            # 정수를 문자열로 변환
            years = [str(year) for year in year_integers]
            
            if company_name:
                logger.info(f"SQLite DB에서 {company_name} 연도 {len(years)}개 조회됨")
            else:
                logger.info(f"SQLite DB에서 전체 연도 {len(years)}개 조회됨")
                
            return years
            
        except Exception as e:
            logger.error(f"SQLite DB 연도 목록 조회 실패: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        지식 베이스 통계 정보 반환
        
        Returns:
            통계 정보
        """
        # 컬렉션별 통계
        collection_stats = {}
        for collection_name in ["company_culture", "work_life_balance", "management", "salary_benefits", "career_growth", "general"]:
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
            for collection_name in ["company_culture", "work_life_balance", "management", "salary_benefits", "career_growth", "general"]:
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