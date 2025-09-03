"""
RAG 검색 및 재랭킹 시스템

고품질 검색 결과를 위한 다단계 검색과 재랭킹을 제공하는 시스템입니다.
의미 기반 검색과 키워드 검색을 결합하여 최적의 결과를 반환합니다.
"""

import asyncio
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from .embeddings import VectorStore
from ..models.base import BaseModel, settings


@dataclass
class SearchResult:
    """검색 결과를 담는 데이터 클래스"""
    
    document: Document           # 검색된 문서
    relevance_score: float       # 관련성 점수 (0-1)
    rank: int                   # 순위
    retrieval_method: str       # 검색 방법 (semantic, keyword, hybrid)
    metadata_scores: Dict[str, float] = None  # 메타데이터 기반 점수들


class ReRanker:
    """
    검색 결과를 재정렬하는 클래스
    
    Cross-encoder 모델을 사용하여 쿼리와 문서 간의
    관련성을 더 정확하게 평가하고 순위를 조정합니다.
    싱글톤 패턴을 사용하여 모델의 중복 로드를 방지합니다.
    """
    
    _instance = None
    _model = None
    _model_loaded = False
    _model_name = None
    
    def __new__(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """싱글톤 패턴 적용"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        재랭커 초기화 - 싱글톤으로 중복 로드 방지
        
        Args:
            model_name: 사용할 Cross-encoder 모델명
        """
        # 이미 같은 모델이 로드되어 있으면 건너뛰기
        if self.__class__._model_loaded and self.__class__._model_name == model_name:
            self.model = self.__class__._model
            self.model_loaded = self.__class__._model_loaded
            return
        
        try:
            # Cross-encoder 모델 로드
            self.model = CrossEncoder(model_name)
            self.model_loaded = True
            
            # 클래스 변수에 저장
            self.__class__._model = self.model
            self.__class__._model_loaded = True
            self.__class__._model_name = model_name
            
            print(f"재랭킹 모델 로드 완료: {model_name}")
        except Exception as e:
            print(f"재랭킹 모델 로드 실패: {str(e)}")
            self.model = None
            self.model_loaded = False
            
            # 클래스 변수 초기화
            self.__class__._model = None
            self.__class__._model_loaded = False
            self.__class__._model_name = None
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        문서들을 재랭킹하여 관련성 순으로 정렬
        
        Args:
            query: 검색 쿼리
            documents: 재랭킹할 문서 리스트
            top_k: 반환할 상위 문서 수
            
        Returns:
            (문서, 재랭킹_점수) 튜플 리스트
        """
        if not self.model_loaded or not documents:
            # 모델이 없으면 원본 순서 반환
            return [(doc, 0.5) for doc in documents[:top_k]]
        
        try:
            # 쿼리-문서 쌍 생성
            query_doc_pairs = [(query, doc.page_content) for doc in documents]
            
            # Cross-encoder로 관련성 점수 계산
            scores = self.model.predict(query_doc_pairs)
            
            # 점수와 문서를 묶어서 정렬
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 k개 반환
            return doc_scores[:top_k]
            
        except Exception as e:
            print(f"재랭킹 실패: {str(e)}")
            return [(doc, 0.5) for doc in documents[:top_k]]
    
    def calculate_relevance_score(self, query: str, document: str) -> float:
        """
        단일 문서의 관련성 점수 계산
        
        Args:
            query: 검색 쿼리
            document: 문서 내용
            
        Returns:
            관련성 점수 (0-1)
        """
        if not self.model_loaded:
            return 0.5  # 기본값
        
        try:
            score = self.model.predict([(query, document)])[0]
            # 점수를 0-1 범위로 정규화
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5


class RAGRetriever:
    """
    고급 RAG 검색 시스템
    
    다양한 검색 전략을 조합하여 최적의 검색 결과를 제공합니다:
    - 의미 기반 벡터 검색
    - 키워드 기반 검색
    - 메타데이터 필터링
    - 시간적 가중치 적용
    - 재랭킹을 통한 결과 개선
    """
    
    def __init__(
        self, 
        vector_store: VectorStore,
        reranker: Optional[ReRanker] = None,
        enable_reranking: bool = True,
        keyword_threshold: float = 0.1
    ):
        """
        RAG 검색기 초기화
        
        Args:
            vector_store: 벡터 데이터베이스
            reranker: 재랭킹 모델 (없으면 자동 생성)
            enable_reranking: 재랭킹 활성화 여부
            keyword_threshold: 키워드 매칭 임계값 (0.0~1.0)
        """
        self.vector_store = vector_store
        self.enable_reranking = enable_reranking
        self.keyword_threshold = keyword_threshold  # 키워드 매칭 임계값 저장
        
        # 재랭커 초기화
        if enable_reranking:
            self.reranker = reranker or ReRanker()
        else:
            self.reranker = None
        
        # 검색 통계
        self._search_count = 0
        self._total_search_time = 0.0
    
    async def search(
        self,
        query: str,
        collection_name: str = "company_general",
        k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "ensemble",
        enable_temporal_boost: bool = False
    ) -> List[SearchResult]:
        """
        고급 검색 실행
        
        Args:
            query: 검색 쿼리
            collection_name: 검색할 컬렉션명
            k: 검색할 문서 수
            filters: 메타데이터 필터
            search_type: 검색 타입 (semantic, keyword, ensemble, hybrid_legacy)
            enable_temporal_boost: 시간적 가중치 적용 여부 (사용하지 않음)
            
        Returns:
            SearchResult 객체 리스트
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            print(f"[RAGRetriever] 검색 시작 - query: {query[:50]}..., collection: {collection_name}, k: {k}")
            print(f"[RAGRetriever] 검색 설정 - type: {search_type}, reranking: {self.enable_reranking}")
            
            # 1단계: 기본 벡터 검색
            raw_documents = await self._perform_vector_search(
                query, collection_name, k * 2, filters  # 더 많이 검색해서 재랭킹
            )
            
            print(f"[RAGRetriever] 1단계 벡터 검색 결과: {len(raw_documents)}개")
            if raw_documents:
                print(f"[RAGRetriever] 첫 번째 결과 거리: {raw_documents[0].get('distance_score', 'N/A')}")
            
            if not raw_documents:
                return []
            
            # 2단계: 검색 방법별 처리
            if search_type == "semantic":
                documents = raw_documents
                method = "semantic"
            # elif search_type == "keyword":
            #     documents = self._keyword_filter_legacy(query, raw_documents)
            #     method = "keyword"
            elif search_type == "ensemble":
                documents = await self._ensemble_search(query, collection_name, k, filters)
                method = "ensemble"
            # else:  # legacy hybrid
            #     documents = self._hybrid_search_legacy(query, raw_documents)
            #     method = "hybrid_legacy"
            
            print(f"[RAGRetriever] 2단계 {search_type} 처리 결과: {len(documents)}개")
            if documents:
                print(f"[RAGRetriever] 첫 번째 결과 거리: {documents[0].get('distance_score', 'N/A')}")
            
            # 3단계: 재랭킹 (활성화된 경우)
            if self.reranker and self.enable_reranking:
                doc_list = [self._result_to_document(result) for result in documents]
                reranked = self.reranker.rerank_documents(query, doc_list, k)
                
                print(f"[RAGRetriever] 4단계 재랭킹 결과: {len(reranked)}개")
                
                # 재랭킹 결과를 SearchResult로 변환
                search_results = []
                for rank, (doc, score) in enumerate(reranked):
                    # 거리를 관련성 점수로 변환 (0~1 범위, 높을수록 관련성 높음)
                    relevance_score = max(0.0, 1.0 - (score / 2.0))
                    print(f"[RAGRetriever] 재랭킹 결과 {rank+1}: distance={score:.3f}, relevance={relevance_score:.3f}")
                    search_results.append(SearchResult(
                        document=doc,
                        relevance_score=relevance_score,
                        rank=rank + 1,
                        retrieval_method=f"{method}_reranked"
                    ))
            else:
                print(f"[RAGRetriever] 재랭킹 없이 상위 {k}개 선택...")
                # 재랭킹 없이 상위 k개 반환
                search_results = []
                for rank, result in enumerate(documents[:k]):
                    distance_score = result.get("distance_score", 1.0)
                    # 거리를 관련성 점수로 변환 (0~1 범위, 높을수록 관련성 높음)
                    relevance_score = max(0.0, 1.0 - (distance_score / 2.0))
                    print(f"[RAGRetriever] 결과 {rank+1}: distance={distance_score:.3f}, relevance={relevance_score:.3f}")
                    search_results.append(SearchResult(
                        document=self._result_to_document(result),
                        relevance_score=relevance_score,
                        rank=rank + 1,
                        retrieval_method=method
                    ))
            
            print(f"[RAGRetriever] 최종 SearchResult 객체:{collection_name} {len(search_results)}개")
            
            # 검색 통계 업데이트
            search_time = asyncio.get_event_loop().time() - start_time
            self._update_search_stats(search_time)
            
            return search_results
            
        except Exception as e:
            print(f"검색 실행 실패: {str(e)}")
            return []
    
    async def _perform_vector_search(
        self,
        query: str,
        collection_name: str,
        k: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """벡터 데이터베이스에서 유사도 검색 실행"""
        return await self.vector_store.search_similar_documents(
            query=query,
            collection_name=collection_name,
            k=k,
            filter_dict=filters
        )
    

    
    async def _ensemble_search(
        self,
        query: str,
        collection_name: str,
        k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ensemble 검색 (BM25 + Vector Similarity)
        
        Args:
            query: 검색 쿼리
            collection_name: 컬렉션명
            k: 반환할 문서 수
            filters: 메타데이터 필터
            
        Returns:
            Ensemble 검색 결과 문서들
        """
        try:
            print(f"[RAGRetriever] Ensemble 검색 시작 - collection: {collection_name}")
            
            # LangChain Chroma 래퍼 가져오기
            chroma = self.vector_store.get_langchain_chroma(collection_name)
            if not chroma:
                print(f"[RAGRetriever] LangChain Chroma 래퍼 없음, 폴백 검색 사용")
                return await self._perform_vector_search(query, collection_name, k, filters)
            
            # ChromaDB에서 모든 문서 가져오기 (BM25용)
            raw_docs = chroma.get(include=["documents", "metadatas"])
            if not raw_docs["documents"]:
                print(f"[RAGRetriever] 컬렉션에 문서가 없음: {collection_name}")
                return []
            
            # Document 객체로 변환
            documents = [
                Document(page_content=doc, metadata=meta or {})
                for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"] or [{}] * len(raw_docs["documents"]))
            ]
            
            print(f"[RAGRetriever] BM25용 문서 로드 완료: {len(documents)}개")
            
            # BM25 Retriever 생성
            bm25_retriever = BM25Retriever.from_documents(documents=documents, k=k)
            print(f"[RAGRetriever] BM25 Retriever 생성 완료")
            
            # Vector Search Retriever 생성
            vector_retriever = chroma.as_retriever(
                search_type="similarity",
                search_kwargs={'k': k}
            )
            print(f"[RAGRetriever] Vector Retriever 생성 완료")
            
            # Ensemble Retriever 생성 (BM25:Vector = 0.5:0.5)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.5, 0.5]
            )
            print(f"[RAGRetriever] Ensemble Retriever 생성 완료")
            
            # Ensemble 검색 실행
            ensemble_results = ensemble_retriever.get_relevant_documents(query)
            print(f"[RAGRetriever] Ensemble 검색 결과: {len(ensemble_results)}개")
            
            # 결과를 기존 형식으로 변환
            converted_results = []
            for i, doc in enumerate(ensemble_results[:k]):
                converted_result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "distance_score": 0.5,  # Ensemble에서는 거리 점수가 없으므로 기본값
                    "ensemble_rank": i + 1
                }
                converted_results.append(converted_result)
            
            print(f"[RAGRetriever] Ensemble 검색 변환 완료: {len(converted_results)}개")
            return converted_results
            
        except Exception as e:
            print(f"[RAGRetriever] Ensemble 검색 실패: {str(e)}")
            # 폴백: 기존 벡터 검색 사용
            return await self._perform_vector_search(query, collection_name, k, filters)
    
    def _result_to_document(self, result: Dict[str, Any]) -> Document:
        """검색 결과를 Document 객체로 변환"""
        return Document(
            page_content=result.get("content", ""),
            metadata=result.get("metadata", {})
        )
    
    def _update_search_stats(self, search_time: float):
        """검색 통계 업데이트"""
        self._search_count += 1
        self._total_search_time += search_time
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계 반환"""
        avg_search_time = (
            self._total_search_time / self._search_count 
            if self._search_count > 0 else 0
        )
        
        return {
            "total_searches": self._search_count,
            "total_search_time": self._total_search_time,
            "average_search_time": avg_search_time,
            "reranking_enabled": self.enable_reranking,
            "reranker_loaded": self.reranker.model_loaded if self.reranker else False
        }
    
    async def search_by_company(
        self,
        company: str,
        query: str,
        category: Optional[str] = None,
        k: int = 10
    ) -> List[SearchResult]:
        """
        특정 회사에 대한 검색
        
        Args:
            company: 회사명
            query: 검색 쿼리
            category: 카테고리 필터 (culture, salary, career)
            k: 반환할 문서 수
            
        Returns:
            SearchResult 객체 리스트
        """
        # 회사 필터 구성
        filters = {"company": company}
        if category:
            filters["category"] = category
        
        # 회사명을 포함한 확장 쿼리 구성
        expanded_query = f"{company} {query}"
        
        return await self.search(
            query=expanded_query,
            filters=filters,
            k=k,
            search_type="ensemble"
        )
    
    async def search_by_category(
        self,
        category: str,
        query: str,
        companies: Optional[List[str]] = None,
        k: int = 10
    ) -> List[SearchResult]:
        """
        특정 카테고리에 대한 검색
        
        Args:
            category: 카테고리 (culture, salary, career)
            query: 검색 쿼리
            companies: 회사 필터 (없으면 모든 회사)
            k: 반환할 문서 수
            
        Returns:
            SearchResult 객체 리스트
        """
        # 카테고리 기반 컬렉션 선택 (새로운 6개 카테고리 1:1 매핑)
        collection_map = {
            "culture": "company_culture",
            "company_culture": "company_culture",
            "work_life_balance": "work_life_balance",
            "management": "management",
            "salary": "salary_benefits",
            "salary_benefits": "salary_benefits",
            "career": "career_growth",
            "career_growth": "career_growth",
            "general": "general"
        }
        
        collection_name = collection_map.get(category, "general")
        
        # 필터 구성
        filters = {"category": category}
        if companies:
            # 여러 회사 중 하나라도 매칭되도록 처리 (ChromaDB는 OR 조건 지원 제한)
            # 여기서는 단순화하여 첫 번째 회사만 사용
            if len(companies) == 1:
                filters["company"] = companies[0]
        
        return await self.search(
            query=query,
            collection_name=collection_name,
            filters=filters,
            k=k,
            search_type="ensemble"
        )
    
    async def get_document_suggestions(
        self,
        document: Document,
        k: int = 5
    ) -> List[SearchResult]:
        """
        특정 문서와 유사한 문서들 추천
        
        Args:
            document: 기준 문서
            k: 추천할 문서 수
            
        Returns:
            유사한 문서들의 SearchResult 리스트
        """
        # 문서 내용의 핵심 키워드 추출 (간단한 방법)
        content = document.page_content
        words = content.split()
        
        # 긴 단어들을 키워드로 사용 (3글자 이상)
        keywords = [word for word in words[:20] if len(word) >= 3]
        query = " ".join(keywords[:10])  # 상위 10개 키워드 사용
        
        # 동일한 회사/카테고리 우선 검색
        metadata = document.metadata
        filters = {}
        if metadata.get("company"):
            filters["company"] = metadata["company"]
        if metadata.get("category"):
            filters["category"] = metadata["category"]
        
        return await self.search(
            query=query,
            filters=filters,
            k=k,
            search_type="semantic"
        )