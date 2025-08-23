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
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        재랭커 초기화
        
        Args:
            model_name: 사용할 Cross-encoder 모델명
        """
        try:
            # Cross-encoder 모델 로드
            self.model = CrossEncoder(model_name)
            self.model_loaded = True
            print(f"재랭킹 모델 로드 완료: {model_name}")
        except Exception as e:
            print(f"재랭킹 모델 로드 실패: {str(e)}")
            self.model = None
            self.model_loaded = False
    
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
        enable_reranking: bool = True
    ):
        """
        RAG 검색기 초기화
        
        Args:
            vector_store: 벡터 데이터베이스
            reranker: 재랭킹 모델 (없으면 자동 생성)
            enable_reranking: 재랭킹 활성화 여부
        """
        self.vector_store = vector_store
        self.enable_reranking = enable_reranking
        
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
        search_type: str = "hybrid",
        enable_temporal_boost: bool = True
    ) -> List[SearchResult]:
        """
        고급 검색 실행
        
        Args:
            query: 검색 쿼리
            collection_name: 검색할 컬렉션명
            k: 검색할 문서 수
            filters: 메타데이터 필터
            search_type: 검색 타입 (semantic, keyword, hybrid)
            enable_temporal_boost: 시간적 가중치 적용 여부
            
        Returns:
            SearchResult 객체 리스트
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 1단계: 기본 벡터 검색
            raw_documents = await self._perform_vector_search(
                query, collection_name, k * 2, filters  # 더 많이 검색해서 재랭킹
            )
            
            if not raw_documents:
                return []
            
            # 2단계: 검색 방법별 처리
            if search_type == "semantic":
                documents = raw_documents
                method = "semantic"
            elif search_type == "keyword":
                documents = self._keyword_filter(query, raw_documents)
                method = "keyword"
            else:  # hybrid
                documents = self._hybrid_search(query, raw_documents)
                method = "hybrid"
            
            # 3단계: 시간적 가중치 적용
            if enable_temporal_boost:
                documents = self._apply_temporal_boost(documents)
            
            # 4단계: 재랭킹 (활성화된 경우)
            if self.reranker and self.enable_reranking:
                doc_list = [self._result_to_document(result) for result in documents]
                reranked = self.reranker.rerank_documents(query, doc_list, k)
                
                # 재랭킹 결과를 SearchResult로 변환
                search_results = []
                for rank, (doc, score) in enumerate(reranked):
                    search_results.append(SearchResult(
                        document=doc,
                        relevance_score=score,
                        rank=rank + 1,
                        retrieval_method=f"{method}_reranked"
                    ))
            else:
                # 재랭킹 없이 상위 k개 반환
                search_results = []
                for rank, result in enumerate(documents[:k]):
                    search_results.append(SearchResult(
                        document=self._result_to_document(result),
                        relevance_score=result.get("similarity_score", 0.5),
                        rank=rank + 1,
                        retrieval_method=method
                    ))
            
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
    
    def _keyword_filter(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        키워드 기반 필터링
        
        Args:
            query: 검색 쿼리
            documents: 필터링할 문서들
            
        Returns:
            키워드가 포함된 문서들
        """
        # 쿼리에서 키워드 추출 (공백으로 분리)
        keywords = [word.strip().lower() for word in query.split() if len(word.strip()) > 1]
        
        if not keywords:
            return documents
        
        filtered_docs = []
        
        for doc in documents:
            content = doc.get("content", "").lower()
            
            # 키워드 매칭 점수 계산
            matched_keywords = sum(1 for keyword in keywords if keyword in content)
            keyword_score = matched_keywords / len(keywords)
            
            # 최소 50% 키워드가 매칭되는 문서만 포함
            if keyword_score >= 0.5:
                # 키워드 점수를 유사도 점수에 반영
                original_score = doc.get("similarity_score", 0.5)
                boosted_score = original_score * (1 + keyword_score * 0.3)  # 최대 30% 부스트
                doc["similarity_score"] = min(1.0, boosted_score)
                doc["keyword_match_score"] = keyword_score
                filtered_docs.append(doc)
        
        # 부스트된 점수로 재정렬
        filtered_docs.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return filtered_docs
    
    def _hybrid_search(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 (의미 + 키워드)
        
        Args:
            query: 검색 쿼리
            documents: 검색된 문서들
            
        Returns:
            하이브리드 점수로 정렬된 문서들
        """
        # 키워드 필터링 적용
        keyword_filtered = self._keyword_filter(query, documents.copy())
        
        # 메타데이터 기반 부스트 적용
        boosted_docs = self._apply_metadata_boost(query, keyword_filtered)
        
        return boosted_docs
    
    def _apply_metadata_boost(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        메타데이터 기반 점수 부스트 적용
        
        Args:
            query: 검색 쿼리
            documents: 부스트할 문서들
            
        Returns:
            부스트된 점수로 정렬된 문서들
        """
        query_lower = query.lower()
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            original_score = doc.get("similarity_score", 0.5)
            boost_factor = 1.0
            
            # 회사명 매칭 부스트
            company = metadata.get("company", "").lower()
            if company and company in query_lower:
                boost_factor += 0.2
            
            # 카테고리 매칭 부스트
            category = metadata.get("category", "").lower()
            category_keywords = {
                "culture": ["문화", "분위기", "워라밸", "복지"],
                "salary": ["연봉", "급여", "보너스", "혜택"],
                "career": ["커리어", "승진", "성장", "면접"]
            }
            
            for cat, keywords in category_keywords.items():
                if category == cat and any(keyword in query_lower for keyword in keywords):
                    boost_factor += 0.15
                    break
            
            # 신뢰도 점수 기반 부스트
            credibility = metadata.get("credibility_score", 0.5)
            if credibility > 0.7:
                boost_factor += 0.1
            
            # 검증된 정보 부스트
            if metadata.get("verified", False):
                boost_factor += 0.1
            
            # 최종 점수 적용
            doc["similarity_score"] = min(1.0, original_score * boost_factor)
            doc["boost_factor"] = boost_factor
        
        # 부스트된 점수로 재정렬
        documents.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return documents
    
    def _apply_temporal_boost(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        시간적 가중치 적용 (최신 정보에 가중치)
        
        Args:
            documents: 가중치를 적용할 문서들
            
        Returns:
            시간적 가중치가 적용된 문서들
        """
        current_time = datetime.now()
        
        for doc in documents:
            metadata = doc.get("metadata", {})
            
            # 문서 생성 시간 파싱
            created_at_str = metadata.get("created_at")
            if not created_at_str:
                continue
            
            try:
                if isinstance(created_at_str, str):
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                else:
                    created_at = created_at_str
                
                # 시간 차이 계산 (일 단위)
                days_old = (current_time - created_at).days
                
                # 시간적 가중치 계산 (최근 30일 내 정보에 가중치)
                if days_old <= 30:
                    temporal_boost = 1.2  # 20% 부스트
                elif days_old <= 90:
                    temporal_boost = 1.1  # 10% 부스트
                elif days_old <= 365:
                    temporal_boost = 1.0  # 가중치 없음
                else:
                    temporal_boost = 0.9  # 10% 페널티
                
                # 기존 점수에 시간적 가중치 적용
                original_score = doc.get("similarity_score", 0.5)
                doc["similarity_score"] = min(1.0, original_score * temporal_boost)
                doc["temporal_boost"] = temporal_boost
                doc["days_old"] = days_old
                
            except Exception as e:
                print(f"시간 파싱 실패: {str(e)}")
                continue
        
        # 시간적 가중치가 적용된 점수로 재정렬
        documents.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return documents
    
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
            search_type="hybrid"
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
        # 카테고리 기반 컬렉션 선택
        collection_map = {
            "culture": "culture_reviews",
            "salary": "salary_discussions", 
            "career": "career_advice"
        }
        
        collection_name = collection_map.get(category, "company_general")
        
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
            search_type="hybrid"
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