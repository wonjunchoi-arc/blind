"""
RAG 검색 및 재랭킹 시스템 (개선 버전)

고품질 검색 결과를 위한 다단계 검색과 재랭킹을 제공하는 시스템입니다.
의미 기반 검색과 키워드 검색을 결합하여 최적의 결과를 반환합니다.
"""

import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

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
    고급 RAG 검색 시스템 (개선 버전)
    
    다양한 검색 전략을 조합하여 최적의 검색 결과를 제공합니다:
    - 의미 기반 벡터 검색
    - 키워드 기반 검색
    - 메타데이터 필터링
    - 캐싱을 통한 성능 최적화
    - 재랭킹을 통한 결과 개선
    """
    
    def __init__(
        self, 
        vector_store: VectorStore,
        reranker: Optional[ReRanker] = None,
        enable_reranking: bool = True,
        keyword_threshold: float = 0.1,
        ensemble_weights: Optional[Tuple[float, float]] = None,
        relevance_threshold: Optional[float] = None,
        cache_ttl: int = 3600
    ):
        """
        RAG 검색기 초기화
        
        Args:
            vector_store: 벡터 데이터베이스
            reranker: 재랭킹 모델 (없으면 자동 생성)
            enable_reranking: 재랭킹 활성화 여부
            keyword_threshold: 키워드 매칭 임계값 (0.0~1.0)
            ensemble_weights: (BM25, Vector) 가중치 튜플 (없으면 설정에서 가져옴)
            relevance_threshold: 관련성 점수 임계값 (없으면 설정에서 가져옴)
            cache_ttl: 캐시 유효 시간 (초)
        """
        self.vector_store = vector_store
        self.enable_reranking = enable_reranking
        self.keyword_threshold = keyword_threshold
        
        # 설정에서 가중치와 임계값 가져오기
        if ensemble_weights is None:
            self.ensemble_weights = (settings.ensemble_weight_bm25, settings.ensemble_weight_vector)
        else:
            self.ensemble_weights = ensemble_weights
            
        if relevance_threshold is None:
            self.relevance_threshold = settings.relevance_score_threshold
        else:
            self.relevance_threshold = relevance_threshold
            
        self.cache_ttl = cache_ttl
        
        # 재랭커 초기화
        if enable_reranking:
            self.reranker = reranker or ReRanker()
        else:
            self.reranker = None
        
        # BM25 retriever 캐시
        self._bm25_cache = {}
        
        # 검색 통계
        self._search_count = 0
        self._total_search_time = 0.0
    
    async def _get_or_create_bm25_retriever(
        self,
        collection_name: str,
        chroma,
        filters: Optional[Dict[str, Any]] = None,
        k: int = 20
    ) -> BM25Retriever:
        """
        캐싱된 BM25 retriever 반환 또는 생성
        
        Args:
            collection_name: 컬렉션명
            chroma: ChromaDB 인스턴스
            filters: 필터 조건
            k: 검색할 문서 수
            
        Returns:
            BM25Retriever 인스턴스
        """
        # 캐시 키 생성
        cache_key = f"{collection_name}_{str(sorted(filters.items()) if filters else 'no_filter')}"
        
        # 캐시 확인
        current_time = time.time()
        if cache_key in self._bm25_cache:
            retriever, timestamp = self._bm25_cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                print(f"[RAGRetriever] BM25 캐시 히트: {cache_key}")
                return retriever
        
        print(f"[RAGRetriever] BM25 새로 생성: {cache_key}")
        
        # 필터링된 문서만 로드
        if filters:
            # ChromaDB where 절 구성
            where_clause = {}
            if len(filters) == 1:
                key, value = next(iter(filters.items()))
                where_clause = {key: {"$eq": value}}
            else:
                # 다중 조건은 $and 사용
                conditions = [{k: {"$eq": v}} for k, v in filters.items()]
                where_clause = {"$and": conditions}
            
            raw_docs = chroma.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )
        else:
            # 필터가 없을 때는 배치로 문서 로드
            raw_docs = await self._batch_load_documents(chroma)
        
        if not raw_docs["documents"]:
            print(f"[RAGRetriever] 문서가 없음: {collection_name}")
            # 빈 BM25 retriever 반환
            return BM25Retriever.from_documents(documents=[], k=k)
        
        # Document 객체로 변환
        documents = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(
                raw_docs["documents"], 
                raw_docs["metadatas"] or [{}] * len(raw_docs["documents"])
            )
        ]
        
        print(f"[RAGRetriever] BM25용 문서 로드 완료: {len(documents)}개 (필터: {bool(filters)})")
        
        # BM25 Retriever 생성
        retriever = BM25Retriever.from_documents(documents=documents, k=k)
        
        # 캐시 저장
        self._bm25_cache[cache_key] = (retriever, current_time)
        
        # 오래된 캐시 정리
        self._cleanup_cache()
        
        return retriever
    
    async def _batch_load_documents(
        self,
        chroma,
        batch_size: int = 1000
    ) -> Dict[str, List]:
        """
        대용량 문서를 배치로 로드
        
        Args:
            chroma: ChromaDB 인스턴스
            batch_size: 배치 크기
            
        Returns:
            문서와 메타데이터 딕셔너리
        """
        all_docs = {"documents": [], "metadatas": []}
        offset = 0
        
        while True:
            try:
                # ChromaDB에서 배치 로드
                batch = chroma.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas"]
                )
                
                if not batch["documents"]:
                    break
                
                all_docs["documents"].extend(batch["documents"])
                all_docs["metadatas"].extend(batch["metadatas"] or [{}] * len(batch["documents"]))
                
                offset += batch_size
                
                # 비동기로 다른 작업 허용
                await asyncio.sleep(0)
                
            except Exception as e:
                print(f"[RAGRetriever] 배치 로드 중 오류: {str(e)}")
                break
        
        print(f"[RAGRetriever] 배치 로드 완료: 총 {len(all_docs['documents'])}개 문서")
        return all_docs
    
    def _cleanup_cache(self):
        """오래된 캐시 항목 정리"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._bm25_cache.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self._bm25_cache[key]
        
        if expired_keys:
            print(f"[RAGRetriever] 캐시 정리: {len(expired_keys)}개 항목 제거")
    
    def _get_doc_id(self, doc: Document) -> str:
        """문서의 고유 ID 생성"""
        # 문서 내용의 처음 100자와 메타데이터를 조합하여 ID 생성
        content_preview = doc.page_content[:100]
        metadata_str = str(sorted(doc.metadata.items())) if doc.metadata else ""
        return f"{hash(content_preview + metadata_str)}"
    
    async def _calculate_ensemble_scores(
        self,
        query: str,
        bm25_results: List[Document],
        vector_results: List[Document],
        weights: Optional[Tuple[float, float]] = None
    ) -> List[Tuple[Document, float]]:
        """
        앙상블 점수 계산 (개선된 버전)
        
        Args:
            query: 검색 쿼리
            bm25_results: BM25 검색 결과
            vector_results: 벡터 검색 결과
            weights: 가중치 (기본값: self.ensemble_weights)
            
        Returns:
            (문서, 점수) 튜플 리스트
        """
        if weights is None:
            weights = self.ensemble_weights
        
        doc_scores = {}
        
        # BM25 점수 계산 (역순위 기반)
        for i, doc in enumerate(bm25_results):
            # 순위 기반 점수: 1/(순위+1) 방식으로 계산
            # 1위: 1.0, 2위: 0.5, 3위: 0.33, ...
            rank_score = 1.0 / (i + 1)
            # 정규화: 상위 20개 기준으로 스케일링
            normalized_score = rank_score / 1.0  # 1위의 점수로 정규화
            
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] = {
                'doc': doc,
                'bm25_score': normalized_score * weights[0],
                'vector_score': 0,
                'bm25_rank': i + 1,
                'vector_rank': None
            }
        
        # Vector 점수 추가
        for i, doc in enumerate(vector_results):
            # 벡터 검색도 동일한 방식으로 점수 계산
            rank_score = 1.0 / (i + 1)
            normalized_score = rank_score / 1.0
            
            doc_id = self._get_doc_id(doc)
            if doc_id in doc_scores:
                doc_scores[doc_id]['vector_score'] = normalized_score * weights[1]
                doc_scores[doc_id]['vector_rank'] = i + 1
            else:
                doc_scores[doc_id] = {
                    'doc': doc,
                    'bm25_score': 0,
                    'vector_score': normalized_score * weights[1],
                    'bm25_rank': None,
                    'vector_rank': i + 1
                }
        
        # 최종 점수 계산 및 정렬
        final_results = []
        for doc_id, scores in doc_scores.items():
            # 가중 평균 점수 계산
            total_score = scores['bm25_score'] + scores['vector_score']
            
            # 두 검색 모두에서 나타난 문서에 보너스 부여
            if scores['bm25_rank'] and scores['vector_rank']:
                # 두 방법 모두에서 상위 10위 안에 들면 추가 보너스
                if scores['bm25_rank'] <= 10 and scores['vector_rank'] <= 10:
                    total_score *= 1.2  # 20% 보너스
            
            # 점수를 0-1 범위로 정규화
            total_score = min(1.0, total_score)
            
            final_results.append((scores['doc'], total_score))
        
        # 점수 기준 내림차순 정렬
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        return final_results
    
    async def _ensemble_search(
        self,
        query: str,
        collection_name: str,
        k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        개선된 Ensemble 검색 (BM25 + Vector Similarity)
        
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
            
            # 1. BM25 Retriever 생성 (캐싱 적용, 필터링된 문서만 사용)
            bm25_retriever = await self._get_or_create_bm25_retriever(
                collection_name=collection_name,
                chroma=chroma,
                filters=filters,
                k=k * 2  # 더 많이 검색해서 앙상블에 활용
            )
            
            # 2. Vector Search Retriever 생성 (동일한 필터 적용)
            search_kwargs = {'k': k * 2}  # 더 많이 검색
            if filters:
                # ChromaDB 필터 형식으로 변환
                if len(filters) > 1:
                    filter_conditions = []
                    for key, value in filters.items():
                        filter_conditions.append({key: {"$eq": value}})
                    search_kwargs['filter'] = {"$and": filter_conditions}
                else:
                    # 단일 조건인 경우 직접 사용
                    key, value = next(iter(filters.items()))
                    search_kwargs['filter'] = {key: {"$eq": value}}
            
            vector_retriever = chroma.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs
            )
            print(f"[RAGRetriever] Vector Retriever 생성 완료 (필터: {bool(filters)})")
            
            # 3. 각 retriever에서 독립적으로 검색
            bm25_results = bm25_retriever.get_relevant_documents(query)
            vector_results = vector_retriever.get_relevant_documents(query)
            
            print(f"[RAGRetriever] BM25 결과: {len(bm25_results)}개, Vector 결과: {len(vector_results)}개")
            
            # 4. 개선된 점수 계산으로 앙상블
            ensemble_results = await self._calculate_ensemble_scores(
                query=query,
                bm25_results=bm25_results,
                vector_results=vector_results,
                weights=self.ensemble_weights
            )
            
            print(f"[RAGRetriever] Ensemble 병합 완료: {len(ensemble_results)}개")
            
            # 5. 결과를 기존 형식으로 변환
            converted_results = []
            for i, (doc, score) in enumerate(ensemble_results[:k]):
                converted_result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "distance_score": score,  # 실제 계산된 앙상블 점수
                    "ensemble_rank": i + 1
                }
                converted_results.append(converted_result)
            
            print(f"[RAGRetriever] Ensemble 검색 변환 완료: {len(converted_results)}개")
            return converted_results
            
        except Exception as e:
            print(f"[RAGRetriever] Ensemble 검색 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            # 폴백: 기존 벡터 검색 사용
            return await self._perform_vector_search(query, collection_name, k, filters)
    
    async def search(
        self,
        query: str,
        collection_name: str = "general",
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
            search_type: 검색 타입 (semantic, keyword, ensemble)
            enable_temporal_boost: 시간적 가중치 적용 여부 (사용하지 않음)
            
        Returns:
            SearchResult 객체 리스트
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            print(f"[RAGRetriever] 검색 시작 - query: {query[:50]}..., collection: {collection_name}, k: {k}")
            print(f"[RAGRetriever] 검색 설정 - type: {search_type}, reranking: {self.enable_reranking}")
            
            # 1단계: 검색 방법별 처리
            if search_type == "semantic":
                # 기본 벡터 검색
                raw_documents = await self._perform_vector_search(
                    query, collection_name, k * 2, filters
                )
                
                print(f"[RAGRetriever] 1단계 벡터 검색 결과: {len(raw_documents)}개")
                if not raw_documents:
                    return []
                
                documents = raw_documents
                method = "semantic"
                
            elif search_type == "ensemble":
                # Ensemble 검색 (개선된 버전)
                documents = await self._ensemble_search(query, collection_name, k, filters)
                method = "ensemble"
                
                print(f"[RAGRetriever] 1단계 Ensemble 검색 결과: {len(documents)}개")
                if not documents:
                    return []
            else:
                # 기본값: semantic 검색
                raw_documents = await self._perform_vector_search(
                    query, collection_name, k * 2, filters
                )
                documents = raw_documents
                method = "semantic"
            
            # 2단계: 재랭킹 (활성화된 경우)
            if self.reranker and self.enable_reranking:
                doc_list = [self._result_to_document(result) for result in documents]
                reranked = self.reranker.rerank_documents(query, doc_list, k)
                
                print(f"[RAGRetriever] 2단계 재랭킹 결과: {len(reranked)}개")
                
                # 재랭킹 결과를 SearchResult로 변환
                search_results = []
                scores = [score for _, score in reranked]
                if scores:
                    min_score = min(scores)
                    max_score = max(scores)
                    score_range = max_score - min_score
                    
                for rank, (doc, score) in enumerate(reranked):
                    # 재랭킹 점수를 0~1 범위로 정규화
                    if score_range > 0:
                        normalized_score = (score - min_score) / score_range
                        relevance_score = 0.1 + (normalized_score * 0.9)
                    else:
                        relevance_score = 0.8
                    
                    search_results.append(SearchResult(
                        document=doc,
                        relevance_score=relevance_score,
                        rank=rank + 1,
                        retrieval_method=f"{method}_reranked"
                    ))
            else:
                # 재랭킹 없이 상위 k개 반환
                search_results = []
                for rank, result in enumerate(documents[:k]):
                    similarity_score = result.get("distance_score", 0.5)
                    search_results.append(SearchResult(
                        document=self._result_to_document(result),
                        relevance_score=similarity_score,
                        rank=rank + 1,
                        retrieval_method=method
                    ))
            
            print(f"[RAGRetriever] 최종 SearchResult 객체: {len(search_results)}개")
            
            # 3단계: 관련성 점수로 필터링
            filtered_results = self._filter_by_relevance_score(search_results)
            
            print(f"[RAGRetriever] 관련성 필터링 후: {len(filtered_results)}개")
            
            # 검색 통계 업데이트
            search_time = asyncio.get_event_loop().time() - start_time
            self._update_search_stats(search_time)
            
            return filtered_results
            
        except Exception as e:
            print(f"검색 실행 실패: {str(e)}")
            import traceback
            traceback.print_exc()
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
    
    def _result_to_document(self, result: Dict[str, Any]) -> Document:
        """검색 결과를 Document 객체로 변환"""
        return Document(
            page_content=result.get("content", ""),
            metadata=result.get("metadata", {})
        )
    
    def _filter_by_relevance_score(self, search_results: List[SearchResult]) -> List[SearchResult]:
        """
        관련성 점수를 기준으로 검색 결과 필터링
        
        Args:
            search_results: 필터링할 검색 결과 리스트
            
        Returns:
            필터링된 검색 결과 리스트
        """
        if not search_results or self.relevance_threshold <= 0:
            return search_results
        
        # 관련성 점수가 임계값 이상인 결과만 유지
        filtered_results = [
            result for result in search_results 
            if result.relevance_score >= self.relevance_threshold
        ]
        
        print(f"[RAGRetriever] 관련성 점수 필터링: {len(search_results)}개 → {len(filtered_results)}개 (임계값: {self.relevance_threshold})")
        
        # 필터링 후 순위 재조정
        for i, result in enumerate(filtered_results):
            result.rank = i + 1
        
        return filtered_results
    
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
            "reranker_loaded": self.reranker.model_loaded if self.reranker else False,
            "cache_size": len(self._bm25_cache),
            "ensemble_weights": self.ensemble_weights,
            "relevance_threshold": self.relevance_threshold
        }
    
    def clear_cache(self):
        """캐시 수동 초기화"""
        self._bm25_cache.clear()
        print("[RAGRetriever] 캐시가 초기화되었습니다.")
    
    # async def search_by_company(
    #     self,
    #     company: str,
    #     query: str,
    #     category: Optional[str] = None,
    #     k: int = 10
    # ) -> List[SearchResult]:
    #     """
    #     특정 회사에 대한 검색
        
    #     Args:
    #         company: 회사명
    #         query: 검색 쿼리
    #         category: 카테고리 필터 (culture, salary, career)
    #         k: 반환할 문서 수
            
    #     Returns:
    #         SearchResult 객체 리스트
    #     """
    #     # 회사 필터 구성
    #     filters = {"company": company}