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
        search_type: str = "hybrid",
        enable_temporal_boost: bool = False
    ) -> List[SearchResult]:
        """
        고급 검색 실행
        
        Args:
            query: 검색 쿼리
            collection_name: 검색할 컬렉션명
            k: 검색할 문서 수
            filters: 메타데이터 필터
            search_type: 검색 타입 (semantic, keyword, hybrid)
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
            elif search_type == "keyword":
                documents = self._keyword_filter(query, raw_documents)
                method = "keyword"
            else:  # hybrid
                documents = self._hybrid_search(query, raw_documents)
                method = "hybrid"
            
            print(f"[RAGRetriever] 2단계 {search_type} 처리 결과: {len(documents)}개")
            if documents:
                print(f"[RAGRetriever] 첫 번째 결과 거리: {documents[0].get('distance_score', 'N/A')}")
            
            # 4단계: 재랭킹 (활성화된 경우)
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
            
            print(f"[RAGRetriever] 최종 SearchResult 객체: {len(search_results)}개")
            
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
            documents: 필터링할 문서들 -> 벡터디비에서 온 문서들
            
        Returns:
            키워드가 포함된 문서들
        """
        # 쿼리에서 키워드 추출 (공백으로 분리)
        keywords = [word.strip().lower() for word in query.split() if len(word.strip()) > 1]
        
        print(f"[RAGRetriever] 키워드 필터링 - 추출된 키워드: {keywords}")
        print(f"[RAGRetriever] 키워드 필터링 - 입력 문서 수: {len(documents)}개")
        
        if not keywords:
            return documents
        
        filtered_docs = []
        
        for idx, doc in enumerate(documents):
            content = doc.get("content", "").lower()
            
            # 키워드 매칭 점수 계산 (한국어 유사 단어 고려)
            matched_keywords = 0
            for keyword in keywords:
                if keyword in content:
                    matched_keywords += 1
                else:
                    # 유사 단어 매칭 (한국어 어미 변화 고려)
                    similar_words = self._get_similar_korean_words(keyword)
                    if any(similar in content for similar in similar_words):
                        matched_keywords += 0.7  # 유사 매칭은 0.7점
            
            keyword_score = matched_keywords / len(keywords)
            
            if idx < 3:  # 처음 3개 문서에 대해서만 디버깅 출력
                print(f"[RAGRetriever] 문서 {idx+1}: 매칭 키워드={matched_keywords}/{len(keywords)}, 점수={keyword_score:.3f}")
                print(f"[RAGRetriever] 문서 내용 일부: {content[:100]}...")
            
            # 키워드 매칭 임계값 적용 (설정 가능한 임계값 사용)
            if keyword_score >= self.keyword_threshold:
                # 키워드 점수를 거리 점수에 반영 (거리 감소 = 더 유사함)
                original_distance = doc.get("distance_score", 1.0)
                distance_reduction = keyword_score * 0.3  # 최대 30% 거리 감소
                improved_distance = max(0.0, original_distance * (1 - distance_reduction))
                doc["distance_score"] = improved_distance
                doc["keyword_match_score"] = keyword_score
                filtered_docs.append(doc)
            else:
                if idx < 3:  # 처음 3개 제외된 문서에 대해서만 디버깅 출력
                    print(f"[RAGRetriever] 문서 {idx+1} 제외됨: 키워드 점수 {keyword_score:.3f} < 임계값 {self.keyword_threshold}")
        
        print(f"[RAGRetriever] 키워드 필터링 결과: {len(filtered_docs)}개 통과 / {len(documents)}개 입력 (임계값: {self.keyword_threshold})")
        
        # 개선된 거리로 재정렬 (낮은 거리가 먼저)
        filtered_docs.sort(key=lambda x: x.get("distance_score", 2.0), reverse=False)
        return filtered_docs
    
    def _get_similar_korean_words(self, keyword: str) -> List[str]:
        """
        한국어 키워드의 유사 단어들을 생성
        
        Args:
            keyword: 기본 키워드
            
        Returns:
            유사 단어 리스트
        """
        similar_words = []
        
        # 기본 어미 변화 패턴
        if keyword.endswith('좋은'):
            similar_words.extend(['좋으신', '좋았', '좋아', '좋더', '좋다'])
        elif keyword.endswith('직원'):
            similar_words.extend(['직원들', '사원', '사원들', '사람', '사람들'])
        elif keyword.endswith('분위기'):
            similar_words.extend(['환경', '문화', '정서', '느낌'])
        elif keyword.endswith('회사'):
            similar_words.extend(['기업', '회사님', '직장'])
        elif keyword.endswith('동료'):
            similar_words.extend(['동료들', '팀원', '팀원들', '동료분', '동료분들'])
        
        # 일반적인 어미 변화 (받침 여부에 따라)
        if len(keyword) > 2:
            root = keyword[:-1]
            # 현재진행형, 과거형, 존댓말 등 기본 변화
            similar_words.extend([
                root + '는', root + '은', root + '한', root + '된',
                root + '고', root + '며', root + '면서', root + '지만'
            ])
        
        return similar_words
    
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
            original_distance = doc.get("distance_score", 1.0)
            distance_reduction_factor = 0.0  # 거리 감소 비율
            
            # 회사명 매칭 부스트 (거리 20% 감소)
            company = metadata.get("company", "").lower()
            if company and company in query_lower:
                distance_reduction_factor += 0.2
            
            # 카테고리 매칭 부스트 (거리 15% 감소)
            category = metadata.get("category", "").lower()
            category_keywords = {
                "company_culture": ["문화", "분위기", "사내문화", "회사문화"],
                "work_life_balance": ["워라밸", "야근", "휴가", "근무시간"],
                "management": ["경영진", "상사", "관리", "리더십"],
                "salary_benefits": ["연봉", "급여", "보너스", "혜택", "복지"],
                "career_growth": ["커리어", "승진", "성장", "발전", "교육"],
                "general": ["일반", "기타", "전반적"]
            }
            
            for cat, keywords in category_keywords.items():
                if category == cat and any(keyword in query_lower for keyword in keywords):
                    distance_reduction_factor += 0.15
                    break
            
            # 신뢰도 점수 기반 부스트 (거리 10% 감소)
            credibility = metadata.get("credibility_score", 0.5)
            if credibility > 0.7:
                distance_reduction_factor += 0.1
            
            # 검증된 정보 부스트 (거리 10% 감소)
            if metadata.get("verified", False):
                distance_reduction_factor += 0.1
            
            # 최종 거리 적용 (최대 55% 감소 가능)
            distance_reduction_factor = min(0.55, distance_reduction_factor)  # 최대 감소율 제한
            improved_distance = max(0.0, original_distance * (1 - distance_reduction_factor))
            doc["distance_score"] = improved_distance
            doc["distance_reduction"] = distance_reduction_factor
        
        # 개선된 거리로 재정렬 (낮은 거리가 먼저)
        documents.sort(key=lambda x: x.get("distance_score", 2.0), reverse=False)
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