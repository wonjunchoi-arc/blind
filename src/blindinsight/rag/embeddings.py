"""
임베딩 관리 및 벡터 데이터베이스 시스템 (수정된 버전)

OpenAI 임베딩을 사용하여 텍스트를 벡터로 변환하고
ChromaDB를 통해 효율적인 벡터 검색을 제공합니다.
"""

import os
import json
import hashlib
import logging
import asyncio
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import functools
import time
# HTTP 요청 로그 레벨 조정
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import numpy as np
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

from ..models.base import BaseModel, settings



@dataclass
class DocumentEmbedding:
    """문서 임베딩 정보를 담는 데이터 클래스"""
    
    document_id: str              # 문서 고유 식별자
    content: str                  # 원본 텍스트 내용
    embedding: List[float]        # 벡터 임베딩
    metadata: Dict[str, Any]      # 메타데이터 (회사, 카테고리 등)
    created_at: datetime          # 생성 시간
    hash_value: str               # 내용 해시값 (중복 검사용)


class EmbeddingManager:
    """
    임베딩 생성 및 관리를 담당하는 클래스 (수정된 버전)
    
    OpenAI의 text-embedding-3-small 모델을 사용하여
    한국어 텍스트를 고품질 벡터로 변환합니다.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 100):
        """
        임베딩 관리자 초기화
        
        Args:
            model_name: 사용할 OpenAI 임베딩 모델명
            batch_size: 배치 처리 크기
        """
        # 모델별 차원 설정
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        dimensions = model_dimensions.get(model_name, 1536)
        
        # OpenAI 임베딩 모델 초기화
        self.embedding_model = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.openai_api_key,
            dimensions=dimensions,
            show_progress_bar=False  # 커스텀 진행률 표시 사용
        )
        
        self.model_name = model_name
        self.dimensions = dimensions
        self.batch_size = batch_size
        
        # LangChain Chroma용 임베딩 함수
        self.embedding_function = self.embedding_model
        
        # 임베딩 캐시 (메모리 효율성을 위해)
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # API 호출 제한을 위한 세마포어
        self.semaphore = asyncio.Semaphore(5)
    
    async def create_embedding(self, text: str) -> List[float]:
        """
        단일 텍스트에 대한 임베딩 생성
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            벡터 임베딩
        """
        if not text or not text.strip():
            return [0.0] * self.dimensions
            
        # 캐시 확인
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[text_hash]
        
        try:
            async with self.semaphore:
                embedding = await self.embedding_model.aembed_query(text)
                self._embedding_cache[text_hash] = embedding
                self._cache_misses += 1
                return embedding
                
        except Exception as e:
            print(f"임베딩 생성 실패: {str(e)}")
            return [0.0] * self.dimensions
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 반환"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cached_items": len(self._embedding_cache)
        }
   
   
    
    async def create_batch_embeddings_optimized(
    self, 
    texts: List[str], 
    max_batch_size: int = 100
    ) -> List[List[float]]:
        """
        최적화된 배치 임베딩 생성 (메모리 효율성 개선)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            max_batch_size: 최대 배치 크기
            
        Returns:
            벡터 임베딩 리스트들의 리스트
        """
        if not texts:
            return []
        
        print(f"[임베딩] 총 {len(texts)}개 텍스트를 {max_batch_size}개씩 배치 처리")
        
        # 텍스트를 배치 크기로 분할
        batches = [
            texts[i:i + max_batch_size] 
            for i in range(0, len(texts), max_batch_size)
        ]
        
        all_embeddings = []
        
        # 배치별 처리 (진행률 표시)
        for i, batch in enumerate(tqdm(batches, desc="임베딩 배치", unit="batch")):
            print(f"  배치 {i+1}/{len(batches)}: {len(batch)}개 텍스트 처리 중...")
            
            try:
                async with self.semaphore:
                    # OpenAI API 배치 호출
                    start_time = time.time()
                    embeddings = await self.embedding_model.aembed_documents(batch)
                    end_time = time.time()
                    
                    print(f"    완료 - {end_time - start_time:.2f}초 소요")
                    all_embeddings.extend(embeddings)
                    
                    # API 제한 대응을 위한 짧은 대기
                    if i < len(batches) - 1:
                        await asyncio.sleep(0.1)
                    
            except Exception as e:
                print(f"    배치 {i+1} API 호출 실패, 개별 처리로 전환: {str(e)}")
                
        
        print(f"[완료] 총 {len(all_embeddings)}개 임베딩 생성 완료")
        return all_embeddings
        
   

class VectorStore:
    """
    벡터 데이터베이스 관리 클래스 (수정된 버전)
    
    ChromaDB를 사용하여 문서 임베딩을 저장하고
    유사도 기반 검색을 제공합니다.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, persist_directory: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, persist_directory: str = None):
        """
        벡터 스토어 초기화 (싱글턴 패턴으로 한 번만 실행)
        
        Args:
            persist_directory: 데이터베이스 저장 경로
        """
        # 이미 초기화된 경우 중복 초기화 방지
        if VectorStore._initialized:
            return
            
        self.persist_directory = persist_directory or settings.vector_db_path
        
        # ChromaDB 클라이언트 초기화
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # 텔레메트리 비활성화
                is_persistent=True,          # 영구 저장 활성화
                allow_reset=True             # 리셋 허용
            )
        )
        
        print(f"ChromaDB 클라이언트 초기화 완료: {self.persist_directory}")
        
        # 임베딩 함수 설정
        self.embedding_manager = EmbeddingManager()
        
        # 회사 메타데이터 관리자 초기화 (지연 import로 순환 import 방지)
        from .json_processor import CompanyMetadataManager
        self.metadata_manager = CompanyMetadataManager()
        print(f"CompanyMetadataManager 초기화 완료: {self.metadata_manager.db_path}")
        
        # 컬렉션 초기화
        self._collections: Dict[str, Any] = {}
        self._langchain_chromas: Dict[str, Chroma] = {}
        self._initialize_collections()
        
        # 초기화 완료 플래그 설정
        VectorStore._initialized = True
    
    def _initialize_collections(self):
        """데이터베이스 컬렉션들 초기화"""
        collection_names = [
            "company_culture",     # company_culture 전용
            "work_life_balance",   # work_life_balance 전용
            "management",          # management 전용
            "salary_benefits",     # salary_benefits 전용
            "career_growth",       # career_growth 전용
        ]
        
        for name in collection_names:
            try:
                # 기존 컬렉션 가져오기 또는 새로 생성
                collection = self.chroma_client.get_or_create_collection(
                    name=name,
                    metadata={"description": f"BlindInsight {name} collection", "hnsw:space": "cosine"}
                )
                self._collections[name] = collection
                
                # LangChain Chroma 래퍼 생성
                langchain_chroma = Chroma(
                    client=self.chroma_client,
                    collection_name=name,
                    embedding_function=self.embedding_manager.embedding_function
                )
                self._langchain_chromas[name] = langchain_chroma
                
                print(f"컬렉션 '{name}' 및 LangChain 래퍼 초기화 완료")
                
            except Exception as e:
                print(f"컬렉션 '{name}' 초기화 실패: {str(e)}")
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """메타데이터 타입 검증 및 정리"""
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool, type(None))):  # bool 포함
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized
    
    async def add_documents(
        self, 
        documents: List[DocumentEmbedding], 
        collection_name: str
    ) -> bool:
        """
        문서들을 벡터 저장소에 추가
        
        Args:
            documents: 추가할 문서 임베딩 리스트
            collection_name: 저장할 컬렉션명
            
        Returns:
            성공 여부
        """
        if not documents:
            return True
            
        # 배치 최적화된 저장 호출
        success = await self.add_documents_batch_optimized(documents, collection_name)
        
        # 성공 시 메타데이터 추출 및 저장
        if success:
            try:
                self.add_company_metadata_from_documents(documents)
                print(f"메타데이터 추출 및 저장 완료: {len(documents)}개 문서")
            except Exception as e:
                print(f"메타데이터 추출 실패: {str(e)}")
        
        return success
    
    async def add_documents_batch_optimized(
    self, 
    documents: List[DocumentEmbedding], 
    collection_name: str,
    batch_size: int = 500
    ) -> bool:
        """
        최적화된 배치 문서 저장 (메모리 효율성 개선)
        
        Args:
            documents: 추가할 문서 임베딩 리스트
            collection_name: 저장할 컬렉션명
            batch_size: ChromaDB 저장 배치 크기
            
        Returns:
            성공 여부
        """
        if collection_name not in self._collections:
            print(f"존재하지 않는 컬렉션: {collection_name}")
            return False
        
        if not documents:
            return True
        
        try:
            collection = self._collections[collection_name]
            total_docs = len(documents)
            
            print(f"[저장] {collection_name} 컬렉션에 {total_docs}개 문서를 {batch_size}개씩 배치 저장")
            
            # 배치 단위로 처리
            success_count = 0
            for i in range(0, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                batch_docs = documents[i:batch_end]
                
                print(f"  배치 {i//batch_size + 1}/{(total_docs-1)//batch_size + 1}: {len(batch_docs)}개 저장 중...")
                
                # 배치 데이터 준비
                batch_ids = [doc.document_id for doc in batch_docs]
                batch_embeddings = [doc.embedding for doc in batch_docs]
                batch_metadatas = [self._sanitize_metadata(doc.metadata) for doc in batch_docs]
                batch_texts = [doc.content for doc in batch_docs]
                
                try:
                    # 기존 문서 삭제 (중복 방지)
                    try:
                        existing = collection.get(ids=batch_ids)
                        if existing['ids']:
                            collection.delete(ids=existing['ids'])
                    except:
                        pass
                    
                    # 배치 추가
                    collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                        documents=batch_texts
                    )
                    
                    success_count += len(batch_docs)
                    print(f"    성공: {len(batch_docs)}개 저장 완료")
                    
                    # 메모리 정리를 위한 짧은 대기
                    if batch_end < total_docs:
                        await asyncio.sleep(0.05)
                        
                except Exception as e:
                    print(f"    배치 저장 실패: {str(e)}")
                    # 개별 저장으로 fallback
                    individual_success = await self._fallback_individual_save(batch_docs, collection)
                    success_count += individual_success
            
            print(f"[완료] {success_count}/{total_docs}개 문서 저장 완료")
            return success_count > 0
            
        except Exception as e:
            print(f"배치 문서 저장 실패: {str(e)}")
            return False

    async def _fallback_individual_save(self, documents: List[DocumentEmbedding], collection) -> int:
        """개별 저장 fallback (성공한 문서 수 반환)"""
        success_count = 0
        for doc in documents:
            try:
                collection.add(
                    ids=[doc.document_id],
                    embeddings=[doc.embedding],
                    metadatas=[self._sanitize_metadata(doc.metadata)],
                    documents=[doc.content]
                )
                success_count += 1
            except Exception as e:
                print(f"      개별 저장 실패 {doc.document_id}: {str(e)}")
        
        return success_count
    
    def get_langchain_chroma(self, collection_name: str):
        """
        LangChain Chroma 래퍼 반환
        
        Args:
            collection_name: 컬렉션명
            
        Returns:
            LangChain Chroma 인스턴스 또는 None
        """
        return self._langchain_chromas.get(collection_name)
    
    async def _batch_save_to_chromadb(
        self, 
        documents: List[DocumentEmbedding], 
        collection, 
        batch_size: int
    ) -> bool:
        """ChromaDB 배치 저장"""
        
        total_docs = len(documents)
        print(f"ChromaDB 배치 저장: {total_docs}개 문서")
        
        # 배치 단위로 처리
        for i in range(0, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            batch_docs = documents[i:batch_end]
            
            # 배치 데이터 준비
            batch_ids = [doc.document_id for doc in batch_docs]
            batch_embeddings = [doc.embedding for doc in batch_docs]
            batch_metadatas = [self._sanitize_metadata(doc.metadata) for doc in batch_docs]
            batch_texts = [doc.content for doc in batch_docs]
            
            try:
                # 기존 문서 일괄 삭제 (중복 방지)
                try:
                    existing = collection.get(ids=batch_ids)
                    if existing['ids']:
                        collection.delete(ids=existing['ids'])
                except:
                    pass
                
                # 배치 추가
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts
                )
                
                print(f"  배치 {i//batch_size + 1}: {len(batch_docs)}개 저장 완료")
                
                # 메모리 정리를 위한 짧은 대기
                if batch_end < total_docs:
                    await asyncio.sleep(0.05)
                    
            except Exception as e:
                print(f"배치 {i//batch_size + 1} 저장 실패: {str(e)}")
                # 개별 저장으로 fallback
                await self._fallback_individual_save(batch_docs, collection)
        
        return True
    
    async def _fallback_individual_save(self, documents: List[DocumentEmbedding], collection):
        """개별 저장 fallback"""
        for doc in documents:
            try:
                collection.add(
                    ids=[doc.document_id],
                    embeddings=[doc.embedding],
                    metadatas=[self._sanitize_metadata(doc.metadata)],
                    documents=[doc.content]
                )
            except Exception as e:
                print(f"개별 저장 실패 {doc.document_id}: {str(e)}")
    
    async def search_similar_documents(
        self,
        query: str,
        collection_name: str = "general",
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        유사한 문서들 검색
        
        Args:
            query: 검색 쿼리
            collection_name: 검색할 컬렉션명
            k: 반환할 문서 수
            filter_dict: 메타데이터 필터 (예: {"company": "네이버"})
            
        Returns:
            검색 결과 리스트 (거리 점수 포함)
        """
        print(f"[VectorStore] 검색 요청 - Collection: {collection_name}, Query: {query[:50]}..., k: {k}")
        print(f"[VectorStore] 필터: {filter_dict}")
        
        if collection_name not in self._collections:
            print(f"[VectorStore] 존재하지 않는 컬렉션: {collection_name}")
            return []
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = await self.embedding_manager.create_embedding(query)
            print(f"[VectorStore] 임베딩 생성 완료 (차원: {len(query_embedding)})")
            
            collection = self._collections[collection_name]
            collection_count = collection.count()
            print(f"[VectorStore] 컬렉션 '{collection_name}' 문서 수: {collection_count}")
            
            # ChromaDB WHERE 절 문법에 맞게 필터 변환
            where_clause = None
            if filter_dict and len(filter_dict) > 0:
                if len(filter_dict) == 1:
                    key, value = next(iter(filter_dict.items()))
                    where_clause = {key: {"$eq": value}}
                else:
                    conditions = []
                    for key, value in filter_dict.items():
                        conditions.append({key: {"$eq": value}})
                    where_clause = {"$and": conditions}
            
            print(f"[VectorStore] ChromaDB where 절: {where_clause}")
            
            # 유사도 검색 실행
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            print(f"[VectorStore] ChromaDB 응답 - documents: {len(results.get('documents', [[]])[0]) if results.get('documents') else 0}개")
            
            # 결과 포맷팅
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # 코사인 거리 사용 (0에 가까울수록 더 유사함)
                    cosine_distance = max(0.0, min(2.0, distance))
                    # 거리를 유사도로 변환 (0~1, 높을수록 유사)
                    similarity_score = max(0.0, 1.0 - (cosine_distance / 2.0))
                    
                    formatted_result = {
                        "rank": i + 1,
                        "content": doc,
                        "metadata": metadata,
                        "distance_score": similarity_score,  # 유사도 점수 (높을수록 좋음)
                        "raw_distance": distance
                    }
                    formatted_results.append(formatted_result)
            
            print(f"[VectorStore] 최종 포맷팅된 결과: {len(formatted_results)}개")
            return formatted_results
            
        except Exception as e:
            print(f"문서 검색 실패: {str(e)}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return []
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 통계 정보 반환"""
        if collection_name not in self._collections:
            return {}
        
        try:
            collection = self._collections[collection_name]
            count = collection.count()
            
            return {
                "collection_name": collection_name,
                "document_count": count,
                "last_updated": datetime.now().isoformat(),
                "langchain_chroma_available": collection_name in self._langchain_chromas
            }
            
        except Exception as e:
            print(f"통계 조회 실패: {str(e)}")
            return {}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """전체 시스템 통계 조회"""
        stats = {
            "vector_store": {
                "persist_directory": self.persist_directory,
                "total_collections": len(self._collections),
                "collection_stats": {}
            },
            "embedding_manager": self.embedding_manager.get_cache_stats(),
        }
        
        # 각 컬렉션 통계
        total_documents = 0
        for collection_name in self._collections.keys():
            collection_stats = self.get_collection_stats(collection_name)
            stats["vector_store"]["collection_stats"][collection_name] = collection_stats
            total_documents += collection_stats.get("document_count", 0)
        
        stats["vector_store"]["total_documents"] = total_documents
        
        return stats
    
    
    def add_company_metadata_from_documents(self, documents: List[DocumentEmbedding]):
        """
        문서들에서 회사 메타데이터를 추출하여 별도 DB에 저장
        
        Args:
            documents: 메타데이터를 추출할 문서들
        """
        company_metadata = {}
        
        for doc in documents:
            metadata = doc.metadata
            company = metadata.get('company', '').strip()
            position = metadata.get('position', '').strip()
            
            # 연도 추출
            year = None
            review_date = metadata.get('review_date', '')
            if review_date and len(str(review_date)) >= 4:
                try:
                    if '-' in str(review_date):
                        year = int(str(review_date)[:4])
                    else:
                        year_candidate = int(str(review_date)[:4])
                        if 1900 <= year_candidate <= 2030:
                            year = year_candidate
                except (ValueError, TypeError):
                    pass
            
            if company:
                if company not in company_metadata:
                    company_metadata[company] = {
                        'positions': set(),
                        'years': set()
                    }
                
                if position and position != 'unknown':
                    company_metadata[company]['positions'].add(position)
                
                if year:
                    company_metadata[company]['years'].add(year)
        
        # 메타데이터 DB에 저장
        for company, data in company_metadata.items():
            try:
                self.metadata_manager.add_company_metadata(
                    company_name=company,
                    positions=data['positions'],
                    years=data['years']
                )
                print(f"메타데이터 저장 완료 - {company}: 직무 {len(data['positions'])}개, 연도 {len(data['years'])}개")
            except Exception as e:
                print(f"메타데이터 저장 실패 ({company}): {str(e)}")
    
    def get_companies_from_metadata(self) -> List[str]:
        """메타데이터 DB에서 회사 목록 조회"""
        return self.metadata_manager.get_all_companies()
    
    def get_positions_from_metadata(self, company_name: str = None) -> List[str]:
        """메타데이터 DB에서 직무 목록 조회"""
        return self.metadata_manager.get_positions_for_company(company_name)
    
    def get_years_from_metadata(self, company_name: str = None) -> List[int]:
        """메타데이터 DB에서 연도 목록 조회"""
        return self.metadata_manager.get_years_for_company(company_name)
    
    def extract_metadata_from_existing_collections_DEPRECATED(self):
        """
        DEPRECATED: 이 메서드는 더 이상 사용되지 않습니다.
        migrate_reviews.py에서 마이그레이션 시 자동으로 메타데이터가 수집됩니다.
        """
        print("[경고] extract_metadata_from_existing_collections는 deprecated 되었습니다.")
        print("migrate_reviews.py를 실행하여 새로운 방식으로 메타데이터를 수집하세요.")
        return 0