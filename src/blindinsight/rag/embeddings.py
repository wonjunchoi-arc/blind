"""
임베딩 관리 및 벡터 데이터베이스 시스템

OpenAI 임베딩을 사용하여 텍스트를 벡터로 변환하고
ChromaDB를 통해 효율적인 벡터 검색을 제공합니다.
"""

import os
import json
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

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
    임베딩 생성 및 관리를 담당하는 클래스
    
    OpenAI의 text-embedding-3-large 모델을 사용하여
    한국어 텍스트를 고품질 벡터로 변환합니다.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-large"):
        """
        임베딩 관리자 초기화
        
        Args:
            model_name: 사용할 OpenAI 임베딩 모델명
        """
        # OpenAI 임베딩 모델 초기화
        self.embedding_model = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.openai_api_key,
            dimensions=settings.embedding_dimensions,  # 3072 차원
            show_progress_bar=True
        )
        
        # 임베딩 캐시 (메모리 효율성을 위해)
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    async def create_embedding(self, text: str) -> List[float]:
        """
        텍스트를 벡터 임베딩으로 변환
        
        Args:
            text: 임베딩할 텍스트
            
        Returns:
            벡터 임베딩 리스트
        """
        # 빈 텍스트 처리
        if not text or not text.strip():
            return [0.0] * settings.embedding_dimensions
        
        # 텍스트 정규화 (공백 제거, 소문자 변환)
        normalized_text = text.strip()
        
        # 캐시 키 생성 (해시 기반)
        cache_key = hashlib.md5(normalized_text.encode('utf-8')).hexdigest()
        
        # 캐시에서 확인
        if cache_key in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[cache_key]
        
        try:
            # OpenAI API를 통해 임베딩 생성
            embedding = await self.embedding_model.aembed_query(normalized_text)
            
            # 캐시에 저장 (메모리 제한 고려)
            if len(self._embedding_cache) < 1000:  # 최대 1000개 캐시
                self._embedding_cache[cache_key] = embedding
            
            self._cache_misses += 1
            return embedding
            
        except Exception as e:
            print(f"임베딩 생성 실패: {str(e)}")
            # 기본값 반환 (영벡터)
            return [0.0] * settings.embedding_dimensions
    
    async def create_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트를 한 번에 임베딩으로 변환 (배치 처리)
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            벡터 임베딩 리스트들의 리스트
        """
        if not texts:
            return []
        
        try:
            # 배치 임베딩 생성 (API 호출 최적화)
            embeddings = await self.embedding_model.aembed_documents(texts)
            return embeddings
            
        except Exception as e:
            print(f"배치 임베딩 생성 실패: {str(e)}")
            # 개별 처리로 fallback
            embeddings = []
            for text in texts:
                embedding = await self.create_embedding(text)
                embeddings.append(embedding)
            return embeddings
    
    def create_document_embedding(
        self, 
        content: str, 
        metadata: Dict[str, Any]
    ) -> DocumentEmbedding:
        """
        문서 임베딩 객체 생성
        
        Args:
            content: 문서 내용
            metadata: 메타데이터 (회사명, 카테고리 등)
            
        Returns:
            DocumentEmbedding 객체
        """
        # 문서 ID 생성 (메타데이터 기반)
        doc_id_source = f"{metadata.get('company', '')}-{metadata.get('category', '')}-{content[:100]}"
        document_id = hashlib.sha256(doc_id_source.encode('utf-8')).hexdigest()[:16]
        
        # 내용 해시값 생성 (중복 검사용)
        hash_value = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        return DocumentEmbedding(
            document_id=document_id,
            content=content,
            embedding=[],  # 나중에 설정
            metadata=metadata,
            created_at=datetime.now(),
            hash_value=hash_value
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """임베딩 캐시 통계 반환"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self._embedding_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


class VectorStore:
    """
    벡터 데이터베이스 관리 클래스
    
    ChromaDB를 사용하여 문서 임베딩을 저장하고
    유사도 기반 검색을 제공합니다.
    """
    
    def __init__(self, persist_directory: str = None):
        """
        벡터 스토어 초기화
        
        Args:
            persist_directory: 데이터베이스 저장 경로
        """
        self.persist_directory = persist_directory or settings.vector_db_path
        
        # ChromaDB 클라이언트 초기화
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # 텔레메트리 비활성화
                is_persistent=True           # 영구 저장 활성화
            )
        )
        
        # 임베딩 함수 설정
        self.embedding_manager = EmbeddingManager()
        
        # 컬렉션 초기화
        self._collections: Dict[str, Any] = {}
        self._initialize_collections()
    
    def _initialize_collections(self):
        """데이터베이스 컬렉션들 초기화"""
        # 카테고리별 컬렉션 생성
        collection_names = [
            "culture_reviews",     # 기업문화 리뷰
            "salary_discussions",  # 연봉 정보
            "career_advice",       # 커리어 조언
            "interview_reviews",   # 면접 후기
            "company_general"      # 일반 회사 정보
        ]
        
        for name in collection_names:
            try:
                # 기존 컬렉션 가져오기 또는 새로 생성
                collection = self.chroma_client.get_or_create_collection(
                    name=name,
                    metadata={"description": f"BlindInsight {name} collection"}
                )
                self._collections[name] = collection
                print(f"컬렉션 '{name}' 초기화 완료")
                
            except Exception as e:
                print(f"컬렉션 '{name}' 초기화 실패: {str(e)}")
    
    async def add_documents(
        self, 
        documents: List[DocumentEmbedding], 
        collection_name: str = "company_general"
    ) -> bool:
        """
        문서들을 벡터 데이터베이스에 추가
        
        Args:
            documents: 추가할 문서 임베딩 리스트
            collection_name: 저장할 컬렉션명
            
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
            
            # 임베딩이 없는 문서들의 임베딩 생성
            texts_to_embed = []
            docs_needing_embedding = []
            
            for doc in documents:
                if not doc.embedding:
                    texts_to_embed.append(doc.content)
                    docs_needing_embedding.append(doc)
            
            if texts_to_embed:
                print(f"{len(texts_to_embed)}개 문서의 임베딩 생성 중...")
                embeddings = await self.embedding_manager.create_batch_embeddings(texts_to_embed)
                
                # 생성된 임베딩을 문서에 할당
                for doc, embedding in zip(docs_needing_embedding, embeddings):
                    doc.embedding = embedding
            
            # ChromaDB에 문서들 추가
            ids = [doc.document_id for doc in documents]
            embeddings = [doc.embedding for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            documents_text = [doc.content for doc in documents]
            
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_text
            )
            
            print(f"{len(documents)}개 문서를 '{collection_name}' 컬렉션에 추가 완료")
            return True
            
        except Exception as e:
            print(f"문서 추가 실패: {str(e)}")
            return False
    
    async def search_similar_documents(
        self,
        query: str,
        collection_name: str = "company_general",
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
        if collection_name not in self._collections:
            print(f"존재하지 않는 컬렉션: {collection_name}")
            return []
        
        try:
            # 쿼리 임베딩 생성
            query_embedding = await self.embedding_manager.create_embedding(query)
            
            collection = self._collections[collection_name]
            
            # 유사도 검색 실행
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_dict,  # 메타데이터 필터 적용
                include=["documents", "metadatas", "distances"]
            )
            
            # 결과 포맷팅
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        "rank": i + 1,
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,  # 거리를 유사도로 변환
                        "distance": distance
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"문서 검색 실패: {str(e)}")
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
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"통계 조회 실패: {str(e)}")
            return {}
    
    def delete_documents(
        self, 
        document_ids: List[str], 
        collection_name: str = "company_general"
    ) -> bool:
        """문서들 삭제"""
        if collection_name not in self._collections:
            return False
        
        try:
            collection = self._collections[collection_name]
            collection.delete(ids=document_ids)
            print(f"{len(document_ids)}개 문서 삭제 완료")
            return True
            
        except Exception as e:
            print(f"문서 삭제 실패: {str(e)}")
            return False
    
    def backup_collection(self, collection_name: str, backup_path: str) -> bool:
        """컬렉션 백업"""
        if collection_name not in self._collections:
            return False
        
        try:
            collection = self._collections[collection_name]
            
            # 모든 문서 조회
            all_data = collection.get(
                include=["documents", "metadatas", "embeddings"]
            )
            
            # JSON 파일로 저장
            backup_data = {
                "collection_name": collection_name,
                "backup_date": datetime.now().isoformat(),
                "data": all_data
            }
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
            print(f"컬렉션 '{collection_name}' 백업 완료: {backup_path}")
            return True
            
        except Exception as e:
            print(f"컬렉션 백업 실패: {str(e)}")
            return False