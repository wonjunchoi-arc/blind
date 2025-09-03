"""
임베딩 관리 및 벡터 데이터베이스 시스템

OpenAI 임베딩을 사용하여 텍스트를 벡터로 변환하고
ChromaDB를 통해 효율적인 벡터 검색을 제공합니다.
"""

#embeddings.py

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

# HTTP 요청 로그 레벨 조정
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

import numpy as np
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from ..models.base import BaseModel, settings


class CompanyMetadataManager:
    """
    회사별 메타데이터 관리 클래스
    
    ChromaDB와 같은 디렉토리의 SQLite 데이터베이스에 
    회사명, 직무, 연도 등의 메타데이터를 별도 저장하고 관리합니다.
    """
    
    def __init__(self, db_path: str = None):
        """
        메타데이터 관리자 초기화
        
        Args:
            db_path: SQLite 데이터베이스 경로
        """
        if db_path is None:
            # ChromaDB와 같은 경로에 메타데이터 DB 생성
            chroma_dir = settings.vector_db_path
            self.db_path = os.path.join(chroma_dir, "company_metadata.db")
        else:
            self.db_path = db_path
        
        # 데이터베이스 초기화
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 스키마 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 회사 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS companies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_name TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # 직무 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_id INTEGER,
                        position_name TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (company_id) REFERENCES companies (id),
                        UNIQUE(company_id, position_name)
                    )
                ''')
                
                # 연도 테이블
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS years (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        company_id INTEGER,
                        year INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (company_id) REFERENCES companies (id),
                        UNIQUE(company_id, year)
                    )
                ''')
                
                # 인덱스 생성
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_companies_name ON companies(company_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_company ON positions(company_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_years_company ON years(company_id)')
                
                conn.commit()
                
        except Exception as e:
            print(f"데이터베이스 초기화 실패: {str(e)}")
            raise
    
    def add_company_metadata(self, company_name: str, positions: Set[str], years: Set[int]):
        """
        회사의 메타데이터 추가/업데이트
        
        Args:
            company_name: 회사명
            positions: 직무 목록
            years: 연도 목록
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 회사 추가/업데이트
                cursor.execute('''
                    INSERT OR IGNORE INTO companies (company_name) VALUES (?)
                ''', (company_name,))
                
                cursor.execute('''
                    UPDATE companies SET updated_at = CURRENT_TIMESTAMP WHERE company_name = ?
                ''', (company_name,))
                
                # 회사 ID 조회
                cursor.execute('SELECT id FROM companies WHERE company_name = ?', (company_name,))
                company_id = cursor.fetchone()[0]
                
                # 직무 추가
                for position in positions:
                    if position and position.strip():
                        cursor.execute('''
                            INSERT OR IGNORE INTO positions (company_id, position_name) VALUES (?, ?)
                        ''', (company_id, position.strip()))
                
                # 연도 추가
                for year in years:
                    if year and isinstance(year, int) and year > 1900:
                        cursor.execute('''
                            INSERT OR IGNORE INTO years (company_id, year) VALUES (?, ?)
                        ''', (company_id, year))
                
                conn.commit()
                
        except Exception as e:
            print(f"회사 메타데이터 추가 실패 ({company_name}): {str(e)}")
            raise
    
    def get_all_companies(self) -> List[str]:
        """모든 회사명 조회"""
        try:
            import logging
            logger = logging.getLogger(__name__)
            
            logger.info(f"SQLite DB 경로: {self.db_path}")
            logger.info(f"DB 파일 존재: {os.path.exists(self.db_path)}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT company_name FROM companies ORDER BY company_name')
                companies = [row[0] for row in cursor.fetchall()]
                
                logger.info(f"조회된 회사 수: {len(companies)}")
                for i, company in enumerate(companies):
                    logger.info(f"  {i+1}. {repr(company)}")
                
                return companies
                
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"회사 목록 조회 실패: {str(e)}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return []
    
    def get_positions_for_company(self, company_name: str = None) -> List[str]:
        """특정 회사의 직무 목록 조회 (None이면 모든 직무)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if company_name:
                    cursor.execute('''
                        SELECT DISTINCT p.position_name 
                        FROM positions p 
                        JOIN companies c ON p.company_id = c.id 
                        WHERE c.company_name = ? 
                        ORDER BY p.position_name
                    ''', (company_name,))
                else:
                    cursor.execute('SELECT DISTINCT position_name FROM positions ORDER BY position_name')
                
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"직무 목록 조회 실패: {str(e)}")
            return []
    
    def get_years_for_company(self, company_name: str = None) -> List[int]:
        """특정 회사의 연도 목록 조회 (None이면 모든 연도)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if company_name:
                    cursor.execute('''
                        SELECT DISTINCT y.year 
                        FROM years y 
                        JOIN companies c ON y.company_id = c.id 
                        WHERE c.company_name = ? 
                        ORDER BY y.year DESC
                    ''', (company_name,))
                else:
                    cursor.execute('SELECT DISTINCT year FROM years ORDER BY year DESC')
                
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"연도 목록 조회 실패: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """메타데이터 통계 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 회사 수
                cursor.execute('SELECT COUNT(*) FROM companies')
                company_count = cursor.fetchone()[0]
                
                # 전체 직무 수
                cursor.execute('SELECT COUNT(DISTINCT position_name) FROM positions')
                position_count = cursor.fetchone()[0]
                
                # 전체 연도 수
                cursor.execute('SELECT COUNT(DISTINCT year) FROM years')
                year_count = cursor.fetchone()[0]
                
                return {
                    "companies": company_count,
                    "unique_positions": position_count,
                    "unique_years": year_count,
                    "database_path": self.db_path
                }
        except Exception as e:
            print(f"통계 조회 실패: {str(e)}")
            return {}


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
    
    OpenAI의 text-embedding-3-small 모델을 사용하여
    한국어 텍스트를 고품질 벡터로 변환합니다.
    """
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        임베딩 관리자 초기화
        
        Args:
            model_name: 사용할 OpenAI 임베딩 모델명
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
            show_progress_bar=True
        )
        
        self.model_name = model_name
        self.dimensions = dimensions
        
        # LangChain Chroma용 임베딩 함수
        self.embedding_function = self.embedding_model
        
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
            return [0.0] * self.dimensions
        
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
            return [0.0] * self.dimensions
    
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
    
    싱글턴 패턴을 사용하여 ChromaDB 클라이언트 중복 생성을 방지합니다.
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
        
        # ChromaDB 클라이언트 초기화 (성능 최적화)
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
        
        # 회사 메타데이터 관리자 초기화
        self.metadata_manager = CompanyMetadataManager()
        print(f"CompanyMetadataManager 초기화 완료: {self.metadata_manager.db_path}")
        
        # 컬렉션 초기화
        self._collections: Dict[str, Any] = {}
        self._langchain_chromas: Dict[str, Chroma] = {}  # LangChain Chroma 래퍼들
        self._initialize_collections()
        
        # 초기화 완료 플래그 설정
        VectorStore._initialized = True
    
    def _initialize_collections(self):
        """데이터베이스 컬렉션들 초기화"""
        # 새로운 6개 카테고리에 맞춘 컬렉션 생성
        collection_names = [
            "company_culture",     # company_culture 전용
            "work_life_balance",   # work_life_balance 전용
            "management",          # management 전용
            "salary_benefits",     # salary_benefits 전용
            "career_growth",       # career_growth 전용
            "general"              # general 전용
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
            
            # ChromaDB에 문서들 추가 (배치 처리)
            print(f"ChromaDB에 {len(documents)}개 문서 추가 중...")
            
            batch_size = 50  # 배치 크기 설정
            total_docs = len(documents)
            
            for i in range(0, total_docs, batch_size):
                batch_end = min(i + batch_size, total_docs)
                batch_docs = documents[i:batch_end]
                
                print(f"  배치 {i//batch_size + 1}: {len(batch_docs)}개 문서 처리 중... ({i+1}-{batch_end}/{total_docs})")
                
                # 배치별 데이터 준비
                ids = [doc.document_id for doc in batch_docs]
                embeddings = [doc.embedding for doc in batch_docs]
                metadatas = [doc.metadata for doc in batch_docs]
                documents_text = [doc.content for doc in batch_docs]
                
                # ChromaDB에 배치 추가 (디버깅 정보 추가)
                try:
                    # 데이터 검증
                    print(f"    디버그: 첫 번째 ID = {ids[0] if ids else 'None'}")
                    print(f"    디버그: 임베딩 차원 = {len(embeddings[0]) if embeddings and embeddings[0] else 'None'}")
                    print(f"    디버그: 첫 번째 메타데이터 키 = {list(metadatas[0].keys()) if metadatas else 'None'}")
                    
                    # 메타데이터 타입 검증
                    for i, metadata in enumerate(metadatas):
                        for key, value in metadata.items():
                            if not isinstance(value, (str, int, float, bool, type(None))):
                                print(f"    오류: 메타데이터[{i}]['{key}'] = {type(value)} (값: {value})")
                                raise ValueError(f"잘못된 메타데이터 타입: {key} = {type(value)}")
                    
                    print(f"    ChromaDB collection.add() 호출 중...")
                    
                    # 개별 문서 단위로 추가 (문제 해결 시도)
                    for j, (doc_id, embedding, metadata, text) in enumerate(zip(ids, embeddings, metadatas, documents_text)):
                        try:
                            print(f"      문서 {j+1}/{len(ids)} 추가: {doc_id[:30]}...")
                            
                            # 기존 문서 확인 및 삭제 (중복 방지)
                            try:
                                existing = collection.get(ids=[doc_id])
                                if existing['ids']:
                                    print(f"      기존 문서 삭제: {doc_id[:30]}...")
                                    collection.delete(ids=[doc_id])
                            except:
                                pass  # 문서가 존재하지 않으면 무시
                            
                            # 새 문서 추가
                            # 실행할 함수(collection.add)와 그 인자(ids, embeddings 등)를 분리해서 전달합니다.
                            # 수정 코드
                            collection.add(
                            ids=[doc_id],
                            embeddings=[embedding],
                            metadatas=[metadata],
                            documents=[text]
                           )
                            print(f"      문서 {j+1} 완료")
                            
                            # 개별 문서 간 작은 지연
                            await asyncio.sleep(0.01)
                            
                        except Exception as single_add_error:
                            print(f"      문서 {j+1} 추가 실패: {str(single_add_error)}")
                            # 개별 문서 실패해도 계속 진행
                            continue
                    
                    print(f"    ChromaDB collection.add() 완료!")
                except Exception as add_error:
                    print(f"    ChromaDB add 오류: {str(add_error)}")
                    print(f"    오류 타입: {type(add_error)}")
                    raise add_error
                
                print(f"  배치 {i//batch_size + 1} 완료 ({len(batch_docs)}개 문서)")
                
                # 배치 간 짧은 지연 (메모리 정리 시간)
                if batch_end < total_docs:  # 마지막 배치가 아닌 경우에만
                    await asyncio.sleep(0.1)
            
            print(f"✅ {total_docs}개 문서를 '{collection_name}' 컬렉션에 추가 완료")
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
        print(f"[VectorStore] 검색 요청 - Collection: {collection_name}, Query: {query[:50]}..., k: {k}")
        print(f"[VectorStore] 필터: {filter_dict}")
        print(f"[VectorStore] 사용 가능한 컬렉션: {list(self._collections.keys())}")
        
        if collection_name not in self._collections:
            print(f"[VectorStore] 존재하지 않는 컬렉션: {collection_name}")
            return []
        
        try:
            # 쿼리 임베딩 생성
            print(f"[VectorStore] 쿼리 임베딩 생성 중...")
            query_embedding = await self.embedding_manager.create_embedding(query)
            print(f"[VectorStore] 임베딩 생성 완료 (차원: {len(query_embedding)})")
            
            collection = self._collections[collection_name]
            collection_count = collection.count()
            print(f"[VectorStore] 컬렉션 '{collection_name}' 문서 수: {collection_count}")
            
            # ChromaDB WHERE 절 문법에 맞게 필터 변환
            where_clause = None
            if filter_dict and len(filter_dict) > 0:
                if len(filter_dict) == 1:
                    # 단일 조건인 경우: {"field": {"$eq": "value"}}
                    key, value = next(iter(filter_dict.items()))
                    where_clause = {key: {"$eq": value}}
                else:
                    # 다중 조건인 경우: {"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$eq": "value2"}}]}
                    conditions = []
                    for key, value in filter_dict.items():
                        conditions.append({key: {"$eq": value}})
                    where_clause = {"$and": conditions}
            
            print(f"[VectorStore] ChromaDB where 절: {where_clause}")
            
            # 유사도 검색 실행
            print(f"[VectorStore] ChromaDB 검색 실행 중...")
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,  # 변환된 필터 적용
                include=["documents", "metadatas", "distances"]
            )
            
            print(f"[VectorStore] ChromaDB 응답 - documents: {len(results.get('documents', [[]])[0]) if results.get('documents') else 0}개")
            
            # 결과 포맷팅
            formatted_results = []
            print(f"[VectorStore] 결과 포맷팅 시작...")
            print(f"[VectorStore] results 구조: {type(results)}")
            print(f"[VectorStore] results.keys(): {results.keys() if isinstance(results, dict) else 'Not dict'}")
            
            if results.get('documents'):
                print(f"[VectorStore] documents 길이: {len(results['documents'])}")
                if results['documents'][0]:
                    print(f"[VectorStore] 첫 번째 documents 길이: {len(results['documents'][0])}")
            
            if results['documents'] and results['documents'][0]:
                print(f"[VectorStore] zip 처리 시작...")
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # 코사인 거리 사용 (0에 가까울수록 더 유사함)
                    # 거리 값을 0~2 범위로 클리핑
                    cosine_distance = max(0.0, min(2.0, distance))
                    formatted_result = {
                        "rank": i + 1,
                        "content": doc,
                        "metadata": metadata,
                        "distance_score": cosine_distance,  # 거리 점수 (낮을수록 좋음)
                        "raw_distance": distance
                    }
                    formatted_results.append(formatted_result)
                    print(f"[VectorStore] 결과 {i+1}: distance={cosine_distance:.3f}, content={doc[:50]}...")
            else:
                print(f"[VectorStore] 결과 포맷팅 조건 불만족 - documents 존재: {bool(results.get('documents'))}, 첫번째 존재: {bool(results.get('documents') and results['documents'][0])}")
            
            print(f"[VectorStore] 최종 포맷팅된 결과: {len(formatted_results)}개")
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
                "last_updated": datetime.now().isoformat(),
                "langchain_chroma_available": collection_name in self._langchain_chromas
            }
            
        except Exception as e:
            print(f"통계 조회 실패: {str(e)}")
            return {}
    
    def get_langchain_chroma(self, collection_name: str) -> Optional['Chroma']:
        """LangChain Chroma 래퍼 반환"""
        return self._langchain_chromas.get(collection_name)
    
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
    
    async def get_all_metadata(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        컬렉션의 모든 메타데이터 조회
        
        Args:
            collection_name: 조회할 컬렉션명
            
        Returns:
            메타데이터 리스트
        """
        if collection_name not in self._collections:
            return []
        
        try:
            collection = self._collections[collection_name]
            
            # 모든 문서의 메타데이터만 조회 (성능 최적화)
            all_data = collection.get(
                include=["metadatas"]
            )
            
            if all_data and "metadatas" in all_data:
                return all_data["metadatas"]
            else:
                return []
                
        except Exception as e:
            print(f"메타데이터 조회 실패 [{collection_name}]: {str(e)}")
            return []
    
    async def search_metadata(
        self, 
        collection_name: str, 
        filter_dict: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        메타데이터 필터링으로 문서 검색
        
        Args:
            collection_name: 검색할 컬렉션명
            filter_dict: 필터 조건
            
        Returns:
            조건에 맞는 메타데이터 리스트
        """
        if collection_name not in self._collections:
            return []
        
        try:
            collection = self._collections[collection_name]
            
            # ChromaDB의 where 필터 사용
            filtered_data = collection.get(
                where=filter_dict,
                include=["metadatas"]
            )
            
            if filtered_data and "metadatas" in filtered_data:
                return filtered_data["metadatas"]
            else:
                return []
                
        except Exception as e:
            print(f"메타데이터 필터 검색 실패 [{collection_name}]: {str(e)}")
            return []
    
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
            
            # 연도 추출 (review_date에서)
            year = None
            review_date = metadata.get('review_date', '')
            if review_date and len(str(review_date)) >= 4:
                try:
                    # 날짜에서 연도 추출 (YYYY-MM-DD 또는 YYYY 형식)
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