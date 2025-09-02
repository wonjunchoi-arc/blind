"""
JSON 청크 데이터 전용 프로세서

tools/data의 chunk_first_vectordb JSON 파일들을 처리하여
ChromaDB 벡터 데이터베이스에 최적화된 형태로 변환합니다.
"""

#json_processor.py

import json
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from langchain.schema import Document

from .embeddings import EmbeddingManager, VectorStore, DocumentEmbedding
from ..models.base import settings

# 로깅 설정
logger = logging.getLogger(__name__)


class ChunkDataProcessor:
    """
    JSON 청크 데이터 전용 프로세서
    
    tools/data의 chunk_first_vectordb JSON 파일들을 처리하여
    RAG 시스템에서 활용할 수 있도록 구조화된 Document 객체로 변환합니다.
    """
    
    def __init__(self):
        """청크 데이터 프로세서 초기화"""
        self.embedding_manager = EmbeddingManager(model_name="text-embedding-3-small")
        self.processed_count = 0
        
        # 새로운 6개 카테고리별 컬렉션 매핑 (1:1 매핑)
        self.collection_mapping = {
            "career_growth": "career_growth",
            "salary_benefits": "salary_benefits", 
            "work_life_balance": "work_life_balance",
            "company_culture": "company_culture",
            "management": "management",
            "general": "general"
        }
    
    def process_json_file(self, file_path: str) -> List[Document]:
        """
        JSON 파일을 로드하고 Document 객체들로 변환
        
        Args:
            file_path: JSON 파일 경로
            
        Returns:
            처리된 Document 객체 리스트
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict) or 'chunks' not in data:
                logger.error(f"잘못된 JSON 구조: {file_path}")
                return []
            
            # 메타데이터 추출
            metadata_info = data.get('metadata', {})
            company_name = metadata_info.get('company', 'Unknown')
            
            documents = []
            
            # 각 청크 처리
            for chunk in data['chunks']:
                chunk_docs = self._process_single_chunk(chunk, company_name)
                documents.extend(chunk_docs)
            
            logger.info(f"{file_path}에서 {len(documents)}개 문서 생성 완료")
            return documents
            
        except Exception as e:
            logger.error(f"JSON 파일 처리 실패 ({file_path}): {str(e)}")
            return []
    
    def _process_single_chunk(self, chunk: Dict[str, Any], company_name: str) -> List[Document]:
        """
        단일 청크를 Document 객체로 변환
        
        Args:
            chunk: 청크 데이터
            company_name: 회사명
            
        Returns:
            처리된 Document 객체 리스트
        """
        documents = []
        
        # 청크 메타데이터 구성
        chunk_metadata = self._extract_chunk_metadata(chunk, company_name)
        
        # 청크 내용으로 Document 생성
        content = chunk.get('content', '')
        if not content:
            return []
        
        # 카테고리에 따른 컬렉션 매핑
        category = chunk_metadata.get('category', 'general')
        collection_target = self.collection_mapping.get(category, 'company_general')
        
        doc = Document(
            page_content=content,
            metadata={
                **chunk_metadata,
                "collection_target": collection_target
            }
        )
        documents.append(doc)
        
        return documents
    
    def _extract_chunk_metadata(self, chunk: Dict[str, Any], company_name: str) -> Dict[str, Any]:
        """
        청크에서 메타데이터 추출
        
        Args:
            chunk: 청크 데이터
            company_name: 회사명
            
        Returns:
            메타데이터 딕셔너리
        """
        metadata = {
            "company": company_name,
            "source_type": "company_chunk",
            "chunk_id": chunk.get('id', 'unknown')
        }
        
        # 청크의 메타데이터에서 정보 추출
        chunk_metadata = chunk.get('metadata', {})
        if chunk_metadata:
            metadata.update({
                "category": str(chunk_metadata.get('category', 'general')),
                "content_type": str(chunk_metadata.get('content_type', 'unknown')),
                "employee_status": str(chunk_metadata.get('employee_status', 'unknown')),
                "position": str(chunk_metadata.get('position', 'unknown')),
                "review_date": str(chunk_metadata.get('review_date', 'unknown')),
                "rating_level": str(chunk_metadata.get('rating_level', 'unknown')),
                "rating_numeric": float(chunk_metadata.get('rating_numeric', 0.0)),
                "sentiment_score": float(chunk_metadata.get('sentiment_score', 0.0)),
                "sentiment_label": str(chunk_metadata.get('sentiment_label', 'neutral')),
                "content_length": int(chunk_metadata.get('content_length', 0)),
                "processing_timestamp": str(chunk_metadata.get('processing_timestamp', '')),
                "chunk_index": int(chunk_metadata.get('chunk_index', 0))
            })
        
        return metadata
    
    def _determine_collection_target(self, metadata: Dict[str, Any]) -> str:
        """
        메타데이터 기반으로 적절한 컬렉션 결정
        
        Args:
            metadata: 문서 메타데이터
            
        Returns:
            대상 컬렉션명
        """
        # 명시적 컬렉션 대상이 있으면 사용
        if metadata.get("collection_target"):
            return metadata["collection_target"]
        
        # 카테고리 기반 컬렉션 매핑
        category = metadata.get("category", "general")
        return self.collection_mapping.get(category, "company_general")


class ChunkDataLoader:
    """
    청크 데이터 로더 및 벡터 데이터베이스 저장 관리자
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        청크 데이터 로더 초기화
        
        Args:
            data_dir: 청크 데이터 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.processor = ChunkDataProcessor()
        self.vector_store = VectorStore()
        
        # 처리 통계
        self.stats = {
            "files_processed": 0,
            "documents_created": 0,
            "companies_processed": set(),
            "processing_time": 0.0
        }
    
    async def load_all_chunks(self, company_filter: Optional[List[str]] = None) -> bool:
        """
        모든 청크 JSON 파일을 로드하고 벡터 데이터베이스에 저장
        
        Args:
            company_filter: 처리할 회사 리스트 (None이면 모든 회사)
            
        Returns:
            성공 여부
        """
        start_time = datetime.now()
        
        try:
            # JSON 파일들 찾기
            json_files = list(self.data_dir.glob("*_chunk_first_vectordb.json"))
            
            if not json_files:
                logger.error(f"JSON 파일을 찾을 수 없습니다: {self.data_dir}")
                return False
            
            logger.info(f"{len(json_files)}개 JSON 파일 발견")
            
            # 회사 필터 적용
            if company_filter:
                filtered_files = []
                for file_path in json_files:
                    company_name = file_path.stem.replace('_chunk_first_vectordb', '')
                    if company_name in company_filter:
                        filtered_files.append(file_path)
                json_files = filtered_files
                logger.info(f"필터 적용 후: {len(json_files)}개 파일")
            
            # 각 파일 처리 (전체 진행상황 표시)
            print(f"총 {len(json_files)}개 회사 청크 처리 시작...")
            
            for i, file_path in enumerate(tqdm(json_files, desc="회사별 청크 처리", unit="회사")):
                company_name = file_path.stem.replace('_chunk_first_vectordb', '')
                tqdm.write(f"[{i+1}/{len(json_files)}] {company_name} 처리 중...")
                await self._process_single_file(file_path)
                tqdm.write(f"[{i+1}/{len(json_files)}] {company_name} 완료!")
            
            # 통계 업데이트
            end_time = datetime.now()
            self.stats["processing_time"] = (end_time - start_time).total_seconds()
            
            logger.info(f"전체 처리 완료: {self.stats}")
            return True
            
        except Exception as e:
            logger.error(f"전체 로딩 중 오류: {str(e)}")
            return False
    
    async def _process_single_file(self, file_path: Path) -> bool:
        """
        단일 JSON 파일 처리 및 벡터 저장소에 저장
        
        Args:
            file_path: JSON 파일 경로
            
        Returns:
            성공 여부
        """
        try:
            # 회사명 추출
            company_name = file_path.stem.replace('_chunk_first_vectordb', '')
            logger.info(f"{company_name} 처리 시작...")
            
            # JSON 파일 처리
            documents = self.processor.process_json_file(str(file_path))
            
            if not documents:
                logger.warning(f"{company_name}: 처리할 문서가 없음")
                return False
            
            # 컬렉션별로 문서 그룹화
            collection_docs = {}
            for doc in documents:
                collection_name = self.processor._determine_collection_target(doc.metadata)
                if collection_name not in collection_docs:
                    collection_docs[collection_name] = []
                collection_docs[collection_name].append(doc)
            
            # 각 컬렉션별로 벡터 저장소에 저장
            total_saved = 0
            for collection_name, docs in collection_docs.items():
                # Document를 DocumentEmbedding으로 변환
                doc_embeddings = []
                
                # 임베딩 생성 진행률 표시 (verbose=False로 HTTP 로그 숨김)
                for doc in tqdm(docs, desc=f"  └─ {collection_name} 임베딩", leave=False, disable=len(docs)<10):
                    # 임베딩 생성
                    embedding = await self.processor.embedding_manager.create_embedding(doc.page_content)
                    
                    # DocumentEmbedding 객체 생성
                    import hashlib
                    hash_value = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
                    
                    doc_embedding = DocumentEmbedding(
                        document_id=f"{company_name}_{len(doc_embeddings)}_{collection_name}",
                        content=doc.page_content,
                        embedding=embedding,
                        metadata=doc.metadata,
                        created_at=datetime.now(),
                        hash_value=hash_value
                    )
                    doc_embeddings.append(doc_embedding)
                
                # 벡터 저장소에 추가
                try:
                    print(f"    저장 중: {collection_name} ({len(doc_embeddings)}개 문서)")
                    success = await self.vector_store.add_documents(doc_embeddings, collection_name)
                    if success:
                        total_saved += len(docs)
                        logger.info(f"{company_name}: {len(docs)}개 문서 → {collection_name} 저장 완료")
                        print(f"    저장 완료: {collection_name}")
                    else:
                        logger.error(f"{company_name}: {collection_name} 저장 실패")
                        print(f"    저장 실패: {collection_name}")
                except Exception as e:
                    logger.error(f"{company_name}: {collection_name} 저장 중 예외 발생: {str(e)}")
                    print(f"    저장 예외: {collection_name} - {str(e)}")
            
            # 회사 메타데이터 추출 및 저장
            if documents:
                print(f"    메타데이터 추출 중: {company_name}")
                try:
                    # 모든 컬렉션의 doc_embeddings를 하나로 합침
                    all_doc_embeddings = []
                    for collection_name, docs in collection_docs.items():
                        for i, doc in enumerate(docs):
                            doc_embedding = DocumentEmbedding(
                                document_id=f"{company_name}_{i}_{collection_name}",
                                content=doc.page_content,
                                embedding=[],
                                metadata=doc.metadata,
                                created_at=datetime.now(),
                                hash_value=""
                            )
                            all_doc_embeddings.append(doc_embedding)
                    
                    self.vector_store.add_company_metadata_from_documents(all_doc_embeddings)
                    print(f"    메타데이터 저장 완료: {company_name}")
                except Exception as e:
                    logger.warning(f"메타데이터 저장 실패 ({company_name}): {str(e)}")
                    print(f"    메타데이터 저장 실패: {company_name} - {str(e)}")
            
            # 통계 업데이트
            self.stats["files_processed"] += 1
            self.stats["documents_created"] += total_saved
            self.stats["companies_processed"].add(company_name)
            
            logger.info(f"{company_name} 완료: {total_saved}개 문서 저장")
            return total_saved > 0
            
        except Exception as e:
            logger.error(f"{file_path} 처리 중 오류: {str(e)}")
            return False
    
    async def load_single_company(self, company_name: str) -> bool:
        """
        특정 회사의 청크 데이터만 로드
        
        Args:
            company_name: 회사명
            
        Returns:
            성공 여부
        """
        return await self.load_all_chunks(company_filter=[company_name])
    
    def get_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        return {
            **self.stats,
            "companies_processed": list(self.stats["companies_processed"])
        }
    
    async def verify_data_integrity(self) -> Dict[str, Any]:
        """
        저장된 데이터의 무결성 검증
        
        Returns:
            검증 결과
        """
        verification_result = {
            "collections_status": {},
            "total_documents": 0,
            "errors": []
        }
        
        # 각 컬렉션별 상태 확인
        collections = ["culture_reviews", "salary_discussions", "career_advice", "interview_reviews", "company_general"]
        
        for collection_name in collections:
            try:
                stats = self.vector_store.get_collection_stats(collection_name)
                verification_result["collections_status"][collection_name] = stats
                verification_result["total_documents"] += stats.get("document_count", 0)
            except Exception as e:
                verification_result["errors"].append(f"{collection_name}: {str(e)}")
        
        return verification_result