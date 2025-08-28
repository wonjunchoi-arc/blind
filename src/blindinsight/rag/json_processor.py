"""
JSON 리뷰 데이터 전용 프로세서

tools/data/reviews 폴더의 회사별 JSON 파일을 처리하여
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


class ReviewDataProcessor:
    """
    JSON 리뷰 데이터 전용 프로세서
    
    tools/data/reviews의 최적화된 JSON 파일들을 처리하여
    RAG 시스템에서 활용할 수 있도록 구조화된 Document 객체로 변환합니다.
    """
    
    def __init__(self):
        """리뷰 데이터 프로세서 초기화"""
        self.embedding_manager = EmbeddingManager(model_name="text-embedding-3-small")
        self.processed_count = 0
        
        # 카테고리별 컬렉션 매핑
        self.collection_mapping = {
            "culture": "culture_reviews",
            "salary": "salary_discussions", 
            "benefits": "salary_discussions",
            "growth": "career_advice",
            "management": "culture_reviews",
            "interview": "interview_reviews",
            "general": "company_general"
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
            
            if not isinstance(data, dict) or 'reviews' not in data:
                logger.error(f"잘못된 JSON 구조: {file_path}")
                return []
            
            # 메타데이터 추출
            metadata_info = data.get('metadata', {})
            company_name = metadata_info.get('company', 'Unknown')
            
            documents = []
            
            # 각 리뷰 처리
            for review in data['reviews']:
                review_docs = self._process_single_review(review, company_name)
                documents.extend(review_docs)
            
            logger.info(f"{file_path}에서 {len(documents)}개 문서 생성 완료")
            return documents
            
        except Exception as e:
            logger.error(f"JSON 파일 처리 실패 ({file_path}): {str(e)}")
            return []
    
    def _process_single_review(self, review: Dict[str, Any], company_name: str) -> List[Document]:
        """
        단일 리뷰를 여러 Document 객체로 변환
        
        Args:
            review: 리뷰 데이터
            company_name: 회사명
            
        Returns:
            처리된 Document 객체 리스트
        """
        documents = []
        
        # 기본 메타데이터 구성
        base_metadata = self._extract_base_metadata(review, company_name)
        
        # processed_data가 있으면 우선 사용, 없으면 raw_data 사용
        if 'processed_data' in review:
            processed_data = review['processed_data']
            
            # 의미적 내용 (content_semantic) 처리 - 주요 검색 대상
            if 'content_semantic' in processed_data:
                semantic_doc = Document(
                    page_content=processed_data['content_semantic'],
                    metadata={
                        **base_metadata,
                        "content_type": "semantic",
                        "collection_target": "culture_reviews"  # 기본적으로 문화 카테고리
                    }
                )
                documents.append(semantic_doc)
            
            # 키워드 내용 (content_keyword) 처리 - 키워드 검색용
            if 'content_keyword' in processed_data:
                keyword_doc = Document(
                    page_content=processed_data['content_keyword'], 
                    metadata={
                        **base_metadata,
                        "content_type": "keyword",
                        "collection_target": "company_general"
                    }
                )
                documents.append(keyword_doc)
        
        # raw_data 처리 (장점/단점 분리)
        if 'raw_data' in review:
            raw_data = review['raw_data']
            
            # 장점 섹션
            if '장점' in raw_data and raw_data['장점']:
                pros_content = f"장점: {raw_data['장점']}"
                pros_doc = Document(
                    page_content=pros_content,
                    metadata={
                        **base_metadata,
                        "content_type": "pros",
                        "section": "advantages", 
                        "collection_target": "culture_reviews"
                    }
                )
                documents.append(pros_doc)
            
            # 단점 섹션
            if '단점' in raw_data and raw_data['단점']:
                cons_content = f"단점: {raw_data['단점']}"
                cons_doc = Document(
                    page_content=cons_content,
                    metadata={
                        **base_metadata,
                        "content_type": "cons",
                        "section": "disadvantages",
                        "collection_target": "culture_reviews"
                    }
                )
                documents.append(cons_doc)
            
            # 제목 기반 문서 (검색 다양성을 위해)
            if '제목' in raw_data and raw_data['제목']:
                title_content = f"제목: {raw_data['제목']}"
                # 내용이 있으면 결합
                if '장점' in raw_data and raw_data['장점']:
                    title_content += f"\n주요 내용: {raw_data['장점'][:200]}..."
                
                title_doc = Document(
                    page_content=title_content,
                    metadata={
                        **base_metadata,
                        "content_type": "title",
                        "collection_target": "company_general"
                    }
                )
                documents.append(title_doc)
        
        # 키워드 기반 추가 문서 생성 (특화된 검색을 위해)
        if 'processed_data' in review and 'extracted_keywords' in review['processed_data']:
            keywords_doc = self._create_keyword_document(
                review['processed_data']['extracted_keywords'],
                base_metadata
            )
            if keywords_doc:
                documents.append(keywords_doc)
        
        return documents
    
    def _extract_base_metadata(self, review: Dict[str, Any], company_name: str) -> Dict[str, Any]:
        """
        리뷰에서 기본 메타데이터 추출
        
        Args:
            review: 리뷰 데이터
            company_name: 회사명
            
        Returns:
            기본 메타데이터 딕셔너리
        """
        metadata = {
            "company": company_name,
            "source_type": "company_review",
            "review_id": review.get('id', 'unknown')
        }
        
        # search_metadata에서 정보 추출 (타입 안전성 강화)
        if 'search_metadata' in review:
            search_meta = review['search_metadata']
            metadata.update({
                "employment_status": str(search_meta.get('employment_status', 'unknown')),
                "position": str(search_meta.get('position', 'unknown')),
                "review_date": str(search_meta.get('review_date', 'unknown')),
                "rating_level": str(search_meta.get('rating_level', 'unknown')),
                "rating_numeric": float(search_meta.get('rating_numeric', 0.0)),
                "has_worklife_mention": bool(search_meta.get('has_worklife_mention', False)),
                "has_salary_mention": bool(search_meta.get('has_salary_mention', False)),
                "has_culture_mention": bool(search_meta.get('has_culture_mention', False)),
                "has_growth_mention": bool(search_meta.get('has_growth_mention', False)),
                "has_benefits_mention": bool(search_meta.get('has_benefits_mention', False)),
                "has_management_mention": bool(search_meta.get('has_management_mention', False)),
                "review_length": int(search_meta.get('review_length', 0))
            })
        
        # raw_data에서 평점 정보 추출 (타입 안전성 강화)
        if 'raw_data' in review:
            raw_data = review['raw_data']
            metadata.update({
                "overall_rating": float(raw_data.get('총점', 0.0)),
                "career_rating": int(raw_data.get('커리어향상', 0)),
                "worklife_rating": int(raw_data.get('워라밸', 0)), 
                "salary_rating": int(raw_data.get('급여복지', 0)),
                "culture_rating": int(raw_data.get('사내문화', 0)),
                "management_rating": int(raw_data.get('경영진', 0)),
                "job_title": str(raw_data.get('직무', 'unknown')),
                "employment_type": str(raw_data.get('재직상태', 'unknown'))
            })
        
        # intelligence_flags에서 특성 정보 추출 (타입 안전성 강화)
        if 'intelligence_flags' in review:
            flags = review['intelligence_flags']
            metadata.update({
                "has_overtime_issues": bool(flags.get('overtime', False)),
                "has_toxic_culture": bool(flags.get('toxic_culture', False)),
                "has_salary_issues": bool(flags.get('salary_issues', False)),
                "has_work_pressure": bool(flags.get('work_pressure', False)),
                "has_negative_worklife": bool(flags.get('has_negative_worklife', False)),
                "has_positive_growth": bool(flags.get('has_positive_growth', False))
            })
        
        # 감정 분석 결과 (타입 안전성 강화)
        if 'processed_data' in review and 'sentiment_analysis' in review['processed_data']:
            sentiment = review['processed_data']['sentiment_analysis']
            metadata.update({
                "sentiment_score": float(sentiment.get('score', 0.0)),
                "sentiment_overall": str(sentiment.get('overall', 'neutral'))
            })
        
        return metadata
    
    def _create_keyword_document(self, keywords: List[Dict], base_metadata: Dict[str, Any]) -> Optional[Document]:
        """
        추출된 키워드들로 별도 문서 생성
        
        Args:
            keywords: 추출된 키워드 리스트
            base_metadata: 기본 메타데이터
            
        Returns:
            키워드 Document 객체 또는 None
        """
        if not keywords:
            return None
        
        # 카테고리별로 키워드 그룹화
        keyword_groups = {}
        for kw in keywords:
            category = kw.get('category', 'general')
            if category not in keyword_groups:
                keyword_groups[category] = []
            keyword_groups[category].append({
                'keyword': kw.get('keyword', ''),
                'context': kw.get('context', ''),
                'frequency': kw.get('frequency', 1)
            })
        
        # 키워드 문서 내용 구성
        content_parts = []
        for category, kw_list in keyword_groups.items():
            content_parts.append(f"{category}: {', '.join([kw['keyword'] for kw in kw_list])}")
        
        if not content_parts:
            return None
        
        content = "주요 키워드: " + " | ".join(content_parts)
        
        # 컬렉션 대상 결정 (키워드 카테고리 기반)
        collection_target = "company_general"
        if "salary" in keyword_groups or "benefits" in keyword_groups:
            collection_target = "salary_discussions"
        elif "culture" in keyword_groups or "management" in keyword_groups:
            collection_target = "culture_reviews"
        elif "growth" in keyword_groups:
            collection_target = "career_advice"
        
        return Document(
            page_content=content,
            metadata={
                **base_metadata,
                "content_type": "keywords",
                "keyword_categories": ", ".join(keyword_groups.keys()),  # 리스트를 문자열로 변환
                "collection_target": collection_target
            }
        )
    
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
        
        # 메타데이터 기반 판단
        if metadata.get("has_salary_mention") or metadata.get("has_benefits_mention"):
            return "salary_discussions"
        elif metadata.get("has_growth_mention"):
            return "career_advice"
        elif metadata.get("has_culture_mention") or metadata.get("has_management_mention"):
            return "culture_reviews"
        elif metadata.get("content_type") == "interview":
            return "interview_reviews"
        else:
            return "company_general"


class ReviewDataLoader:
    """
    리뷰 데이터 로더 및 벡터 데이터베이스 저장 관리자
    """
    
    def __init__(self, data_dir: str = "tools/data/reviews"):
        """
        리뷰 데이터 로더 초기화
        
        Args:
            data_dir: 리뷰 데이터 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.processor = ReviewDataProcessor()
        self.vector_store = VectorStore()
        
        # 처리 통계
        self.stats = {
            "files_processed": 0,
            "documents_created": 0,
            "companies_processed": set(),
            "processing_time": 0.0
        }
    
    async def load_all_reviews(self, company_filter: Optional[List[str]] = None) -> bool:
        """
        모든 리뷰 JSON 파일을 로드하고 벡터 데이터베이스에 저장
        
        Args:
            company_filter: 처리할 회사 리스트 (None이면 모든 회사)
            
        Returns:
            성공 여부
        """
        start_time = datetime.now()
        
        try:
            # JSON 파일들 찾기
            json_files = list(self.data_dir.glob("*_rag_optimized.json"))
            
            if not json_files:
                logger.error(f"JSON 파일을 찾을 수 없습니다: {self.data_dir}")
                return False
            
            logger.info(f"{len(json_files)}개 JSON 파일 발견")
            
            # 회사 필터 적용
            if company_filter:
                filtered_files = []
                for file_path in json_files:
                    company_name = file_path.stem.replace('_rag_optimized', '')
                    if company_name in company_filter:
                        filtered_files.append(file_path)
                json_files = filtered_files
                logger.info(f"필터 적용 후: {len(json_files)}개 파일")
            
            # 각 파일 처리 (전체 진행상황 표시)
            print(f"총 {len(json_files)}개 회사 리뷰 처리 시작...")
            
            for i, file_path in enumerate(tqdm(json_files, desc="회사별 리뷰 처리", unit="회사")):
                company_name = file_path.stem.replace('_rag_optimized', '')
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
            company_name = file_path.stem.replace('_rag_optimized', '')
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
        특정 회사의 리뷰 데이터만 로드
        
        Args:
            company_name: 회사명
            
        Returns:
            성공 여부
        """
        return await self.load_all_reviews(company_filter=[company_name])
    
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