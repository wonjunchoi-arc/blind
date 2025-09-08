"""
JSON 청크 데이터 전용 프로세서 (성능 최적화 버전)

tools/data의 chunk_first_vectordb JSON 파일들을 처리하여
ChromaDB 벡터 데이터베이스에 최적화된 형태로 변환합니다.
"""

#json_processor.py

import json
import asyncio
import sqlite3
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from pathlib import Path
import logging
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from langchain.schema import Document

from .embeddings import EmbeddingManager, VectorStore, DocumentEmbedding
from ..models.base import settings

# 로깅 설정
logger = logging.getLogger(__name__)


class CompanyMetadataManager:
    """
    회사 메타데이터 (직무, 연도) 관리자
    
    마이그레이션 과정에서 회사별 직무와 연도 정보를 추출하고
    company_metadata.db SQLite DB에 저장합니다.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        CompanyMetadataManager 초기화
        
        Args:
            db_path: SQLite DB 파일 경로 (기본값: C:/blind/data/embeddings/company_metadata.db)
        """
        if db_path is None:
            db_path = "C:/blind/data/embeddings/company_metadata.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        # 회사별 데이터를 저장하는 딕셔너리
        # {company_name: {'positions': set(), 'years': set()}}
        self.company_data = {}
        
    def _init_database(self):
        """SQLite 데이터베이스 및 테이블 초기화"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # companies 테이블
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS companies (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE
                    )
                """)
                
                # positions 테이블 (회사와 관계 유지)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        company_id INTEGER NOT NULL,
                        FOREIGN KEY (company_id) REFERENCES companies (id),
                        UNIQUE(name, company_id)
                    )
                """)
                
                # years 테이블 (회사와 관계 유지)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS years (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        year INTEGER NOT NULL,
                        company_id INTEGER NOT NULL,
                        FOREIGN KEY (company_id) REFERENCES companies (id),
                        UNIQUE(year, company_id)
                    )
                """)
                
                logger.info(f"Company metadata DB 초기화 완료: {self.db_path}")
        except Exception as e:
            logger.error(f"DB 초기화 실패: {str(e)}")
            raise
    
    def extract_position_from_content(self, content: str) -> Set[str]:
        """
        컨텐츠에서 직무/포지션 추출
        
        Args:
            content: 분석할 텍스트 컨텐츠
            
        Returns:
            추출된 직무명 집합
        """
        positions = set()
        
        # 일반적인 직무 키워드들
        position_patterns = [
            r'(개발자|developer|엔지니어|engineer)',
            r'(프론트엔드|frontend|백엔드|backend|풀스택|fullstack)',
            r'(시니어|senior|주니어|junior|신입)',
            r'(팀장|manager|매니저|리더|leader)',
            r'(기획자|planner|PM|프로젝트\s*매니저)',
            r'(디자이너|designer|UX|UI)',
            r'(마케팅|marketing|세일즈|sales)',
            r'(인사|HR|human\s*resource)',
            r'(재무|finance|회계|accounting)',
            r'(운영|operation|ops)',
            r'(QA|품질|quality|테스터|tester)',
            r'(데이터\s*분석가|data\s*analyst|데이터\s*사이언티스트|data\s*scientist)',
            r'(보안|security|시스템\s*관리자|system\s*admin)',
            r'(컨설턴트|consultant|영업|business)',
            r'(연구원|researcher|R&D)',
            r'(인턴|intern|계약직|contractor|정규직|permanent)'
        ]
        
        for pattern in position_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.UNICODE)
            for match in matches:
                position = match.group(0).strip()
                if position and len(position) > 1:
                    positions.add(position)
        
        return positions
    
    def extract_year_from_content(self, content: str) -> Set[int]:
        """
        컨텐츠에서 연도 정보 추출
        
        Args:
            content: 분석할 텍스트 컨텐츠
            
        Returns:
            추출된 연도 집합
        """
        years = set()
        
        # 년도 패턴들 (2020~2024)
        year_patterns = [
            r'\b(202[0-4])\b',  # 2020-2024
            r'\b(201[5-9])\b',  # 2015-2019
            r'(202[0-4])년',    # 2020년-2024년 한국어
            r'(201[5-9])년',    # 2015년-2019년 한국어
        ]
        
        for pattern in year_patterns:
            matches = re.finditer(pattern, content, re.UNICODE)
            for match in matches:
                try:
                    year = int(match.group(1))
                    if 2010 <= year <= 2024:  # 유효한 년도 범위
                        years.add(year)
                except (ValueError, IndexError):
                    continue
        
        return years
    
    def add_company_data(self, company_name: str, content: str, metadata: Dict[str, Any] = None):
        """
        회사별 데이터에서 메타데이터 추출 및 추가
        
        Args:
            company_name: 회사명
            content: 분석할 컨텐츠
            metadata: 추가 메타데이터
        """
        if not company_name or company_name == 'Unknown':
            return
        
        # 회사별 데이터 구조 초기화
        if company_name not in self.company_data:
            self.company_data[company_name] = {'positions': set(), 'years': set()}
        
        # 메타데이터에서 직접 정보 추출 (JSON에서 제공하는 값 우선 사용)
        positions = set()
        years = set()
        
        if metadata:
            # 직무 정보 - JSON 메타데이터에서 직접 가져오기
            for key in ['position', 'job_title', 'role']:
                if key in metadata and metadata[key]:
                    positions.add(str(metadata[key]))
            
            # 연도 정보 - JSON 메타데이터에서 직접 가져오기
            if 'year' in metadata and metadata['year']:
                try:
                    year = int(metadata['year'])
                    if 2010 <= year <= 2025:  # 범위 확장 (2025 포함)
                        years.add(year)
                except (ValueError, TypeError):
                    pass
            
            # 추가 연도 정보 (created_at, review_date 등)
            for key in ['created_at', 'review_date']:
                if key in metadata and metadata[key]:
                    try:
                        year_str = str(metadata[key])
                        year_match = re.search(r'(202[0-5]|201[5-9])', year_str)  # 2025까지 포함
                        if year_match:
                            year = int(year_match.group(1))
                            if 2010 <= year <= 2025:
                                years.add(year)
                    except (ValueError, TypeError):
                        continue
        
        # JSON에 정보가 없을 때만 content에서 추출 (fallback)
        if not positions:
            positions = self.extract_position_from_content(content)
        if not years:
            years = self.extract_year_from_content(content)
        
        # 해당 회사의 직무와 연도 추가
        self.company_data[company_name]['positions'].update(positions)
        self.company_data[company_name]['years'].update(years)
    
    def save_to_database(self):
        """
        회사별 메타데이터를 SQLite DB에 저장 (관계형 구조)
        
        Returns:
            저장된 레코드 수
        """
        saved_count = 0
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                for company_name, data in self.company_data.items():
                    # 1. 회사 저장
                    try:
                        cursor.execute("INSERT OR IGNORE INTO companies (name) VALUES (?)", (company_name,))
                        saved_count += 1
                        
                        # 회사 ID 가져오기
                        cursor.execute("SELECT id FROM companies WHERE name = ?", (company_name,))
                        company_id = cursor.fetchone()[0]
                        
                        # 2. 해당 회사의 직무들 저장
                        for position in data['positions']:
                            if position and position.strip():
                                try:
                                    cursor.execute(
                                        "INSERT OR IGNORE INTO positions (name, company_id) VALUES (?, ?)",
                                        (position, company_id)
                                    )
                                    saved_count += 1
                                except sqlite3.Error as e:
                                    logger.warning(f"직무 저장 실패 ({company_name} - {position}): {e}")
                        
                        # 3. 해당 회사의 연도들 저장
                        for year in data['years']:
                            try:
                                cursor.execute(
                                    "INSERT OR IGNORE INTO years (year, company_id) VALUES (?, ?)",
                                    (year, company_id)
                                )
                                saved_count += 1
                            except sqlite3.Error as e:
                                logger.warning(f"연도 저장 실패 ({company_name} - {year}): {e}")
                                
                    except sqlite3.Error as e:
                        logger.warning(f"회사 저장 실패 ({company_name}): {e}")
                
                conn.commit()
                logger.info(f"Company metadata DB에 {saved_count}개 레코드 저장 완료")
                
        except Exception as e:
            logger.error(f"DB 저장 중 오류: {str(e)}")
            raise
        
        return saved_count
    
    def get_all_companies(self) -> List[str]:
        """
        모든 회사 목록 조회
        
        Returns:
            회사명 리스트
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT name FROM companies ORDER BY name")
                results = cursor.fetchall()
                return [row[0] for row in results]
                
        except Exception as e:
            logger.error(f"회사 목록 조회 중 오류: {str(e)}")
            return []
    
    def get_positions_for_company(self, company_name: str = None) -> List[str]:
        """
        특정 회사의 직무 목록 조회
        
        Args:
            company_name: 회사명 (None이면 모든 회사의 직무)
            
        Returns:
            직무 리스트
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                if company_name:
                    # 특정 회사의 직무만 조회
                    cursor.execute("""
                        SELECT DISTINCT p.name
                        FROM positions p
                        JOIN companies c ON p.company_id = c.id
                        WHERE c.name = ?
                        ORDER BY p.name
                    """, (company_name,))
                else:
                    # 모든 회사의 직무 조회
                    cursor.execute("SELECT DISTINCT name FROM positions ORDER BY name")
                
                results = cursor.fetchall()
                return [row[0] for row in results]
                
        except Exception as e:
            logger.error(f"직무 조회 중 오류: {str(e)}")
            return []
    
    def get_years_for_company(self, company_name: str = None) -> List[int]:
        """
        특정 회사의 연도 목록 조회
        
        Args:
            company_name: 회사명 (None이면 모든 회사의 연도)
            
        Returns:
            연도 리스트 (내림차순)
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.cursor()
                
                if company_name:
                    # 특정 회사의 연도만 조회
                    cursor.execute("""
                        SELECT DISTINCT y.year
                        FROM years y
                        JOIN companies c ON y.company_id = c.id
                        WHERE c.name = ?
                        ORDER BY y.year DESC
                    """, (company_name,))
                else:
                    # 모든 회사의 연도 조회
                    cursor.execute("SELECT DISTINCT year FROM years ORDER BY year DESC")
                
                results = cursor.fetchall()
                return [row[0] for row in results]
                
        except Exception as e:
            logger.error(f"연도 조회 중 오류: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """수집된 메타데이터 통계 반환"""
        all_positions = set()
        all_years = set()
        
        for company_data in self.company_data.values():
            all_positions.update(company_data['positions'])
            all_years.update(company_data['years'])
        
        return {
            'total_companies': len(self.company_data),
            'total_positions': len(all_positions),
            'total_years': len(all_years),
            'companies': sorted(list(self.company_data.keys())),
            'positions': sorted(list(all_positions)),
            'years': sorted(list(all_years), reverse=True),
            'company_details': {
                company: {
                    'positions': sorted(list(data['positions'])),
                    'years': sorted(list(data['years']), reverse=True)
                }
                for company, data in self.company_data.items()
            }
        }


class ChunkDataProcessor:
    """
    JSON 청크 데이터 전용 프로세서 (성능 최적화 버전)
    
    tools/data의 chunk_first_vectordb JSON 파일들을 처리하여
    RAG 시스템에서 활용할 수 있도록 구조화된 Document 객체로 변환합니다.
    """
    
    def __init__(self, metadata_manager: Optional[CompanyMetadataManager] = None):
        """청크 데이터 프로세서 초기화"""
        self.embedding_manager = EmbeddingManager(model_name="text-embedding-3-small", batch_size=100)
        self.processed_count = 0
        
        # 메타데이터 매니저 (선택사항)
        self.metadata_manager = metadata_manager
        
        # 새로운 6개 카테고리별 컬렉션 매핑 (1:1 매핑)
        self.collection_mapping = {
            "career_growth": "career_growth",
            "salary_benefits": "salary_benefits", 
            "work_life_balance": "work_life_balance",
            "company_culture": "company_culture",
            "management": "management",
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
                
                # 메타데이터 매니저가 있으면 메타데이터 수집
                if self.metadata_manager and chunk_docs:
                    chunk_metadata = chunk.get('metadata', {})
                    # JSON에서 직접 position과 year 가져오기
                    position = chunk_metadata.get("position")
                    year = chunk_metadata.get("year")

                    # 메타데이터 정보를 이용해 company data 추가
                    if position or year:
                        self.metadata_manager.add_company_data(company_name, chunk.get('content', ''), chunk_metadata)
            
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
        collection_target = self.collection_mapping.get(category, 'general')
        
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
        청크에서 메타데이터 추출 (None 값 안전 처리)
        
        Args:
            chunk: 청크 데이터
            company_name: 회사명
            
        Returns:
            메타데이터 딕셔너리
        """
        metadata = {
            "company": company_name,
            "source_type": "company_chunk",
            "chunk_id": str(chunk.get('id', 'unknown'))
        }
        
        # 청크의 메타데이터에서 정보 추출
        chunk_metadata = chunk.get('metadata', {})
        if chunk_metadata:
            # None 값을 안전하게 처리하는 헬퍼 함수
            def safe_str(value, default='unknown'):
                return str(value) if value is not None else default
            
            def safe_float(value, default=0.0):
                try:
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    return default
            
            def safe_int(value, default=0):
                try:
                    return int(value) if value is not None else default
                except (ValueError, TypeError):
                    return default
            
            def safe_bool(value, default='unknown'):
                if value is None:
                    return default
                elif isinstance(value, bool):
                    return str(value).lower()
                else:
                    return safe_str(value, default)
            
            metadata.update({
                # 실제 JSON에 존재하는 필드들 (None 값 안전 처리)
                "category": safe_str(chunk_metadata.get('category'), 'general'),
                "category_kr": safe_str(chunk_metadata.get('category_kr')),
                "content_type": safe_str(chunk_metadata.get('content_type')),
                "is_positive": safe_bool(chunk_metadata.get('is_positive')),
                "source_section": safe_str(chunk_metadata.get('source_section')),
                "priority": safe_str(chunk_metadata.get('priority')),
                "rating": safe_float(chunk_metadata.get('rating')),
                "confidence_score": safe_float(chunk_metadata.get('confidence_score')),
                "classification_method": safe_str(chunk_metadata.get('classification_method')),
                "employee_status": safe_str(chunk_metadata.get('employee_status')),
                "employee_type": safe_str(chunk_metadata.get('employee_type')),
                "position": safe_str(chunk_metadata.get('position')),
                "year": safe_str(chunk_metadata.get('year')),
                "sentence_count": safe_int(chunk_metadata.get('sentence_count')),
                "chunk_index": safe_int(chunk_metadata.get('chunk_index')),
                "content_length": safe_int(chunk_metadata.get('content_length'))
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
        return self.collection_mapping.get(category, "general")


class ChunkDataLoader:
    """
    청크 데이터 로더 및 벡터 데이터베이스 저장 관리자 (성능 최적화 버전)
    """
    
    def __init__(self, data_dir: str = "data", enable_metadata_collection: bool = True):
        """
        청크 데이터 로더 초기화
        
        Args:
            data_dir: 청크 데이터 디렉토리 경로
            enable_metadata_collection: 메타데이터 수집 활성화 여부
        """
        self.data_dir = Path(data_dir)
        
        # 메타데이터 매니저 초기화
        self.metadata_manager = CompanyMetadataManager() if enable_metadata_collection else None
        
        self.processor = ChunkDataProcessor(metadata_manager=self.metadata_manager)
        self.vector_store = VectorStore()
        
        # 처리 통계
        self.stats = {
            "files_processed": 0,
            "documents_created": 0,
            "companies_processed": set(),
            "processing_time": 0.0,
            "metadata_records": 0
        }
    
    async def load_all_chunks_optimized(self, company_filter: Optional[List[str]] = None) -> bool:
        """
        모든 청크 JSON 파일을 최적화된 배치 방식으로 로드 및 저장
        
        Args:
            company_filter: 처리할 회사 리스트 (None이면 모든 회사)
            
        Returns:
            성공 여부
        """
        start_time = datetime.now()
        
        try:
            # JSON 파일들 찾기
            chunk_files = list(self.data_dir.glob("*_chunk_first_vectordb.json"))
            batch_ai_files = list(self.data_dir.glob("*_ai_batch_vectordb.json"))
            json_files = chunk_files + batch_ai_files
            
            if not json_files:
                logger.error(f"JSON 파일을 찾을 수 없습니다: {self.data_dir}")
                return False
            
            # 회사 필터 적용
            if company_filter:
                filtered_files = []
                for file_path in json_files:
                    company_name = self._extract_company_name(file_path)
                    if company_name in company_filter:
                        filtered_files.append(file_path)
                json_files = filtered_files
            
            logger.info(f"배치 최적화 처리: {len(json_files)}개 파일")
            print(f"[배치최적화] {len(json_files)}개 회사 처리 시작...")
            
            # 전체 파일을 여러 배치로 나누어 처리 (메모리 효율성)
            batch_size = 5  # 동시에 처리할 회사 수
            file_batches = [
                json_files[i:i + batch_size] 
                for i in range(0, len(json_files), batch_size)
            ]
            
            total_processed = 0
            
            for batch_idx, file_batch in enumerate(file_batches):
                print(f"\n[배치 {batch_idx + 1}/{len(file_batches)}] {len(file_batch)}개 회사 동시 처리...")
                
                # 배치 내 파일들을 병렬 처리
                batch_tasks = []
                for file_path in file_batch:
                    task = self._process_single_file_optimized(file_path)
                    batch_tasks.append(task)
                
                # 배치 내 파일들 병렬 실행
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # 결과 확인
                batch_success_count = sum(1 for result in batch_results if result is True)
                total_processed += batch_success_count
                
                print(f"    배치 {batch_idx + 1} 완료: {batch_success_count}/{len(file_batch)}개 성공")
                
                # 배치 간 메모리 정리를 위한 대기
                if batch_idx < len(file_batches) - 1:
                    await asyncio.sleep(1.0)
            
            # 통계 업데이트
            end_time = datetime.now()
            self.stats["processing_time"] = (end_time - start_time).total_seconds()
            
            # 메타데이터 매니저가 있으면 DB에 저장
            if self.metadata_manager:
                print(f"\n[메타데이터] 회사별 직무/연도 정보 저장 중...")
                try:
                    metadata_count = self.metadata_manager.save_to_database()
                    self.stats["metadata_records"] = metadata_count
                    
                    # 통계 출력
                    metadata_stats = self.metadata_manager.get_statistics()
                    print(f"[메타데이터] {metadata_count}개 레코드 저장 완료")
                    print(f"[메타데이터] 회사 {metadata_stats['total_companies']}개, 직무 {metadata_stats['total_positions']}개, 연도 {metadata_stats['total_years']}개 수집됨")
                    print(f"[메타데이터] 회사 목록: {', '.join(metadata_stats['companies'][:5])}{'...' if len(metadata_stats['companies']) > 5 else ''}")
                    print(f"[메타데이터] 연도 범위: {min(metadata_stats['years']) if metadata_stats['years'] else 0} ~ {max(metadata_stats['years']) if metadata_stats['years'] else 0}")
                        
                except Exception as e:
                    logger.error(f"메타데이터 저장 실패: {str(e)}")
                    print(f"[경고] 메타데이터 저장 실패: {str(e)}")
            
            print(f"\n[완료] 배치 최적화 처리 완료: {total_processed}/{len(json_files)}개 회사 성공")
            logger.info(f"배치 최적화 처리 완료: {self.stats}")
            return total_processed > 0
            
        except Exception as e:
            logger.error(f"배치 최적화 로딩 중 오류: {str(e)}")
            return False

    
    async def _process_single_file_optimized(self, file_path: Path) -> bool:
        """
        단일 JSON 파일 처리 및 벡터 저장소에 저장 (최적화된 배치 처리)
        
        Args:
            file_path: JSON 파일 경로
            
        Returns:
            성공 여부
        """
        try:
            # 회사명 추출 (두 가지 패턴 모두 지원)
            company_name = file_path.stem
            if company_name.endswith('_chunk_first_vectordb'):
                company_name = company_name.replace('_chunk_first_vectordb', '')
            elif company_name.endswith('_ai_batch_vectordb'):
                company_name = company_name.replace('_ai_batch_vectordb', '')
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
            
            # 각 컬렉션별로 최적화된 배치 처리
            total_saved = 0
            for collection_name, docs in collection_docs.items():
                print(f"  └─ {collection_name} 컬렉션: {len(docs)}개 문서 처리 중...")
                
                # Document를 DocumentEmbedding으로 변환 (배치 임베딩 생성)
                doc_embeddings = await self._create_batch_document_embeddings_optimized(docs, company_name, collection_name)
                
                # 최적화된 배치 저장
                try:
                    success = await self.vector_store.add_documents_batch_optimized(
                        doc_embeddings, 
                        collection_name,
                        batch_size=500  # ChromaDB 배치 크기
                    )
                    
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
            
            # 회사 메타데이터 추출 및 저장 (최적화)
            if total_saved > 0:
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
    
    async def _create_batch_document_embeddings_optimized(
    self, 
    docs: List[Document], 
    company_name: str, 
    collection_name: str
    ) -> List[DocumentEmbedding]:
        """
        Document 리스트를 DocumentEmbedding 리스트로 최적화된 배치 변환
        
        Args:
            docs: Document 객체 리스트
            company_name: 회사명
            collection_name: 컬렉션명
            
        Returns:
            DocumentEmbedding 객체 리스트
        """
        if not docs:
            return []
        
        print(f"    임베딩 생성: {len(docs)}개 문서")
        
        # 1단계: 모든 텍스트 추출
        texts = [doc.page_content for doc in docs]
        
        # 2단계: 최적화된 배치 임베딩 생성
        embeddings = await self.processor.embedding_manager.create_batch_embeddings_optimized(
            texts, 
            max_batch_size=100
        )
        
        # 3단계: DocumentEmbedding 객체 생성
        doc_embeddings = []
        for i, (doc, embedding) in enumerate(zip(docs, embeddings)):
            import hashlib
            
            # 문서 ID 생성
            doc_id = f"{company_name}_{i}_{collection_name}"
            
            # 해시값 생성
            hash_value = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()
            
            doc_embedding = DocumentEmbedding(
                document_id=doc_id,
                content=doc.page_content,
                embedding=embedding,
                metadata=doc.metadata,
                created_at=datetime.now(),
                hash_value=hash_value
            )
            doc_embeddings.append(doc_embedding)
        
        return doc_embeddings
    

    
    async def load_single_company_optimized(self, company_name: str) -> bool:
        """
        특정 회사의 청크 데이터를 최적화된 방식으로 로드
        
        Args:
            company_name: 회사명
            
        Returns:
            성공 여부
        """
        return await self.load_all_chunks_optimized(company_filter=[company_name])
    def _extract_company_name(self, file_path: Path) -> str:
        """파일 경로에서 회사명 추출"""
        company_name = file_path.stem
        if company_name.endswith('_chunk_first_vectordb'):
            return company_name.replace('_chunk_first_vectordb', '')
        elif company_name.endswith('_ai_batch_vectordb'):
            return company_name.replace('_ai_batch_vectordb', '')
        return company_name
    
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
        collections = ["company_culture", "work_life_balance", "management", "salary_benefits", "career_growth", "general"]
        
        for collection_name in collections:
            try:
                stats = self.vector_store.get_collection_stats(collection_name)
                verification_result["collections_status"][collection_name] = stats
                verification_result["total_documents"] += stats.get("document_count", 0)
            except Exception as e:
                verification_result["errors"].append(f"{collection_name}: {str(e)}")
        
        return verification_result


class BatchPerformanceMonitor:
    """
    배치 처리 성능 모니터링 클래스
    """
    
    def __init__(self):
        self.metrics = {
            "total_documents": 0,
            "total_embeddings_created": 0,
            "total_api_calls": 0,
            "embedding_time": 0.0,
            "storage_time": 0.0,
            "start_time": None,
            "end_time": None
        }
    
    def start_monitoring(self):
        """모니터링 시작"""
        self.metrics["start_time"] = datetime.now()
    
    def stop_monitoring(self):
        """모니터링 종료"""
        self.metrics["end_time"] = datetime.now()
    
    def add_embedding_metrics(self, doc_count: int, api_calls: int, time_taken: float):
        """임베딩 메트릭 추가"""
        self.metrics["total_documents"] += doc_count
        self.metrics["total_embeddings_created"] += doc_count
        self.metrics["total_api_calls"] += api_calls
        self.metrics["embedding_time"] += time_taken
    
    def add_storage_metrics(self, doc_count: int, time_taken: float):
        """저장 메트릭 추가"""
        self.metrics["storage_time"] += time_taken
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 보고서 생성"""
        if not self.metrics["start_time"] or not self.metrics["end_time"]:
            return {"error": "모니터링이 완료되지 않음"}
        
        total_time = (self.metrics["end_time"] - self.metrics["start_time"]).total_seconds()
        
        report = {
            "processing_summary": {
                "total_time_seconds": total_time,
                "total_documents": self.metrics["total_documents"],
                "documents_per_second": self.metrics["total_documents"] / total_time if total_time > 0 else 0,
                "total_api_calls": self.metrics["total_api_calls"],
                "api_calls_per_second": self.metrics["total_api_calls"] / total_time if total_time > 0 else 0
            },
            "embedding_performance": {
                "embedding_time_seconds": self.metrics["embedding_time"],
                "embedding_percentage": (self.metrics["embedding_time"] / total_time * 100) if total_time > 0 else 0,
                "embeddings_per_second": self.metrics["total_embeddings_created"] / self.metrics["embedding_time"] if self.metrics["embedding_time"] > 0 else 0
            },
            "storage_performance": {
                "storage_time_seconds": self.metrics["storage_time"],
                "storage_percentage": (self.metrics["storage_time"] / total_time * 100) if total_time > 0 else 0,
                "storage_per_second": self.metrics["total_documents"] / self.metrics["storage_time"] if self.metrics["storage_time"] > 0 else 0
            },
            "efficiency_metrics": {
                "batch_efficiency": "최적화된 배치 처리 사용",
                "memory_efficiency": "메모리 최적화 적용",
                "api_efficiency": f"평균 배치 크기: {self.metrics['total_embeddings_created'] / max(1, self.metrics['total_api_calls']):.1f}"
            }
        }
        
        return report