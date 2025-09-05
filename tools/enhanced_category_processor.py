# enhanced_category_processor.py
"""
개선된 카테고리 프로세서 v3.2 - 배치 처리 최적화
- 크롤링과 AI 분류 분리
- 대용량 배치 처리 지원
- OpenAI API 호출 효율성 극대화
"""

import re
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import os
import kss
from tqdm import tqdm

# 기존 모듈 import
from keyword_dictionary import korean_keywords

# 새로운 텍스트 처리 모듈 import
try:
    from text_processor import EnhancedTextProcessor
    TEXT_PROCESSOR_AVAILABLE = True
except ImportError:
    TEXT_PROCESSOR_AVAILABLE = False

# Settings import (환경변수 설정 사용)
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from blindinsight.models.base import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False

# 로깅 설정 간소화
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

@dataclass
class CategoryChunk:
    """카테고리 청크 데이터 구조"""
    id: str
    category: str
    company: str
    content: str
    rating: float
    metadata: Dict[str, Any]
    created_at: str

class CategorySpecificProcessor:
    """카테고리 프로세서 - 배치 처리 최적화"""
    
    def __init__(self, openai_api_key: str = None, enable_spell_check: bool = True):
        self.keyword_dict = korean_keywords
        
        # 청킹 설정 (환경변수 사용)
        if SETTINGS_AVAILABLE:
            self.chunk_size = settings.category_chunk_size
            self.max_chunk_size = settings.category_max_chunk_size
            self.min_sentence_length = settings.category_min_sentence_length
        else:
            # 폴백 기본값
            self.chunk_size = int(os.getenv("CATEGORY_CHUNK_SIZE", "1"))
            self.max_chunk_size = int(os.getenv("CATEGORY_MAX_CHUNK_SIZE", "2"))
            self.min_sentence_length = int(os.getenv("CATEGORY_MIN_SENTENCE_LENGTH", "5"))
        
        # AI 분류 설정
        self.use_ai_classification = bool(openai_api_key) and TEXT_PROCESSOR_AVAILABLE
        
        # 텍스트 처리기 초기화
        if self.use_ai_classification:
            try:
                self.text_processor = EnhancedTextProcessor(
                    openai_api_key=openai_api_key,
                    enable_spell_check=enable_spell_check
                )
            except Exception as e:
                print(f"⚠️ AI 분류기 초기화 실패, 키워드 분류로 전환: {e}")
                self.use_ai_classification = False
                self.text_processor = None
        else:
            self.text_processor = None
        
        # 카테고리 한글명 매핑
        self.category_names_kr = {
            "career_growth": "커리어 향상",
            "salary_benefits": "급여 및 복지",
            "work_life_balance": "업무와 삶의 균형", 
            "company_culture": "사내 문화",
            "management": "경영진"
        }
        
        self.all_categories = list(self.category_names_kr.keys())
        
        # 간소화된 통계
        self.stats = {
            "total_reviews": 0,
            "total_chunks": 0,
            "batch_classifications": 0,
            "keyword_classifications": 0
        }
    
    def create_chunks_without_classification(self, review_data: Dict) -> Dict[str, List[Dict]]:
        """분류 없이 청크만 생성 (배치 처리용)"""
        
        # 기본 정보 추출
        company = review_data["회사"]
        review_id = review_data.get("id", f"review_{company}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        review_number = review_id.split('_')[-1]
        
        # 직원 정보 파싱
        employee_info = self._parse_employee_info(review_data.get("직원유형_원본", ""))
        
        # 평점 정보
        ratings = {
            "career_growth": float(review_data.get("커리어향상", 0)),
            "salary_benefits": float(review_data.get("급여복지", 0)),
            "work_life_balance": float(review_data.get("워라밸", 0)),
            "company_culture": float(review_data.get("사내문화", 0)),
            "management": float(review_data.get("경영진", 0))
        }
        
        # 원본 텍스트 추출
        title_text = review_data.get("제목", "")
        pros_text = review_data.get("장점", "")
        cons_text = review_data.get("단점", "")
        
        # 결과 저장 딕셔너리
        all_chunk_info = {
            "title": [],
            "pros": [],
            "cons": []
        }
        
        # 1. 제목 청크 정보 생성
        if title_text.strip():
            title_sentences = self._split_text_with_kss(title_text)
            if title_sentences:
                title_chunk_info = self._create_chunk_info_without_classification(
                    title_sentences, "title", company, review_number, employee_info, ratings, None
                )
                if title_chunk_info:
                    all_chunk_info["title"].append(title_chunk_info)
        
        # 2. 장점/단점 청크 정보 생성
        pros_chunk_infos = self._create_chunks_info_for_content(
            pros_text, "pros", company, review_number, employee_info, ratings, is_positive=True
        )
        
        cons_chunk_infos = self._create_chunks_info_for_content(
            cons_text, "cons", company, review_number, employee_info, ratings, is_positive=False
        )
        
        all_chunk_info["pros"].extend(pros_chunk_infos)
        all_chunk_info["cons"].extend(cons_chunk_infos)
        
        # 통계 업데이트
        self.stats["total_reviews"] += 1
        total_chunks = sum(len(chunks) for chunks in all_chunk_info.values())
        self.stats["total_chunks"] += total_chunks
        
        return all_chunk_info
    
    def _create_chunks_info_for_content(self, text: str, chunk_type: str, company: str,
                                       review_number: str, employee_info: Dict,
                                       ratings: Dict[str, float], is_positive: bool) -> List[Dict]:
        """텍스트에서 청크 정보들 생성 (분류 없이)"""
        
        if not text or not text.strip():
            return []
        
        # 1단계: 문장 분할 및 청크 그룹핑
        sentences = self._split_text_with_kss(text)
        if not sentences:
            return []
        
        sentence_groups = self._group_sentences_into_chunks(sentences)
        if not sentence_groups:
            return []
        
        # 2단계: 청크 정보 생성
        chunk_infos = []
        
        for i, group in enumerate(sentence_groups):
            content = self._join_sentences_naturally(group)
            if len(content.strip()) >= 20:  # 최소 길이 체크
                chunk_info = self._create_chunk_info_without_classification(
                    group, chunk_type, company, review_number, employee_info, ratings, is_positive, i
                )
                if chunk_info:
                    chunk_infos.append(chunk_info)
        
        return chunk_infos
    
    def _create_chunk_info_without_classification(self, sentences: List[str], chunk_type: str,
                                                company: str, review_number: str, employee_info: Dict,
                                                ratings: Dict[str, float], is_positive: Optional[bool], 
                                                chunk_idx: int = 0) -> Dict:
        """분류 없는 청크 정보 생성"""
        
        content = self._join_sentences_naturally(sentences)
        
        # 기본 청크 정보
        chunk_info = {
            "content": content,
            "chunk_type": chunk_type,
            "company": company,
            "review_number": review_number,
            "employee_info": employee_info,
            "ratings": ratings,
            "is_positive": is_positive,
            "sentences": sentences,
            "chunk_idx": chunk_idx,
            "content_length": len(content),
            "sentence_count": len(sentences)
        }
        
        return chunk_info
    
    def create_final_chunk(self, chunk_info: Dict, classification_result: Dict, 
                          priority: str = "primary") -> Dict:
        """분류 결과를 바탕으로 최종 청크 생성"""
        
        # 분류 결과에서 카테고리 추출
        if priority == "primary":
            category = classification_result.get("primary_category", "career_growth")
            confidence = classification_result.get("primary_confidence", 0.5)
        else:
            category = classification_result.get("secondary_category", "career_growth")
            confidence = classification_result.get("secondary_confidence", 0.3)
        
        # 청크 ID 생성
        priority_suffix = "_2nd" if priority == "secondary" else ""
        chunk_id = (f"{category}_{chunk_info['chunk_type']}_{chunk_info['company']}_"
                   f"{chunk_info['review_number']}_{chunk_info['chunk_idx']:02d}{priority_suffix}")
        
        # 해당 카테고리의 평점
        category_rating = chunk_info['ratings'].get(category, max(chunk_info['ratings'].values()))
        
        # 메타데이터 생성
        metadata = {
            "company": chunk_info['company'],
            "category": category,
            "category_kr": self.category_names_kr.get(category, category),
            "content_type": chunk_info['chunk_type'],
            "is_positive": chunk_info['is_positive'],
            "source_section": self._get_source_section_name(chunk_info['chunk_type'], chunk_info['is_positive']),
            "priority": priority,
            "rating": category_rating,
            "confidence_score": confidence,
            "classification_method": classification_result.get("method", "unknown"),
            "employee_status": chunk_info['employee_info'].get("employment_status", "정보 없음"),
            "sentence_count": chunk_info['sentence_count'],
            "chunk_index": chunk_info['chunk_idx'],
            "content_length": chunk_info['content_length']
        }
        
        return {
            "id": chunk_id,
            "category": category,
            "company": chunk_info['company'],
            "content": chunk_info['content'],
            "rating": category_rating,
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        }
    
    def _get_source_section_name(self, chunk_type: str, is_positive: Optional[bool]) -> str:
        """소스 섹션 이름 반환"""
        if chunk_type == "title":
            return "제목"
        elif is_positive is True:
            return "장점"
        elif is_positive is False:
            return "단점"
        else:
            return "기타"
    
    def _classify_with_keywords_fallback(self, content: str) -> Dict[str, Any]:
        """키워드 기반 폴백 분류 (기존과 동일)"""
        category_scores = {}
        
        for category in self.all_categories:
            keywords = self.keyword_dict.get_category_keywords(category)
            score = self._calculate_keyword_score(content, keywords)
            category_scores[category] = score
        
        # 정렬하여 1순위, 2순위 선택
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_category = sorted_categories[0][0] if sorted_categories[0][1] > 0 else "career_growth"
        primary_confidence = min(0.6, max(0.2, sorted_categories[0][1]))
        
        secondary_category = ""
        secondary_confidence = 0.0
        
        if len(sorted_categories) > 1 and sorted_categories[1][1] > 0.1:
            secondary_category = sorted_categories[1][0]
            secondary_confidence = min(0.4, max(0.1, sorted_categories[1][1]))
        
        self.stats["keyword_classifications"] += 1
        
        return {
            "primary_category": primary_category,
            "primary_confidence": primary_confidence,
            "secondary_category": secondary_category,
            "secondary_confidence": secondary_confidence,
            "normalized_content": content,
            "method": "keyword_fallback"
        }
    
    def _calculate_keyword_score(self, content: str, keywords: Dict) -> float:
        """키워드 점수 계산 (기존과 동일)"""
        if not content or not keywords:
            return 0.0
        
        total_score = 0.0
        content_lower = content.lower()
        
        for keyword_type, keyword_list in keywords.items():
            if keyword_type not in ["primary", "secondary", "context", "negative"]:
                continue
                
            for keyword in keyword_list:
                if keyword in content_lower:
                    if keyword_type == "primary":
                        total_score += 3.0
                    elif keyword_type == "secondary":
                        total_score += 1.5
                    elif keyword_type == "context":
                        total_score += 0.8
                    elif keyword_type == "negative":
                        total_score += 2.0
        
        # 길이로 정규화
        words = content.split()
        if len(words) > 0:
            total_score = total_score / len(words)
        
        return min(total_score, 5.0)
    
    # 기존 헬퍼 메서드들 유지
    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[List[str]]:
        """문장들을 청크로 그룹핑"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        # 환경변수에서 최대 청크 길이 설정
        if SETTINGS_AVAILABLE:
            max_chunk_length = settings.max_chunk_length
        else:
            max_chunk_length = int(os.getenv("MAX_CHUNK_LENGTH", "300"))
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            should_add = (
                len(current_chunk) == 0 or
                (len(current_chunk) < self.max_chunk_size and
                 current_length + sentence_length < max_chunk_length)
            )
            
            if should_add:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [sentence]
                current_length = sentence_length
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _join_sentences_naturally(self, sentences: List[str]) -> str:
        """문장 자연스럽게 연결"""
        if not sentences:
            return ""
        
        content = ""
        for i, sentence in enumerate(sentences):
            content += sentence.strip()
            if i < len(sentences) - 1:
                if sentence.strip().endswith(('.', '!', '?', '다', '요', '음', '함', '됨')):
                    content += "\n"
                else:
                    content += " "
        
        return content
    
    def _split_text_with_kss(self, text: str) -> List[str]:
        """kss를 사용한 문장 분할"""
        if not text or text.strip() in ['정보 없음', '추출 실패', '오류', '']:
            return []
        
        try:
            sentences = kss.split_sentences(text)
            
            valid_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) >= self.min_sentence_length:
                    valid_sentences.append(sentence)
            
            return valid_sentences
            
        except Exception:
            sentences = re.split(r'[.!?]+\s*', text)
            return [s.strip() for s in sentences if s.strip() and len(s.strip()) >= self.min_sentence_length]
    
    def _parse_employee_info(self, employee_raw_text: str) -> Dict[str, str]:
        """직원 정보 파싱"""
        parsed_info = {
            "employment_status": "정보 없음",
            "position": "정보 없음",
            "review_date": "정보 없음"
        }
        
        if not employee_raw_text:
            return parsed_info
        
        try:
            # 날짜 패턴 추출
            date_pattern = r'(\d{4})\.(\d{2})\.(\d{2})'
            date_match = re.search(date_pattern, employee_raw_text)
            if date_match:
                year, month, day = date_match.groups()
                parsed_info["review_date"] = f"{year}.{month}.{day}"
            
            # 직원 상태 추출
            parts = employee_raw_text.split('·')
            if len(parts) >= 1:
                parsed_info["employment_status"] = parts[0].strip()
            if len(parts) >= 3:
                raw_position = parts[2].strip()
                position_clean = re.sub(r'\s*-\s*\d{4}\.\d{2}\.\d{2}', '', raw_position)
                parsed_info["position"] = position_clean.strip()
                
        except Exception:
            pass
        
        return parsed_info
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        stats = self.stats.copy()
        
        total_classifications = stats["batch_classifications"] + stats["keyword_classifications"]
        if total_classifications > 0:
            stats["batch_usage_rate"] = stats["batch_classifications"] / total_classifications
        
        if stats["total_reviews"] > 0:
            stats["avg_chunks_per_review"] = stats["total_chunks"] / stats["total_reviews"]
        
        return stats


class VectorDBOptimizer:
    """벡터 DB 최적화 클래스"""
    
    def __init__(self):
        self.max_chunk_size = 3000
        self.min_chunk_size = 50
    
    def optimize_chunks_for_vectordb(self, final_chunks: List[Dict]) -> List[Dict]:
        """청크를 벡터 DB 저장용으로 최적화"""
        
        optimized_chunks = []
        
        for chunk in final_chunks:
            # 딕셔너리 형태의 청크를 처리
            if isinstance(chunk, dict):
                optimized_chunk = {
                    "id": chunk.get("id", "unknown"),
                    "content": chunk.get("content", ""),
                    "metadata": chunk.get("metadata", {})
                }
            else:
                # CategoryChunk 객체인 경우
                optimized_chunk = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "metadata": chunk.metadata
                }
            
            optimized_chunks.append(optimized_chunk)
        
        return optimized_chunks
    
    def save_vectordb_file(self, company: str, optimized_chunks: List[Dict], 
                          output_dir: str = "./data/vectordb",
                          category_processor: CategorySpecificProcessor = None) -> str:
        """벡터 DB 파일 저장"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일명 생성
        date_str = datetime.now().strftime('%Y%m%d')
        classification_method = "ai_batch" if category_processor and category_processor.use_ai_classification else "keyword"
        filename = f"{date_str}_{company}_{classification_method}_vectordb.json"
        filepath = os.path.join(output_dir, filename)
        
        # 통계 계산
        total_chunks = len(optimized_chunks)
        
        # 분류 방법별 청크 수 계산
        ai_chunks = 0
        keyword_chunks = 0
        
        for chunk in optimized_chunks:
            method = chunk.get("metadata", {}).get("classification_method", "keyword")
            if method == "ai_batch":
                ai_chunks += 1
            else:
                keyword_chunks += 1
        
        vectordb_data = {
            "metadata": {
                "company": company,
                "total_chunks": total_chunks,
                "ai_classified_chunks": ai_chunks,
                "keyword_classified_chunks": keyword_chunks,
                "created_at": datetime.now().isoformat(),
                "classification_method": classification_method,
                "version": "v3.2_batch_optimized"
            },
            "chunks": optimized_chunks
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vectordb_data, f, ensure_ascii=False, indent=2)
        
        return filepath