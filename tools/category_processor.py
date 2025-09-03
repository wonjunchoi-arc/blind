"""
청크 우선 생성 방식 카테고리 프로세서

새로운 접근법:
1. 모든 문장을 2-3개씩 청크로 분할
2. 각 청크에 대해 카테고리 점수 계산 
3. 주요 카테고리와 점수를 메타데이터에 저장
4. 정보 누락 없이 모든 내용 보존

수정 방안:
- 문장단위로 쪼갠 내용을 llm에 배치로 넣어 카테고리 별 분류를 하도록 하는 것
- 기존의 6개 형식이 아닌 정확한 5개의 카테고리로 분류 general은 없이

"""

#category_processor.py

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from keyword_dictionary import korean_keywords

# kss 라이브러리 import 추가
try:
    import kss
except ImportError:
    print("kss 라이브러리가 설치되지 않았습니다. 다음 명령어로 설치해주세요:")
    print("pip install kss")
    raise

logger = logging.getLogger(__name__)

@dataclass
class CategoryChunk:
    """카테고리별 청크 데이터 구조 (기존 구조 유지)"""
    id: str
    category: str
    company: str
    content: str
    rating: float
    metadata: Dict[str, Any]
    created_at: str

class CategorySpecificProcessor:
    """청크 우선 생성 방식 카테고리 프로세서"""
    
    def __init__(self):
        self.keyword_dict = korean_keywords
        
        # 청킹 설정
        self.chunk_size = 2         # 기본 청크 크기 (문장 수)
        self.max_chunk_size = 3     # 최대 청크 크기
        self.min_sentence_length = 5  # 최소 문장 길이
        
        # 카테고리 한글명 매핑
        self.category_names_kr = {
            "career_growth": "커리어 향상",
            "salary_benefits": "급여 및 복지",
            "work_life_balance": "업무와 삶의 균형", 
            "company_culture": "사내 문화",
            "management": "경영진"
        }
        
        # 모든 카테고리 리스트
        self.all_categories = list(self.category_names_kr.keys())
    
    def process_review_by_categories(self, review_data: Dict) -> Dict[str, List[CategoryChunk]]:
        """새로운 방식: 청크 우선 생성 후 카테고리 분류"""
        
        # 기본 정보 추출
        company = review_data["회사"]
        review_id = review_data.get("id", f"review_{company}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        review_number = review_id.split('_')[-1]
        
        # 직원 정보 파싱
        employee_info = self._parse_employee_info(review_data.get("직원유형_원본", ""))
        
"""
청크 우선 생성 방식 카테고리 프로세서 (상위 2개 카테고리)

새로운 접근법:
1. 모든 문장을 2-3개씩 청크로 분할
2. 각 청크에 대해 모든 카테고리 점수 계산 
3. 상위 2개 카테고리만 선택 (임계값 조건 포함)
4. 주요 카테고리와 점수를 메타데이터에 저장
"""

#category_processor.py

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from keyword_dictionary import korean_keywords

# kss 라이브러리 import 추가
try:
    import kss
except ImportError:
    print("kss 라이브러리가 설치되지 않았습니다. 다음 명령어로 설치해주세요:")
    print("pip install kss")
    raise

logger = logging.getLogger(__name__)

@dataclass
class CategoryChunk:
    """카테고리별 청크 데이터 구조 (기존 구조 유지)"""
    id: str
    category: str
    company: str
    content: str
    rating: float
    metadata: Dict[str, Any]
    created_at: str

class CategorySpecificProcessor:
    """청크 우선 생성 방식 카테고리 프로세서 (상위 2개 카테고리)"""
    
    def __init__(self):
        self.keyword_dict = korean_keywords
        
        # 청킹 설정
        self.chunk_size = 3         # 기본 청크 크기 (문장 수)
        self.max_chunk_size = 5     # 최대 청크 크기
        self.min_sentence_length = 5  # 최소 문장 길이
        
        # 카테고리 선택 설정
        self.max_categories_per_chunk = 2      # 청크당 최대 카테고리 수
        self.category_score_threshold = 0.3    # 카테고리 선택 최소 임계값
        self.score_gap_threshold = 0.1         # 점수 차이 임계값
        
        # 카테고리 한글명 매핑
        self.category_names_kr = {
            "career_growth": "커리어 향상",
            "salary_benefits": "급여 및 복지",
            "work_life_balance": "업무와 삶의 균형", 
            "company_culture": "사내 문화",
            "management": "경영진"
        }
        
        # 모든 카테고리 리스트
        self.all_categories = list(self.category_names_kr.keys())
    
    def process_review_by_categories(self, review_data: Dict) -> Dict[str, List[CategoryChunk]]:
        """새로운 방식: 청크 우선 생성 후 상위 2개 카테고리 분류"""
        
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
        
        # 원본 텍스트와 문장 분할
        title_text = review_data.get("제목", "")
        pros_text = review_data.get("장점", "")
        cons_text = review_data.get("단점", "")
        
        title_sentences = self._split_text_with_kss(title_text)
        pros_sentences = self._split_text_with_kss(pros_text)
        cons_sentences = self._split_text_with_kss(cons_text)
        
        # 결과를 저장할 딕셔너리
        all_chunks = {
            "title": [],
            "pros": [],
            "cons": []
        }
        
        # 1. 제목 청크 생성
        if title_sentences:
            title_chunk = self._create_title_chunk(
                title_sentences, company, review_number, employee_info
            )
            all_chunks["title"].append(title_chunk)
        
        # 2. 장점 청크 생성 (청크 우선 방식)
        if pros_sentences:
            pros_chunks = self._create_chunks_with_top_categories(
                pros_sentences, "pros", company, review_number, 
                employee_info, ratings, is_positive=True
            )
            # 카테고리별로 분류하여 저장
            self._distribute_chunks_by_category(pros_chunks, all_chunks)
        
        # 3. 단점 청크 생성 (청크 우선 방식)
        if cons_sentences:
            cons_chunks = self._create_chunks_with_top_categories(
                cons_sentences, "cons", company, review_number,
                employee_info, ratings, is_positive=False
            )
            # 카테고리별로 분류하여 저장
            self._distribute_chunks_by_category(cons_chunks, all_chunks)
        
        return all_chunks
    
    def _create_chunks_with_top_categories(self, sentences: List[str], chunk_type: str, 
                                         company: str, review_number: str, employee_info: Dict,
                                         ratings: Dict[str, float], is_positive: bool) -> List[CategoryChunk]:
        """문장을 청크로 분할하고 각 청크에 상위 2개 카테고리 할당"""
        
        if not sentences:
            return []
        
        # 1단계: 문장들을 청크로 그룹핑
        sentence_groups = self._group_sentences_into_chunks(sentences)
        
        # 2단계: 각 청크에 대해 카테고리 점수 계산 및 상위 2개 선택
        chunks = []
        
        for chunk_idx, sentence_group in enumerate(sentence_groups):
            # 청크 콘텐츠 생성
            content = self._join_sentences_naturally(sentence_group)
            
            # 너무 짧은 청크는 제외
            if len(content.strip()) < 20:
                continue
            
            # 각 카테고리에 대한 점수 계산
            category_scores = self._calculate_chunk_category_scores(content)
            
            # 상위 2개 카테고리 선택
            top_categories = self._select_top_categories(category_scores)
            
            # 선택된 카테고리가 없으면 general로 분류
            if not top_categories:
                top_categories = [("general", 0.1)]
            
            # 각 선택된 카테고리별로 청크 생성
            for category, score in top_categories:
                chunk_id = f"{category}_{chunk_type}_{company}_{review_number}_{chunk_idx:02d}"
                
                # 해당 카테고리의 평점 가져오기
                category_rating = ratings.get(category, max(ratings.values()))
                
                # 메타데이터 생성
                metadata = self._create_chunk_metadata_with_scores(
                    category, chunk_type, company, employee_info, 
                    category_rating, sentence_group, is_positive, 
                    chunk_idx, category_scores, top_categories
                )
                
                # 청크 생성
                chunk = CategoryChunk(
                    id=chunk_id,
                    category=category,
                    company=company,
                    content=content,
                    rating=category_rating,
                    metadata=metadata,
                    created_at=datetime.now().isoformat()
                )
                
                chunks.append(chunk)
        
        return chunks
    
    def _calculate_chunk_category_scores(self, content: str) -> Dict[str, float]:
        """청크에 대한 모든 카테고리 점수 계산"""
        
        category_scores = {}
        
        for category in self.all_categories:
            keywords = self.keyword_dict.get_category_keywords(category)
            score = self._calculate_content_relevance(content, keywords)
            category_scores[category] = score
        
        return category_scores
    
    def _select_top_categories(self, category_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        """상위 2개 카테고리 선택 (스마트 로직)"""
        
        # 점수 순으로 정렬
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected_categories = []
        
        # 1순위는 임계값 이상이면 무조건 선택
        if sorted_categories[0][1] >= self.category_score_threshold:
            selected_categories.append(sorted_categories[0])
            
            # 2순위 선택 조건:
            # 1) 임계값 이상
            # 2) 1순위와의 점수 차이가 너무 크지 않음
            if len(sorted_categories) > 1:
                second_score = sorted_categories[1][1]
                first_score = sorted_categories[0][1]
                
                if (second_score >= self.category_score_threshold and 
                    (first_score - second_score) <= self.score_gap_threshold * 2):  # 관대한 기준
                    selected_categories.append(sorted_categories[1])
        
        return selected_categories
    
    def _distribute_chunks_by_category(self, chunks: List[CategoryChunk], 
                                     all_chunks: Dict[str, List[CategoryChunk]]):
        """생성된 청크들을 카테고리별로 분류하여 저장"""
        
        for chunk in chunks:
            chunk_type = chunk.metadata.get("content_type", "pros")
            
            # pros/cons 청크는 해당 섹션에 저장
            if chunk_type in ["pros", "cons"]:
                all_chunks[chunk_type].append(chunk)
    
    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[List[str]]:
        """문장들을 청크 단위로 그룹핑"""
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_length = 400
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # 현재 청크에 추가할지 판단
            should_add = (
                len(current_chunk) < self.chunk_size or  
                (len(current_chunk) < self.max_chunk_size and 
                 current_length + sentence_length < max_chunk_length)
            )
            
            if should_add:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # 현재 청크 저장하고 새 청크 시작
                if current_chunk:
                    chunks.append(current_chunk)
                
                current_chunk = [sentence]
                current_length = sentence_length
        
        # 마지막 청크 저장
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _calculate_content_relevance(self, content: str, keywords: Dict) -> float:
        """콘텐츠의 카테고리 관련성 점수 계산"""
        if not content or not keywords:
            return 0.0
        
        total_score = 0.0
        content_lower = content.lower()
        
        # 키워드 매칭 점수
        for keyword_type, keyword_list in keywords.items():
            if keyword_type not in ["primary", "secondary", "context", "negative"]:
                continue
                
            for keyword in keyword_list:
                if keyword in content_lower:
                    # 키워드 타입별 가중치
                    if keyword_type == "primary":
                        total_score += 3.0
                    elif keyword_type == "secondary":
                        total_score += 1.5
                    elif keyword_type == "context":
                        total_score += 0.8
                    elif keyword_type == "negative":
                        total_score += 2.0
        
        # 콘텐츠 길이로 정규화
        words = content.split()
        if len(words) > 0:
            total_score = total_score / len(words)
        
        return min(total_score, 10.0)  # 최대 10점으로 제한
    
    def _create_chunk_metadata_with_scores(self, category: str, chunk_type: str, company: str,
                                         employee_info: Dict, rating: float, 
                                         sentences: List[str], is_positive: bool, chunk_idx: int,
                                         category_scores: Dict[str, float], 
                                         top_categories: List[Tuple[str, float]]) -> Dict[str, Any]:
        """점수 정보를 포함한 청크 메타데이터 생성"""
        
        content = ' '.join(sentences)
        
        # 키워드 테마 추출
        keywords = self.keyword_dict.get_category_keywords(category)
        key_themes = self._extract_key_themes(content, keywords)
        
        return {
            "company": company,
            "category": category,
            "category_kr": self.category_names_kr.get(category, category),
            "content_type": chunk_type,
            "is_positive": is_positive,
            "source_section": "장점" if is_positive else "단점",
            "rating": rating,
            "rating_level": self._categorize_rating(rating),
            "sentiment": "positive" if is_positive else "negative",
            
            # 직원 정보
            "employee_status": employee_info.get("employment_status", "정보 없음"),
            "position": employee_info.get("position", "정보 없음"),
            "review_date": employee_info.get("review_date", "정보 없음"),
            "review_year": employee_info.get("review_year", "정보 없음"),
            "review_month": employee_info.get("review_month", "정보 없음"),
            "year_category": self._get_year_category(employee_info.get("review_year")),
            "is_recent": employee_info.get("review_year", "0") >= "2023",
            
            # 카테고리 점수 정보 (새로 추가)
            "all_category_scores": category_scores,
            "selected_categories": dict(top_categories),
            "primary_category": category,
            "primary_category_score": dict(top_categories).get(category, 0.0),
            "is_multi_category": len(top_categories) > 1,
            
            # 콘텐츠 정보
            "sentence_count": len(sentences),
            "chunk_index": chunk_idx,
            "key_themes": key_themes,
            "content_length": len(content),
            "chunking_strategy": "chunk_first_top2_categories",
            
            # 처리 정보
            "processing_timestamp": datetime.now().isoformat()
        }
    
    def _join_sentences_naturally(self, sentences: List[str]) -> str:
        """문장을 자연스럽게 연결"""
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
        """kss를 사용하여 문장 단위로 분할"""
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
            
        except Exception as e:
            logger.warning(f"kss 문장 분할 실패: {e}, 기본 분할 사용")
            sentences = re.split(r'[.!?]+\s*', text)
            return [s.strip() for s in sentences if s.strip() and len(s.strip()) >= self.min_sentence_length]
    
    def _extract_key_themes(self, content: str, keywords: Dict) -> List[str]:
        """키워드 기반 주요 테마 추출"""
        themes = []
        
        if not keywords:
            return themes
        
        for keyword in keywords.get("primary", []):
            if keyword in content:
                themes.append(keyword)
        
        for keyword in keywords.get("secondary", [])[:5]:
            if keyword in content and keyword not in themes:
                themes.append(keyword)
        
        return themes[:8]
    
    def _create_title_chunk(self, title_sentences: List[str], company: str, 
                          review_number: str, employee_info: Dict) -> CategoryChunk:
        """제목 청크 생성"""
        
        chunk_id = f"title_{company}_{review_number}"
        content = "\n".join(title_sentences)
        
        metadata = {
            "company": company,
            "category": "general",
            "category_kr": "전체",
            "content_type": "title",
            "is_positive": None,
            "source_section": "제목",
            "employee_status": employee_info.get("employment_status", "정보 없음"),
            "position": employee_info.get("position", "정보 없음"),
            "review_date": employee_info.get("review_date", "정보 없음"),
            "review_year": employee_info.get("review_year", "정보 없음"),
            "review_month": employee_info.get("review_month", "정보 없음"),
            "year_category": self._get_year_category(employee_info.get("review_year")),
            "sentence_count": len(title_sentences),
            "provides_context": True,
            "applies_to_all_categories": True,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return CategoryChunk(
            id=chunk_id,
            category="general",
            company=company,
            content=content,
            rating=0.0,
            metadata=metadata,
            created_at=datetime.now().isoformat()
        )
    
    def _parse_employee_info(self, employee_raw_text: str) -> Dict[str, str]:
        """직원 정보 파싱"""
        parsed_info = {
            "employment_status": "정보 없음",
            "position": "정보 없음",
            "review_date": "정보 없음",
            "review_year": "정보 없음",
            "review_month": "정보 없음"
        }
        
        if not employee_raw_text:
            return parsed_info
        
        try:
            date_pattern = r'(\d{4})\.(\d{2})\.(\d{2})'
            date_match = re.search(date_pattern, employee_raw_text)
            if date_match:
                year, month, day = date_match.groups()
                parsed_info["review_date"] = f"{year}.{month}.{day}"
                parsed_info["review_year"] = year
                parsed_info["review_month"] = month
            
            parts = employee_raw_text.split('·')
            if len(parts) >= 1:
                parsed_info["employment_status"] = parts[0].strip()
            if len(parts) >= 3:
                raw_position = parts[2].strip()
                position_clean = re.sub(r'\s*-\s*\d{4}\.\d{2}\.\d{2}', '', raw_position)
                parsed_info["position"] = position_clean.strip()
                
        except Exception as e:
            logger.warning(f"직원 정보 파싱 실패: {e}")
        
        return parsed_info
    
    def _get_year_category(self, year_str: str) -> str:
        """연도 카테고리 분류"""
        try:
            year = int(year_str) if year_str and year_str != "정보 없음" else 0
            if year >= 2024:
                return "최신"
            elif year >= 2022:
                return "최근"
            elif year >= 2020:
                return "과거"
            else:
                return "오래됨"
        except (ValueError, TypeError):
            return "알수없음"
    
    def _categorize_rating(self, rating: float) -> str:
        """평점 카테고리화"""
        if rating >= 4.0:
            return "high"
        elif rating >= 3.0:
            return "medium"
        else:
            return "low"


class VectorDBOptimizer:
    """벡터 DB 저장 최적화"""
    
    def __init__(self):
        self.max_chunk_size = 3000
        self.min_chunk_size = 50
    
    def optimize_chunks_for_vectordb(self, all_chunks: Dict[str, List[CategoryChunk]]) -> List[Dict]:
        """청크를 벡터 DB 저장용으로 최적화"""
        
        optimized_chunks = []
        
        for chunk_type, chunks in all_chunks.items():
            for chunk in chunks:
                optimized_chunk = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "metadata": chunk.metadata
                }
                optimized_chunks.append(optimized_chunk)
        
        return optimized_chunks
    
    def save_vectordb_file(self, company: str, optimized_chunks: List[Dict], 
                          output_dir: str = "./data/vectordb") -> str:
        """청크 벡터 DB 파일 저장"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{company}_chunk_first_vectordb.json"
        filepath = os.path.join(output_dir, filename)
        
        # 통계 계산
        total_chunks = len(optimized_chunks)
        multi_category_chunks = len([c for c in optimized_chunks if c["metadata"].get("is_multi_category", False)])
        
        vectordb_data = {
            "metadata": {
                "company": company,
                "total_chunks": total_chunks,
                "multi_category_chunks": multi_category_chunks,
                "single_category_chunks": total_chunks - multi_category_chunks,
                "created_at": datetime.now().isoformat(),
                "version": "3.0",
                "chunking_strategy": "chunk_first_top2_categories"
            },
            "chunks": optimized_chunks
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vectordb_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"청크 우선 벡터 DB 파일 저장 완료: {filepath}")
        return filepath