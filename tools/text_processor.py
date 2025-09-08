# text_processor.py
"""
텍스트 정규화 및 OpenAI 대용량 배치 분류 모듈 v3.2
- 배치 크기 최대 50개로 확대
- API 호출 최적화 및 비용 절약
- 대용량 데이터 처리 성능 향상
- 개선된 에러 처리 및 폴백 메커니즘

설치 필요한 패키지:
pip install hanspell openai python-dotenv tqdm
"""

import re
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from tqdm import tqdm

try:
    from hanspell import spell_checker
except ImportError:
    spell_checker = None

try:
    import openai
    from openai import OpenAI
except ImportError:
    openai = None

# 환경변수 로드
load_dotenv()

# 로깅 설정 간소화
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

@dataclass
class CategoryResult:
    """카테고리 분류 결과"""
    primary_category: str
    primary_confidence: float
    secondary_category: str = ""
    secondary_confidence: float = 0.0

class TextNormalizer:
    """텍스트 정규화 처리기"""
    
    def __init__(self, enable_spell_check=True):
        self.enable_spell_check = enable_spell_check and spell_checker is not None
        
        # 정규화 패턴
        self.patterns = {
            'html_tags': re.compile(r'<[^>]+>'),
            'special_chars': re.compile(r'[^\w\sㄱ-ㅎㅏ-ㅣ가-힣.,!?()-]'),
            'multiple_spaces': re.compile(r'\s+'),
            'multiple_punct': re.compile(r'([.!?]){2,}'),
            'unnecessary_symbols': re.compile(r'[ㄱ-ㅎㅏ-ㅣ]+')
        }
    
    def normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # 기본 정리
            normalized = self._basic_cleanup(text)
            
            # 맞춤법 검사 (100자 이하만, 조용하게 처리)
            if self.enable_spell_check and 5 < len(normalized) < 200:
                try:
                    result = spell_checker.check(normalized)
                    normalized = result.checked
                except:
                    pass  # 실패시 원본 사용
            
            return self._final_cleanup(normalized)
            
        except Exception:
            return self._basic_cleanup(text)
    
    def _basic_cleanup(self, text: str) -> str:
        """기본 정리"""
        text = self.patterns['html_tags'].sub('', text)
        text = self.patterns['unnecessary_symbols'].sub('', text)
        text = self.patterns['special_chars'].sub(' ', text)
        text = self.patterns['multiple_punct'].sub(r'\1', text)
        text = self.patterns['multiple_spaces'].sub(' ', text)
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """최종 정리"""
        text = self.patterns['multiple_spaces'].sub(' ', text)
        text = text.strip()
        text = re.sub(r'\(\s*\)', '', text)
        text = re.sub(r'\[\s*\]', '', text)
        return text

class OptimizedBatchClassifier:
    """최적화된 OpenAI 대용량 배치 분류기"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini-2024-07-18"):
        if not openai:
            raise ImportError("openai 라이브러리가 필요합니다")
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API 키가 필요합니다")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # 5개 카테고리 정의
        self.categories = {
            "career_growth": "커리어 향상",
            "salary_benefits": "급여 복지",  
            "work_life_balance": "워라밸",
            "company_culture": "사내문화",
            "management": "경영진"
        }
        
        # 최적화된 배치 설정
        self.default_batch_size = 30  # 대용량 배치 크기
        self.max_retries = 3
        self.retry_delay = 2
        
        # 통계 추적
        self.total_api_calls = 0
        self.total_tokens_used = 0
        self.successful_batches = 0
        self.failed_batches = 0
    
    def classify_chunks_batch(self, chunks: List[str], batch_size: int = None) -> List[CategoryResult]:
        """대용량 청크 배치 분류"""
        if not chunks:
            return []
        
        # 배치 크기 설정
        effective_batch_size = batch_size or self.default_batch_size
        
        all_results = []
        total_batches = (len(chunks) + effective_batch_size - 1) // effective_batch_size
        
        print(f"🔍 대용량 배치 분류 시작:")
        print(f"   - 총 청크: {len(chunks)}개")
        print(f"   - 배치 크기: {effective_batch_size}개")
        print(f"   - 예상 배치 수: {total_batches}개")
        
        for i in range(0, len(chunks), effective_batch_size):
            batch = chunks[i:i + effective_batch_size]
            current_batch_num = i // effective_batch_size + 1
            
            try:
                batch_results = self._process_large_batch(batch, current_batch_num, total_batches)
                all_results.extend(batch_results)
                self.successful_batches += 1
                
            except Exception as e:
                print(f"\n⚠️ 배치 {current_batch_num} 처리 실패: {str(e)[:100]}...")
                # 폴백 처리
                fallback_results = [self._create_fallback_result() for _ in batch]
                all_results.extend(fallback_results)
                self.failed_batches += 1
            
            # API 레이트 제한 고려
            if i + effective_batch_size < len(chunks):
                print(f"⏳ API 레이트 제한으로 1초 대기...")
                time.sleep(1)
        
        print(f"\n📊 배치 분류 완료:")
        print(f"   - 성공한 배치: {self.successful_batches}개")
        print(f"   - 실패한 배치: {self.failed_batches}개")
        print(f"   - 총 API 호출: {self.total_api_calls}회")
        
        return all_results
    
    def _process_large_batch(self, batch: List[str], batch_num: int, total_batches: int) -> List[CategoryResult]:
        """대용량 배치 단일 처리"""
        
        print(f"\n🔍 배치 {batch_num}/{total_batches} 처리 시작 ({len(batch)}개 청크)")
        batch_start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                # 1단계: 프롬프트 생성
                print(f"  📝 1단계: 프롬프트 생성 중...")
                start_time = time.time()
                
                prompt = self._create_optimized_batch_prompt(batch)
                prompt_time = time.time() - start_time
                print(f"  ✅ 프롬프트 생성 완료 ({prompt_time:.3f}s)")
                
                # 2단계: API 호출 (가장 시간이 오래 걸리는 부분)
                print(f"  🤖 2단계: OpenAI API 호출 중... (모델: {self.model})")
                api_start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_optimized_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=5000,
                    timeout=60  # 타임아웃 설정
                )
                
                api_time = time.time() - api_start_time
                print(f"  ✅ API 호출 완료 ({api_time:.1f}s) - 토큰: {response.usage.total_tokens}")
                
                self.total_api_calls += 1
                self.total_tokens_used += response.usage.total_tokens
                
                # 3단계: 응답 파싱
                print(f"  🔧 3단계: 응답 파싱 중...")
                parse_start_time = time.time()
                
                result_text = response.choices[0].message.content.strip()
                results = self._parse_optimized_batch_response(result_text, len(batch))
                
                parse_time = time.time() - parse_start_time
                print(f"  ✅ 응답 파싱 완료 ({parse_time:.3f}s) - 결과: {len(results)}개")
                
                # 4단계: 결과 검증
                print(f"  ✔️ 4단계: 결과 검증 중...")
                validation_start_time = time.time()
                
                # 결과 검증
                if len(results) == len(batch):
                    validation_time = time.time() - validation_start_time
                    batch_total_time = time.time() - batch_start_time
                    
                    print(f"  ✅ 결과 검증 완료 ({validation_time:.3f}s)")
                    print(f"🎉 배치 {batch_num} 완료! 총 시간: {batch_total_time:.1f}s")
                    
                    # 첫 배치만 상세 분석 출력
                    if batch_num == 1:
                        print(f"\n📊 배치 처리 시간 분석:")
                        print(f"   - 프롬프트 생성: {prompt_time:.3f}s ({prompt_time/batch_total_time*100:.1f}%)")
                        print(f"   - API 호출: {api_time:.1f}s ({api_time/batch_total_time*100:.1f}%) ⚠️ [주요 병목]")
                        print(f"   - 응답 파싱: {parse_time:.3f}s ({parse_time/batch_total_time*100:.1f}%)")
                        print(f"   - 결과 검증: {validation_time:.3f}s ({validation_time/batch_total_time*100:.1f}%)")
                        print(f"   - 총 시간: {batch_total_time:.1f}s")
                    
                    return results
                else:
                    # 부족한 결과는 폴백으로 채움
                    print(f"  ⚠️ 결과 부족 ({len(results)}/{len(batch)}) - 폴백으로 보정")
                    while len(results) < len(batch):
                        results.append(self._create_fallback_result())
                    
                    validation_time = time.time() - validation_start_time
                    batch_total_time = time.time() - batch_start_time
                    
                    print(f"  ✅ 폴백 보정 완료 ({validation_time:.3f}s)")
                    print(f"🎉 배치 {batch_num} 완료! 총 시간: {batch_total_time:.1f}s")
                    
                    return results[:len(batch)]
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"\n❌ 배치 {batch_num} 시도 {attempt+1} 실패: {str(e)[:80]}...")
                    retry_delay = self.retry_delay * (attempt + 1)
                    print(f"⏳ {retry_delay}초 후 재시도...")
                    time.sleep(retry_delay)
                else:
                    print(f"\n💥 배치 {batch_num} 모든 시도 실패 - 폴백 처리")
                    raise e
        
        # 모든 시도 실패시 폴백
        return [self._create_fallback_result() for _ in batch]
    
    def _get_optimized_system_prompt(self) -> str:
        """최적화된 시스템 프롬프트"""
        return 
    """당신은 블라인드(Blind) 기업 리뷰 분류 전문 AI입니다.  
주어진 리뷰 텍스트를 빠르고 정확하게 아래 5개 카테고리 중 하나로만 분류하세요.  

카테고리 정의:  
1. career_growth → 승진, 교육, 성장, 개발, 커리어 기회, 스킬 향상 관련  
2. salary_benefits → 급여, 연봉, 복지, 보너스, 인센티브, 휴가, 보험, 복리후생 관련  
3. work_life_balance → 근무시간, 야근, 워라밸, 휴일, 스트레스, 피로 관련  
4. company_culture → 회사 분위기, 조직문화, 동료 관계, 인간관계, 소통, 협업 관련  
5. management → 경영진, 상사, 관리자, 리더십, 의사결정, 방향성, 비전 관련  

분류 규칙:  
- 각 리뷰당 반드시 1개의 primary_category만 선택  
- primary_confidence는 0.1~1.0 범위의 소수점 수치  
  - 확실한 경우: 0.9 이상  
  - 중간 정도 확신: 0.6 ~ 0.8  
  - 애매한 경우: 0.5 이하  
- 내부적으로 단계적으로 생각해 근거를 판단한 뒤, 최종 응답은 **JSON 배열만 출력**  
- 입력 텍스트 순서와 출력 순서 반드시 동일  

응답 형식 (예시):  
[{"primary_category": "salary_benefits", "primary_confidence": 0.85}]

---

### Few-shot 예시

입력:  
["연봉은 업계 평균 이상이고 복지 제도도 잘 되어 있다."]  

출력:  
[{"primary_category": "salary_benefits", "primary_confidence": 0.95}]  

입력:  
["업무 강도가 높아서 야근이 많고 워라밸이 거의 없다."]  

출력:  
[{"primary_category": "work_life_balance", "primary_confidence": 0.92}]  

---

이제 실제 입력 데이터를 분류하라.
"""
# """당신은 한국 기업 리뷰 분석 전문 AI입니다. 
# 주어진 텍스트들을 빠르고 정확하게 다음 5개 카테고리로 분류하세요:

# 1. career_growth: 승진/교육/성장/개발/커리어/스킬
# 2. salary_benefits: 급여/연봉/복지/보너스/인센티브/휴가/보험
# 3. work_life_balance: 야근/워라밸/근무시간/휴일/스트레스/피로
# 4. company_culture: 분위기/문화/동료/인간관계/소통/조직
# 5. management: 경영진/상사/리더십/의사결정/방향성/비전

# 응답 규칙:
# - 각 텍스트마다 1순위 카테고리 선택
# - primary_confidence는 분류에 대해 너가 생각하는 0.1-1.0 범위 (현실적 수치)
# - JSON 배열로만 응답 (설명 불필요)
# - 텍스트 순서와 결과 순서 일치 필수

# 응답 형식:
# [{"primary_category": "카테고리", "primary_confidence": 0.8}]"""
    
    def _create_optimized_batch_prompt(self, batch: List[str]) -> str:
        """최적화된 배치 프롬프트 생성"""
        
        # 프롬프트 헤더
        prompt = f"다음 {len(batch)}개 텍스트를 각각 분류하세요:\n\n"
        
        # 텍스트 추가 (간결하게)
        for i, chunk in enumerate(batch, 1):
            # 너무 긴 텍스트는 잘라냄 (토큰 절약)
            truncated_chunk = chunk[:200] if len(chunk) > 200 else chunk
            prompt += f"{i}. {truncated_chunk}\n"
        
        # 푸터
        prompt += f"\n{len(batch)}개 결과를 JSON 배열로 응답하세요."
        
        return prompt
    
    def _parse_optimized_batch_response(self, response_text: str, expected_count: int) -> List[CategoryResult]:
        """최적화된 배치 응답 파싱"""
        
        try:
            # JSON 추출 (더 관대한 파싱)
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                
                # JSON 정리 (불완전한 JSON 수정 시도)
                json_str = self._clean_json_string(json_str)
                
                results_json = json.loads(json_str)
                
                # 결과 변환
                results = []
                for item in results_json[:expected_count]:
                    if isinstance(item, dict):
                        result = CategoryResult(
                            primary_category=self._validate_category(
                                item.get("primary_category", "career_growth")
                            ),
                            primary_confidence=self._validate_confidence(
                                item.get("primary_confidence", 0.5)
                            ),
            
                        )
                        results.append(result)
                
                return results
                
        except Exception as e:
            pass
        
        # 파싱 실패시 빈 리스트 반환 (호출자에서 폴백 처리)
        return []
    
    def _clean_json_string(self, json_str: str) -> str:
        """JSON 문자열 정리"""
        # 불완전한 JSON 수정 시도
        json_str = json_str.strip()
        
        # 마지막 쉼표 제거
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        
        # 불완전한 객체 완성 시도
        if json_str.endswith(','):
            json_str = json_str[:-1]
        
        return json_str
    
    def _validate_category(self, category: str) -> str:
        """카테고리 유효성 검사"""
        if category and category in self.categories:
            return category
        return "career_growth"  # 기본값
    
    def _validate_confidence(self, confidence: Any) -> float:
        """신뢰도 유효성 검사"""
        try:
            conf = float(confidence)
            return max(0.1, min(1.0, conf))
        except (ValueError, TypeError):
            return 0.5  # 기본값
    
    def _create_fallback_result(self) -> CategoryResult:
        """폴백 결과 생성"""
        return CategoryResult(
            primary_category="career_growth",
            primary_confidence=0.3,
            secondary_category="company_culture",
            secondary_confidence=0.2
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """분류 통계 반환"""
        total_batches = self.successful_batches + self.failed_batches
        
        stats = {
            "total_api_calls": self.total_api_calls,
            "total_tokens_used": self.total_tokens_used,
            "successful_batches": self.successful_batches,
            "failed_batches": self.failed_batches,
            "success_rate": self.successful_batches / total_batches if total_batches > 0 else 0,
            "avg_tokens_per_call": self.total_tokens_used / self.total_api_calls if self.total_api_calls > 0 else 0,
            "estimated_cost_usd": self.total_tokens_used * 0.00002  # GPT-4o-mini 가격
        }
        
        return stats

class EnhancedTextProcessor:
    """통합 텍스트 처리기 - 배치 최적화 버전"""
    
    def __init__(self, openai_api_key: str = None, enable_spell_check: bool = True):
        self.normalizer = TextNormalizer(enable_spell_check=enable_spell_check)
        self.classifier = OptimizedBatchClassifier(api_key=openai_api_key) if openai_api_key else None
        
        # 간소화된 통계
        self.stats = {
            "texts_processed": 0,
            "batches_processed": 0,
            "classification_successes": 0,
            "classification_failures": 0,
            "total_api_calls_saved": 0
        }
    
    def process_chunks_batch(self, chunks: List[str], batch_size: int = None) -> List[Dict[str, Any]]:
        """대용량 청크 배치 처리"""
        if not chunks:
            return []
        
        # 배치 크기 설정 (None이면 classifier의 기본값 사용)
        if batch_size is None:
            batch_size = self.classifier.default_batch_size if self.classifier else 50
        
        # API 호출 절약 계산 (개별 처리 vs 배치 처리)
        individual_calls = len(chunks)  # 개별 처리시 호출 수
        batch_calls = (len(chunks) + batch_size - 1) // batch_size  # 배치 처리시 호출 수
        api_calls_saved = individual_calls - batch_calls
        
        print(f"💡 API 효율성:")
        print(f"   - 개별 처리시: {individual_calls}회 호출")
        print(f"   - 배치 처리시: {batch_calls}회 호출")
        print(f"   - 절약된 호출: {api_calls_saved}회 ({api_calls_saved/individual_calls*100:.1f}% 절약)")
        
        self.stats["total_api_calls_saved"] = api_calls_saved
        
        # 1단계: 텍스트 정규화 (빠른 처리)
        print("\n📍 1단계: 텍스트 정규화 시작...")
        normalize_start = time.time()
        
        normalized_chunks = []
        for chunk in tqdm(chunks, desc="정규화", unit="청크", ncols=60):
            normalized = self.normalizer.normalize_text(chunk)
            normalized_chunks.append(normalized)
        
        normalize_time = time.time() - normalize_start
        print(f"✅ 1단계 완료: 텍스트 정규화 ({normalize_time:.1f}초)")
        
        # 2단계: 대용량 배치 분류
        print(f"\n📍 2단계: AI 배치 분류 시작... (배치크기: {batch_size})")
        classify_start = time.time()
        
        classification_results = []
        if self.classifier and normalized_chunks:
            try:
                classification_results = self.classifier.classify_chunks_batch(
                    normalized_chunks, batch_size=batch_size
                )
                self.stats["classification_successes"] += len(classification_results)
                self.stats["batches_processed"] += 1
                
            except Exception as e:
                print(f"\n❌ 배치 분류 전체 실패: {str(e)[:100]}...")
                self.stats["classification_failures"] += len(chunks)
                # 전체 폴백 결과 생성
                classification_results = [
                    CategoryResult(primary_category="career_growth", primary_confidence=0.2)
                    for _ in chunks
                ]
        
        classify_time = time.time() - classify_start
        print(f"✅ 2단계 완료: AI 배치 분류 ({classify_time:.1f}초)")
        
        # 3단계: 결과 조합
        results = []
        for i, (original, normalized) in enumerate(zip(chunks, normalized_chunks)):
            classification = classification_results[i] if i < len(classification_results) else CategoryResult(primary_category="career_growth", primary_confidence=0.1)
            
            result = {
                "original_text": original,
                "normalized_text": normalized,
                "text_changed": original != normalized,
                "character_reduction": len(original) - len(normalized),
                "primary_category": classification.primary_category,
                "primary_confidence": classification.primary_confidence,
                "method": "ai_batch"
            }
            results.append(result)
        
        self.stats["texts_processed"] += len(chunks)
        
        # 최종 통계 출력
        self._print_processing_summary()
        
        return results
    
    def _print_processing_summary(self):
        """처리 요약 출력"""
        if self.classifier:
            classifier_stats = self.classifier.get_statistics()
            
            print(f"\n📈 배치 처리 요약:")
            print(f"   - 처리된 텍스트: {self.stats['texts_processed']}개")
            print(f"   - API 호출 절약: {self.stats['total_api_calls_saved']}회")
            print(f"   - 예상 비용 절약: ${self.stats['total_api_calls_saved'] * 0.002:.2f}")
            print(f"   - 실제 API 호출: {classifier_stats['total_api_calls']}회")
            print(f"   - 사용된 토큰: {classifier_stats['total_tokens_used']:,}개")
            print(f"   - 예상 총 비용: ${classifier_stats['estimated_cost_usd']:.4f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """처리 통계 반환"""
        stats = self.stats.copy()
        
        if self.classifier:
            classifier_stats = self.classifier.get_statistics()
            stats.update(classifier_stats)
        
        if stats["texts_processed"] > 0 and (stats["classification_successes"] + stats["classification_failures"]) > 0:
            success_rate = stats["classification_successes"] / (stats["classification_successes"] + stats["classification_failures"])
            stats["classification_success_rate"] = round(success_rate, 3)
        
        return stats
    
    def reset_stats(self):
        """통계 초기화"""
        self.stats = {
            "texts_processed": 0,
            "batches_processed": 0,
            "classification_successes": 0,
            "classification_failures": 0,
            "total_api_calls_saved": 0
        }
        
        if self.classifier:
            self.classifier.total_api_calls = 0
            self.classifier.total_tokens_used = 0
            self.classifier.successful_batches = 0
            self.classifier.failed_batches = 0

# 편의 함수들
def normalize_text(text: str, enable_spell_check: bool = True) -> str:
    """텍스트 정규화 편의 함수"""
    normalizer = TextNormalizer(enable_spell_check=enable_spell_check)
    return normalizer.normalize_text(text)

def classify_chunks_batch_optimized(chunks: List[str], api_key: str = None, batch_size: int = None) -> List[CategoryResult]:
    """최적화된 청크 배치 분류 편의 함수"""
    classifier = OptimizedBatchClassifier(api_key=api_key)
    # batch_size가 None이면 classifier의 기본값(50) 사용
    return classifier.classify_chunks_batch(chunks, batch_size=batch_size)

