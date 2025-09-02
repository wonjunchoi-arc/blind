# 📊 블라인드 크롤링 데이터의 RAG 최적화 전략

## 📋 목차
1. [기본 RAG 구조 (수정 전)](#기본-rag-구조-수정-전)
2. [하이브리드 검색 최적화 구조 (수정 후)](#하이브리드-검색-최적화-구조-수정-후)
3. [실제 데이터 활용을 위한 특화 전략](#실제-데이터-활용을-위한-특화-전략)
4. [구현 가이드](#구현-가이드)

---

## 기본 RAG 구조 (수정 전)

### 1. 기본 데이터 변환 전략

**원본 크롤링 데이터 → RAG 문서 분할**

```python
# 원본 데이터 (1행 = 1리뷰)
{
    "총점": 4.2, "워라밸": 3.5, "급여": 4.5, "문화": 4.0, "경영진": 3.8,
    "제목": "성장할 수 있는 좋은 회사",
    "직원유형": "현직원", 
    "장점": "복지가 좋고 성장 기회가 많음",
    "단점": "업무 강도가 높고 야근이 잦음"
}

# RAG 변환 후 (1리뷰 → 3문서)
Document 1: 종합 평가
Document 2: 장점 중심  
Document 3: 단점 중심
```

### 2. 자연어 컨텍스트 강화

```python
def enrich_rating_context(scores):
    """평점을 자연어로 변환"""
    # 4.2점 → "매우 만족스러운 수준"
    # 3.5점 → "보통 수준" 
    # 2.8점 → "아쉬운 수준"
```

### 3. 메타데이터 구조

```json
{
    "company": "네이버",
    "aspect": "summary", // summary, pros, cons
    "employee_type": "현직원",
    "rating_total": 4.2,
    "rating_category": "high", // high(4.0+), medium(3.0-4.0), low(3.0-)
    "content_type": "company_review"
}
```

### 4. 기본 구조의 한계점

❌ **키워드 검색 부족**
- 벡터 유사도에만 의존
- "야근", "복지" 같은 정확한 키워드 매칭 어려움
- 자연어 변환으로 핵심 키워드 희석

❌ **전문 검색 미지원**
- BM25 같은 전통적 검색 알고리즘 활용 불가
- 하이브리드 점수 계산 로직 없음

---

## 하이브리드 검색 최적화 구조 (수정 후)

### 1. 다층 데이터 구조

```python
def create_hybrid_optimized_document(row):
    return {
        # 벡터 검색용 (의미적 유사도)
        "content_semantic": f"""
        네이버에 대한 현직원 평가: 총점 4.2점으로 높은 만족도.
        급여 복지 4.5점으로 만족스러우나, 워라밸 3.5점으로 개선 필요.
        장점: {enhanced_pros}
        단점: {enhanced_cons}
        """,
        
        # 키워드 검색용 (원본 보존 + 강화)
        "content_keyword": f"""
        제목: {row['제목']}
        장점: {row['장점']} 
        단점: {row['단점']}
        회사: 네이버
        직원유형: {row['직원유형']}
        """,
        
        # 추출된 키워드 및 가중치
        "keywords": {
            "extracted": ["야근", "복지", "성장기회"],
            "weighted": {"야근": 0.85, "복지": 0.92},
            "categories": {
                "worklife": ["야근", "잔업"],
                "benefits": ["복지", "연봉"],  
                "growth": ["성장기회", "승진"]
            },
            "synonyms": {"야근": ["잔업", "오버타임"]}
        },
        
        # 검색 최적화 필드
        "search_fields": {
            "title_text": row['제목'],
            "pros_text": row['장점'],
            "cons_text": row['단점'], 
            "all_text": f"{row['제목']} {row['장점']} {row['단점']}"
        },
        
        "metadata": {
            "company": "네이버",
            "aspect": "pros",
            "employee_type": row['직원유형'],
            "ratings": {...},
            # 추가 메타데이터
            "department": extract_department(row),
            "seniority": extract_seniority(row),
            "sentiment_score": calculate_sentiment(row)
        }
    }
```

### 2. 키워드 추출 및 처리 시스템

```python
class KeywordProcessor:
    def __init__(self):
        # 블라인드 리뷰 특화 키워드 사전
        self.company_keywords = {
            "salary": ["연봉", "급여", "월급", "기본급", "상여금", "보너스", "인센티브"],
            "worklife": ["야근", "잔업", "오버타임", "워라밸", "퇴근시간", "휴가"],
            "culture": ["분위기", "문화", "동료", "상사", "갑질", "회식", "눈치"],
            "growth": ["성장", "승진", "교육", "경력", "기회"],
            "management": ["경영진", "임원", "대표", "리더십"],
            "benefits": ["복지", "보험", "스톡옵션", "주식", "식대", "교통비"]
        }
        
        self.negative_patterns = [
            "눈치", "갑질", "강제", "스트레스", "야근", "삭감", 
            "불만", "힘들", "문제", "아쉬운", "별로"
        ]
        
        self.positive_patterns = [
            "만족", "좋", "훌륭", "최고", "추천", "성장", 
            "기회", "자유", "편리", "여유"
        ]

    def extract_keywords(self, text):
        """고도화된 키워드 추출"""
        keywords = []
        
        # 1. 카테고리별 키워드 매칭
        for category, words in self.company_keywords.items():
            for word in words:
                if word in text:
                    keywords.append({
                        "keyword": word,
                        "category": category,
                        "frequency": text.count(word),
                        "context": self.extract_context(text, word)
                    })
        
        # 2. 감성 키워드 태깅
        sentiment_keywords = self.extract_sentiment_keywords(text)
        
        # 3. 부정 표현 감지
        negative_contexts = self.detect_negative_contexts(text)
        
        return {
            "keywords": keywords,
            "sentiment_keywords": sentiment_keywords, 
            "negative_contexts": negative_contexts
        }

    def extract_context(self, text, keyword):
        """키워드 주변 맥락 추출"""
        import re
        
        # 키워드 전후 10글자씩 추출
        pattern = f".{{0,10}}{re.escape(keyword)}.{{0,10}}"
        matches = re.findall(pattern, text)
        return matches[0] if matches else ""
```

### 3. 하이브리드 검색 엔진

```python
class HybridSearchEngine:
    def __init__(self):
        self.vector_db = ChromaDB()  # 벡터 검색
        self.text_index = ElasticsearchIndex()  # 전문 검색
        self.metadata_db = MetadataIndex()  # 메타데이터 검색
        
    def search(self, query, top_k=10, search_weights=None):
        """하이브리드 검색 실행"""
        
        # 기본 가중치
        if not search_weights:
            search_weights = {"semantic": 0.4, "keyword": 0.4, "metadata": 0.2}
        
        # 1. 쿼리 분석
        parsed_query = self.parse_query(query)
        
        # 2. 다중 검색 실행
        results = {}
        
        # 벡터 검색 (의미적 유사도)
        if search_weights["semantic"] > 0:
            vector_results = self.vector_search(parsed_query["semantic_query"])
            results["vector"] = vector_results
            
        # 키워드 검색 (BM25)
        if search_weights["keyword"] > 0:
            keyword_results = self.keyword_search(parsed_query["keywords"])
            results["keyword"] = keyword_results
            
        # 메타데이터 검색
        if search_weights["metadata"] > 0:
            metadata_results = self.metadata_search(parsed_query["filters"])
            results["metadata"] = metadata_results
        
        # 3. 점수 결합 및 랭킹
        final_results = self.combine_scores(results, search_weights)
        
        return final_results[:top_k]
    
    def parse_query(self, query):
        """쿼리 분석 및 분해"""
        # "네이버 야근 많은가요?" → 
        # {
        #   "company": "네이버",
        #   "keywords": ["야근"],
        #   "intent": "worklife", 
        #   "sentiment": "negative",
        #   "semantic_query": query,
        #   "filters": {"company": "네이버", "aspect": "cons"}
        # }
        
        parsed = {
            "semantic_query": query,
            "keywords": [],
            "filters": {},
            "intent": "general"
        }
        
        # 회사명 추출
        companies = ["네이버", "카카오", "삼성", "LG", "현대"]
        for company in companies:
            if company in query:
                parsed["filters"]["company"] = company
                break
        
        # 키워드 추출
        keyword_patterns = {
            "야근": ["worklife", "negative"],
            "연봉": ["salary", "neutral"], 
            "복지": ["benefits", "positive"],
            "분위기": ["culture", "neutral"]
        }
        
        for keyword, (category, sentiment) in keyword_patterns.items():
            if keyword in query:
                parsed["keywords"].append(keyword)
                parsed["intent"] = category
                if sentiment == "negative":
                    parsed["filters"]["aspect"] = "cons"
                elif sentiment == "positive":
                    parsed["filters"]["aspect"] = "pros"
        
        return parsed
    
    def combine_scores(self, results, weights):
        """다중 검색 결과 점수 결합"""
        combined_scores = {}
        
        # 각 검색 결과를 정규화하고 결합
        for search_type, weight in weights.items():
            if search_type in results and weight > 0:
                for doc_id, score in results[search_type]:
                    if doc_id not in combined_scores:
                        combined_scores[doc_id] = 0
                    combined_scores[doc_id] += weight * score
        
        # 점수 순으로 정렬
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_results
```

### 4. 동의어 및 컨텍스트 확장

```python
class SynonymExpander:
    def __init__(self):
        self.synonym_map = {
            # 워라밸 관련
            "야근": ["잔업", "오버타임", "늦은퇴근", "밤샘"],
            "워라밸": ["업무와삶의균형", "일과삶의균형", "근무시간"],
            "퇴근": ["칼퇴", "정시퇴근", "업무종료"],
            
            # 연봉 관련  
            "연봉": ["급여", "월급", "임금", "페이", "소득"],
            "상여금": ["보너스", "인센티브", "특별급여"],
            "기본급": ["베이스", "고정급", "기본임금"],
            
            # 문화 관련
            "분위기": ["문화", "환경", "분위기"],
            "갑질": ["권위적", "강압적", "일방적"],
            "눈치": ["암묵적강요", "분위기압박"],
            
            # 성장 관련
            "성장": ["발전", "향상", "성장기회", "경력개발"],
            "승진": ["진급", "커리어업", "직급상승"]
        }
    
    def expand_query(self, keywords):
        """키워드를 동의어로 확장"""
        expanded = set(keywords)
        
        for keyword in keywords:
            if keyword in self.synonym_map:
                expanded.update(self.synonym_map[keyword])
        
        return list(expanded)
```

---

## 실제 데이터 활용을 위한 특화 전략

### 1. 진짜 연봉 정보 추출 및 구조화

```python
class SalaryIntelligenceExtractor:
    """연봉 정보의 숨겨진 진실 추출"""
    
    def __init__(self):
        self.salary_patterns = {
            "실수령액": ["세후", "실제받는", "손에쥐는", "실수령", "net"],
            "기본급비율": ["기본급", "베이스", "고정급", "비율"],
            "상여금삭감": ["상여금삭감", "보너스컷", "인센티브없음"],
            "부서차이": ["부서별", "팀별", "직무별", "연봉차이"],
            "인상현실": ["인상률", "연봉협상", "실제인상", "페이크인상"]
        }
        
        self.negative_salary_indicators = [
            "말뿐인", "명목상", "겉으로만", "실제론", "속여서",
            "페이크", "가짜", "눈속임", "삭감", "없어짐"
        ]
    
    def extract_salary_intelligence(self, review_text):
        """연봉 관련 핵심 정보 추출"""
        
        intelligence = {
            "actual_salary_gap": self.detect_actual_vs_nominal(review_text),
            "hidden_structure": self.detect_hidden_salary_structure(review_text),  
            "department_differences": self.detect_department_gaps(review_text),
            "raise_reality": self.detect_raise_reality(review_text)
        }
        
        return intelligence
    
    def detect_actual_vs_nominal(self, text):
        """명목 연봉 vs 실수령액 차이 감지"""
        patterns = [
            r"(\d+천만원||\d+억).*(실제|세후|손에).*(\\d+천만원|\d+백만원)",
            r"(명목|액면).*(실제|진짜).*(차이|갭|다름)",
            r"(말로는|공식적으론).*(실제론|사실은)"
        ]
        
        matches = []
        for pattern in patterns:
            import re
            found = re.findall(pattern, text)
            if found:
                matches.extend(found)
        
        return matches
    
    def detect_hidden_salary_structure(self, text):
        """숨겨진 연봉 구조 파악"""
        structure_indicators = {
            "low_base_ratio": ["기본급이낮", "베이스적", "고정급비율"],
            "bonus_cuts": ["상여금삭감", "보너스컷", "인센티브없앰"],
            "fake_allowances": ["수당명목", "가짜수당", "허울뿐"]
        }
        
        found_structures = {}
        for indicator_type, patterns in structure_indicators.items():
            for pattern in patterns:
                if pattern in text:
                    found_structures[indicator_type] = True
                    break
        
        return found_structures
```

### 2. 워라밸 민낯 분석

```python
class WorkLifeRealityExtractor:
    """워라밸의 숨겨진 진실 추출"""
    
    def __init__(self):
        self.worklife_deception_patterns = {
            "fake_flex_time": ["정시퇴근권장하지만", "눈치게임", "명목상정시"],
            "hidden_overtime": ["카톡업무", "집에서일", "보이지않는야근"],
            "vacation_pressure": ["눈치보며휴가", "자유롭다지만", "실제론못써"],
            "weekend_contact": ["주말연락", "휴일업무", "24시간대기"]
        }
    
    def extract_worklife_reality(self, review_text):
        """워라밸 현실 분석"""
        
        reality_check = {
            "actual_leaving_time": self.detect_actual_leaving_time(review_text),
            "hidden_overtime_culture": self.detect_hidden_overtime(review_text),
            "vacation_reality": self.detect_vacation_pressure(review_text),
            "weekend_intrusion": self.detect_weekend_work(review_text)
        }
        
        return reality_check
    
    def detect_actual_leaving_time(self, text):
        """실제 퇴근시간 패턴 분석"""
        time_patterns = [
            r"(\d{1,2}시).*(퇴근|끝남|마감)",
            r"(정시퇴근|칼퇴).*(눈치|어려|불가능)",
            r"(\d{1,2}시).*(일상|보통|평균)"
        ]
        
        actual_times = []
        for pattern in time_patterns:
            import re
            matches = re.findall(pattern, text)
            actual_times.extend(matches)
        
        return actual_times
    
    def detect_hidden_overtime(self, text):
        """숨겨진 야근 문화 감지"""
        hidden_patterns = [
            "카톡으로업무", "집에서일", "재택이라지만", 
            "보이지않는야근", "숨은업무", "비공식적"
        ]
        
        detected = []
        for pattern in hidden_patterns:
            if pattern in text.replace(" ", ""):
                detected.append(pattern)
        
        return detected
```

### 3. 조직문화 & 인간관계 리얼 분석

```python
class OrganizationCultureAnalyzer:
    """조직문화의 숨겨진 면 분석"""
    
    def __init__(self):
        self.toxic_culture_patterns = {
            "power_abuse": ["갑질", "권위적", "일방적", "강압적"],
            "internal_politics": ["파벌", "줄서기", "정치적", "편가르기"],
            "promotion_unfairness": ["실력무시", "학벌차별", "성별차별", "줄서기승진"],
            "forced_socializing": ["강제회식", "술자리압박", "참여강요"],
            "harassment": ["괴롭힘", "성희롱", "따돌림", "인격모독"]
        }
    
    def analyze_toxic_culture(self, review_text):
        """독성 문화 요소 분석"""
        
        toxic_elements = {}
        
        for culture_type, patterns in self.toxic_culture_patterns.items():
            detected_patterns = []
            for pattern in patterns:
                if pattern in review_text:
                    context = self.extract_context(review_text, pattern, 20)
                    detected_patterns.append({
                        "pattern": pattern,
                        "context": context,
                        "severity": self.assess_severity(context)
                    })
            
            if detected_patterns:
                toxic_elements[culture_type] = detected_patterns
        
        return toxic_elements
    
    def detect_specific_person_issues(self, text):
        """특정 인물 관련 문제 감지"""
        person_patterns = [
            r"(팀장|부장|임원|대표).*(갑질|문제|성격)",
            r"(OO팀장|XX부장).*(힘든|문제)", 
            r"특정.*(상사|임원).*(문제|갑질)"
        ]
        
        person_issues = []
        for pattern in person_patterns:
            import re
            matches = re.findall(pattern, text)
            person_issues.extend(matches)
        
        return person_issues
```

### 4. 회사 내부 사정 인텔리전스

```python
class CompanyIntelligenceAnalyzer:
    """회사 내부 사정 분석"""
    
    def __init__(self):
        self.financial_reality_patterns = {
            "fake_profitability": ["흑자라지만", "실제론적자", "숫자조작"],
            "restructuring_signs": ["구조조정징조", "인력감축", "부서통폐합"],
            "department_discrimination": ["홀대받는부서", "천덕꾸러기", "예산삭감"],
            "labor_conflicts": ["노조갈등", "파업", "노사분규"]
        }
    
    def analyze_internal_intelligence(self, review_text):
        """내부 정보 분석"""
        
        intelligence = {
            "financial_reality": self.detect_financial_issues(review_text),
            "restructuring_signals": self.detect_restructuring_signs(review_text),
            "department_issues": self.detect_department_discrimination(review_text),
            "labor_relations": self.detect_labor_conflicts(review_text)
        }
        
        return intelligence
    
    def detect_financial_issues(self, text):
        """재정 상황 진실 감지"""
        financial_indicators = [
            "실제론적자", "숨겨진부채", "회계조작", "자금난",
            "투자유치실패", "매출하락", "적자전환"
        ]
        
        detected_issues = []
        for indicator in financial_indicators:
            if indicator in text.replace(" ", ""):
                detected_issues.append(indicator)
        
        return detected_issues
```

### 5. 업무환경 진실 분석

```python
class WorkEnvironmentAnalyzer:
    """업무 환경의 현실 분석"""
    
    def __init__(self):
        self.environment_reality_patterns = {
            "workload_intensity": ["무리한업무", "인력부족", "과도한업무"],
            "equipment_issues": ["노후장비", "느린컴퓨터", "시설낙후"],
            "system_problems": ["불편한시스템", "IT인프라부족", "업무효율저하"]
        }
    
    def analyze_work_environment(self, review_text):
        """업무 환경 현실 분석"""
        
        environment_analysis = {
            "actual_workload": self.detect_workload_reality(review_text),
            "equipment_reality": self.detect_equipment_issues(review_text),
            "system_efficiency": self.detect_system_problems(review_text)
        }
        
        return environment_analysis
    
    def detect_workload_reality(self, text):
        """실제 업무 강도 감지"""
        intensity_patterns = [
            r"(\d+시간|\\d+개월).*(연속|계속|무리)",
            r"(여유로워보이지만|편해보이지만).*(실제론|사실은)",
            r"(겉으론|표면적으론).*(내부적으론|실상은)"
        ]
        
        workload_reality = []
        for pattern in intensity_patterns:
            import re
            matches = re.findall(pattern, text)
            workload_reality.extend(matches)
        
        return workload_reality
```

### 6. 통합 하이브리드 검색 전략

```python
class AdvancedHybridSearchStrategy:
    """특화된 하이브리드 검색 전략"""
    
    def __init__(self):
        self.domain_specific_extractors = {
            "salary": SalaryIntelligenceExtractor(),
            "worklife": WorkLifeRealityExtractor(), 
            "culture": OrganizationCultureAnalyzer(),
            "internal": CompanyIntelligenceAnalyzer(),
            "environment": WorkEnvironmentAnalyzer()
        }
    
    def create_specialized_documents(self, review_data):
        """특화된 문서 생성"""
        
        specialized_docs = []
        
        for _, row in review_data.iterrows():
            review_text = f"{row['제목']} {row['장점']} {row['단점']}"
            
            # 기본 문서 생성
            base_docs = self.create_base_documents(row)
            
            # 도메인별 특화 정보 추출
            for domain, extractor in self.domain_specific_extractors.items():
                specialized_info = extractor.analyze(review_text)
                
                if specialized_info:  # 유의미한 정보가 있는 경우만
                    specialized_doc = {
                        "content": self.create_specialized_content(
                            row, domain, specialized_info
                        ),
                        "metadata": {
                            **base_docs[0]["metadata"],
                            "specialization": domain,
                            "intelligence_level": "high",
                            "extracted_insights": specialized_info
                        },
                        "keywords": {
                            **base_docs[0]["keywords"],
                            "specialized_keywords": self.extract_domain_keywords(
                                domain, specialized_info
                            )
                        }
                    }
                    specialized_docs.append(specialized_doc)
        
        return specialized_docs
    
    def search_with_intelligence(self, query, intelligence_focus=None):
        """인텔리전스 중심 검색"""
        
        # 쿼리 의도 분석
        query_intent = self.analyze_query_intent(query)
        
        # 인텔리전스 포커스가 있는 경우 가중치 조정
        if intelligence_focus:
            search_weights = self.adjust_weights_for_intelligence(
                intelligence_focus, query_intent
            )
        else:
            search_weights = {"semantic": 0.3, "keyword": 0.4, "intelligence": 0.3}
        
        # 특화 검색 실행
        results = self.hybrid_search_engine.search(
            query, 
            search_weights=search_weights,
            intelligence_filter=intelligence_focus
        )
        
        return results
    
    def analyze_query_intent(self, query):
        """쿼리 의도 심층 분석"""
        
        intent_patterns = {
            "salary_reality": ["실제연봉", "진짜급여", "세후수령", "실수령액"],
            "worklife_truth": ["진짜워라밸", "실제퇴근시간", "야근현실"],
            "culture_dark": ["갑질", "독성문화", "인간관계", "파벌"],
            "internal_info": ["내부사정", "구조조정", "회사상황", "재정"],
            "environment_real": ["업무강도", "시설현황", "업무환경"]
        }
        
        detected_intents = []
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in query:
                    detected_intents.append(intent)
        
        return detected_intents
    
    def adjust_weights_for_intelligence(self, focus, query_intent):
        """인텔리전스 포커스에 따른 가중치 조정"""
        
        weight_strategies = {
            "salary_reality": {"semantic": 0.2, "keyword": 0.5, "intelligence": 0.3},
            "worklife_truth": {"semantic": 0.3, "keyword": 0.4, "intelligence": 0.3},
            "culture_dark": {"semantic": 0.4, "keyword": 0.3, "intelligence": 0.3},
            "internal_info": {"semantic": 0.2, "keyword": 0.3, "intelligence": 0.5},
            "environment_real": {"semantic": 0.3, "keyword": 0.4, "intelligence": 0.3}
        }
        
        if focus in weight_strategies:
            return weight_strategies[focus]
        else:
            return {"semantic": 0.3, "keyword": 0.4, "intelligence": 0.3}
```

### 7. 실제 활용 시나리오

```python
# 시나리오 1: 연봉 현실 파악
query = "네이버 개발자 실제 연봉 얼마나 받나요?"
results = search_engine.search_with_intelligence(
    query, 
    intelligence_focus="salary_reality"
)

# 시나리오 2: 워라밸 진실 탐구  
query = "카카오 정말 워라밸 좋은가요?"
results = search_engine.search_with_intelligence(
    query,
    intelligence_focus="worklife_truth" 
)

# 시나리오 3: 조직문화 민낯
query = "삼성전자 갑질 문화 어떤가요?"
results = search_engine.search_with_intelligence(
    query,
    intelligence_focus="culture_dark"
)
```

---

## 구현 가이드

### 1. 기술 스택 권장사항

**벡터 데이터베이스**
- **ChromaDB**: 개발/프로토타이핑
- **Pinecone**: 대용량/프로덕션
- **Weaviate**: 하이브리드 검색 특화

**전문 검색**  
- **Elasticsearch**: 고성능 전문 검색
- **OpenSearch**: 오픈소스 대안
- **Whoosh**: 파이썬 네이티브

**임베딩 모델**
- **한국어 특화**: KoBERT, KoSBERT
- **다국어**: multilingual-E5, BGE-M3
- **경량화**: sentence-transformers/all-MiniLM-L6-v2

### 2. 구현 단계

**Phase 1: 기본 RAG 구축**
1. 크롤링 데이터 → 기본 문서 변환
2. 벡터 임베딩 및 저장
3. 간단한 유사도 검색 구현

**Phase 2: 하이브리드 검색 추가**
1. 키워드 추출 시스템 구축
2. 전문 검색 인덱스 생성
3. 점수 결합 알고리즘 구현

**Phase 3: 인텔리전스 특화**
1. 도메인별 특화 추출기 개발
2. 고급 쿼리 분석 시스템
3. 개인화 및 컨텍스트 인식

### 3. 성능 최적화

**인덱싱 최적화**
```python
# 배치 처리로 인덱싱 속도 향상
batch_size = 1000
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    vector_db.add_batch(batch)
    
# 비동기 처리로 응답 속도 향상  
import asyncio

async def hybrid_search_async(query):
    vector_task = asyncio.create_task(vector_search(query))
    keyword_task = asyncio.create_task(keyword_search(query))
    metadata_task = asyncio.create_task(metadata_search(query))
    
    results = await asyncio.gather(vector_task, keyword_task, metadata_task)
    return combine_results(results)
```

**메모리 관리**
```python
# 대용량 데이터 처리를 위한 청킹
def process_large_dataset(data, chunk_size=10000):
    for chunk in pd.read_csv(data, chunksize=chunk_size):
        processed_chunk = process_documents(chunk)
        yield processed_chunk

# 캐싱으로 반복 검색 최적화
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_vector_search(query_hash):
    return vector_search(query_hash)
```

### 4. 품질 평가 지표

**검색 성능 지표**
- **Precision@K**: 상위 K개 결과 중 관련성 있는 비율
- **Recall@K**: 전체 관련 문서 중 상위 K개에서 찾은 비율  
- **MRR (Mean Reciprocal Rank)**: 평균 역순위
- **NDCG (Normalized DCG)**: 순위를 고려한 성능

**하이브리드 검색 평가**
```python
def evaluate_hybrid_search(test_queries, ground_truth):
    metrics = {
        "precision_at_5": [],
        "recall_at_10": [], 
        "mrr": [],
        "hybrid_effectiveness": []
    }
    
    for query, expected_docs in zip(test_queries, ground_truth):
        # 하이브리드 검색 실행
        results = hybrid_search(query, top_k=10)
        
        # 각 지표 계산
        precision = calculate_precision_at_k(results, expected_docs, k=5)
        recall = calculate_recall_at_k(results, expected_docs, k=10)
        mrr = calculate_mrr(results, expected_docs)
        
        metrics["precision_at_5"].append(precision)
        metrics["recall_at_10"].append(recall)
        metrics["mrr"].append(mrr)
    
    return {
        metric: np.mean(values) 
        for metric, values in metrics.items()
    }
```

이 전략을 통해 블라인드 크롤링 데이터에서 **진짜 유용한 인사이트**를 정확하고 효과적으로 검색할 수 있는 고도화된 RAG 시스템을 구축할 수 있습니다.