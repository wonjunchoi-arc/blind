"""
에이전트에서 RAG 시스템 활용 예제

BlindInsight AI 에이전트들이 새로운 RAG 시스템을 어떻게 활용할 수 있는지 보여주는 예제입니다.
"""

import asyncio
from typing import List, Dict, Any, Optional

from .knowledge_base import KnowledgeBase
from .retriever import SearchResult


class EnhancedRAGAgent:
    """
    향상된 RAG 시스템을 사용하는 에이전트 예제
    
    기존 에이전트들(CultureAnalysisAgent, CompensationAnalysisAgent 등)에서
    이 패턴을 사용하여 더 정확한 정보 검색이 가능합니다.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        에이전트 초기화
        
        Args:
            knowledge_base: KnowledgeBase 인스턴스
        """
        self.kb = knowledge_base
    
    async def retrieve_company_culture_info(
        self, 
        company_name: str, 
        specific_aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        회사 문화 정보 검색 (CultureAnalysisAgent에서 활용)
        
        Args:
            company_name: 분석할 회사명
            specific_aspects: 특정 문화 측면들 (워라밸, 분위기, 복지 등)
            
        Returns:
            문화 분석 결과
        """
        try:
            # 1. 기본 문화 정보 검색
            culture_queries = [
                f"{company_name} 회사 문화 분위기",
                f"{company_name} 워라밸 업무환경",
                f"{company_name} 복지 혜택"
            ]
            
            if specific_aspects:
                # 특정 측면에 대한 쿼리 추가
                for aspect in specific_aspects:
                    culture_queries.append(f"{company_name} {aspect}")
            
            # 2. 리뷰 데이터에서 문화 관련 정보 검색
            all_results = []
            for query in culture_queries:
                results = await self.kb.search_reviews(
                    query=query,
                    company_name=company_name,
                    review_aspects=["culture", "management"],
                    k=5
                )
                all_results.extend(results)
            
            # 3. 결과 분석 및 구조화
            culture_analysis = {
                "company": company_name,
                "culture_aspects": {},
                "key_insights": [],
                "overall_sentiment": "neutral",
                "evidence_count": len(all_results)
            }
            
            # 문화 측면별 분류
            aspect_keywords = {
                "work_life_balance": ["워라밸", "근무시간", "야근", "휴가"],
                "company_atmosphere": ["분위기", "환경", "동료", "상사"],
                "benefits": ["복지", "혜택", "급식", "교육"],
                "growth_opportunities": ["성장", "승진", "교육", "기회"],
                "management_style": ["경영진", "상사", "관리", "소통"]
            }
            
            for aspect, keywords in aspect_keywords.items():
                aspect_results = []
                positive_count = 0
                negative_count = 0
                
                for result in all_results:
                    content = result.document.page_content.lower()
                    if any(keyword in content for keyword in keywords):
                        aspect_results.append(result)
                        
                        # 간단한 감정 분석
                        positive_words = ["좋", "만족", "훌륭", "추천"]
                        negative_words = ["나쁘", "불만", "힘들", "스트레스"]
                        
                        pos_score = sum(1 for word in positive_words if word in content)
                        neg_score = sum(1 for word in negative_words if word in content)
                        
                        if pos_score > neg_score:
                            positive_count += 1
                        elif neg_score > pos_score:
                            negative_count += 1
                
                if aspect_results:
                    culture_analysis["culture_aspects"][aspect] = {
                        "result_count": len(aspect_results),
                        "positive_mentions": positive_count,
                        "negative_mentions": negative_count,
                        "sentiment_ratio": positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0.5,
                        "top_evidence": [
                            {
                                "content": result.document.page_content[:200] + "...",
                                "score": result.relevance_score,
                                "metadata": result.document.metadata
                            }
                            for result in aspect_results[:3]
                        ]
                    }
            
            # 4. 주요 인사이트 추출
            if all_results:
                culture_analysis["key_insights"] = [
                    f"총 {len(all_results)}개의 관련 리뷰 발견",
                    f"문화 관련 주요 측면: {', '.join(culture_analysis['culture_aspects'].keys())}",
                ]
                
                # 전체적인 감정 분석
                total_positive = sum(aspect["positive_mentions"] for aspect in culture_analysis["culture_aspects"].values())
                total_negative = sum(aspect["negative_mentions"] for aspect in culture_analysis["culture_aspects"].values())
                
                if total_positive > total_negative:
                    culture_analysis["overall_sentiment"] = "positive"
                elif total_negative > total_positive:
                    culture_analysis["overall_sentiment"] = "negative"
                else:
                    culture_analysis["overall_sentiment"] = "mixed"
            
            return culture_analysis
            
        except Exception as e:
            return {
                "company": company_name,
                "error": f"문화 정보 검색 중 오류: {str(e)}",
                "culture_aspects": {},
                "key_insights": [],
                "overall_sentiment": "unknown",
                "evidence_count": 0
            }
    
    async def retrieve_salary_info(
        self, 
        company_name: str, 
        position_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        연봉 정보 검색 (CompensationAnalysisAgent에서 활용)
        
        Args:
            company_name: 분석할 회사명
            position_filter: 직무 필터 (개발자, 마케팅 등)
            
        Returns:
            연봉 분석 결과
        """
        try:
            # 1. 연봉 관련 쿼리 구성
            salary_queries = [
                f"{company_name} 연봉 급여",
                f"{company_name} 보너스 인센티브",
                f"{company_name} 복지 혜택"
            ]
            
            if position_filter:
                salary_queries.append(f"{company_name} {position_filter} 연봉")
            
            # 2. 연봉 관련 검색
            all_results = []
            for query in salary_queries:
                results = await self.kb.search_reviews(
                    query=query,
                    company_name=company_name,
                    review_aspects=["salary", "benefits"],
                    k=10
                )
                all_results.extend(results)
            
            # 3. 연봉 정보 추출 및 분석
            salary_analysis = {
                "company": company_name,
                "position_filter": position_filter,
                "salary_mentions": [],
                "benefits_info": [],
                "satisfaction_level": "unknown",
                "evidence_count": len(all_results)
            }
            
            # 연봉 관련 키워드와 패턴 추출
            import re
            salary_patterns = [
                r'(\d{2,4})\s*만원',      # 3000만원
                r'(\d{1,2})\s*천',        # 5천
                r'연봉\s*(\d+)',          # 연봉 5000
                r'월급\s*(\d+)',          # 월급 300
            ]
            
            benefits_keywords = ["복지", "보험", "점심", "교육비", "휴가", "리프레시"]
            
            for result in all_results:
                content = result.document.page_content
                metadata = result.document.metadata
                
                # 연봉 정보 추출
                for pattern in salary_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        salary_analysis["salary_mentions"].append({
                            "amount": match,
                            "context": content[:100] + "...",
                            "position": metadata.get("position", "unknown"),
                            "date": metadata.get("review_date", "unknown"),
                            "rating": metadata.get("salary_rating", 0)
                        })
                
                # 복지 정보 추출
                content_lower = content.lower()
                for keyword in benefits_keywords:
                    if keyword in content_lower:
                        salary_analysis["benefits_info"].append({
                            "benefit_type": keyword,
                            "context": content[:150] + "...",
                            "sentiment": "positive" if any(pos in content_lower for pos in ["좋", "만족", "훌륭"]) else "neutral"
                        })
            
            # 4. 만족도 분석
            salary_ratings = []
            for result in all_results:
                metadata = result.document.metadata
                if metadata.get("salary_rating"):
                    try:
                        rating = float(metadata["salary_rating"])
                        salary_ratings.append(rating)
                    except ValueError:
                        pass
            
            if salary_ratings:
                avg_rating = sum(salary_ratings) / len(salary_ratings)
                if avg_rating >= 4:
                    salary_analysis["satisfaction_level"] = "high"
                elif avg_rating >= 3:
                    salary_analysis["satisfaction_level"] = "medium"
                else:
                    salary_analysis["satisfaction_level"] = "low"
                
                salary_analysis["average_salary_rating"] = avg_rating
            
            return salary_analysis
            
        except Exception as e:
            return {
                "company": company_name,
                "error": f"연봉 정보 검색 중 오류: {str(e)}",
                "salary_mentions": [],
                "benefits_info": [],
                "satisfaction_level": "unknown",
                "evidence_count": 0
            }
    
    async def retrieve_growth_opportunities(
        self, 
        company_name: str
    ) -> Dict[str, Any]:
        """
        성장 기회 정보 검색 (GrowthStabilityAgent에서 활용)
        
        Args:
            company_name: 분석할 회사명
            
        Returns:
            성장 분석 결과
        """
        try:
            # 1. 성장 관련 쿼리
            growth_queries = [
                f"{company_name} 승진 기회",
                f"{company_name} 교육 성장",
                f"{company_name} 커리어 발전",
                f"{company_name} 스킬업 역량개발"
            ]
            
            # 2. 성장 관련 검색
            all_results = []
            for query in growth_queries:
                results = await self.kb.search_reviews(
                    query=query,
                    company_name=company_name,
                    review_aspects=["growth", "career"],
                    k=8
                )
                all_results.extend(results)
            
            # 3. 성장 기회 분석
            growth_analysis = {
                "company": company_name,
                "growth_aspects": {
                    "promotion_opportunities": [],
                    "skill_development": [],
                    "learning_culture": [],
                    "mentorship": []
                },
                "overall_growth_score": 0.0,
                "evidence_count": len(all_results)
            }
            
            # 성장 관련 키워드 분류
            aspect_keywords = {
                "promotion_opportunities": ["승진", "팀장", "리드", "시니어"],
                "skill_development": ["교육", "스킬", "기술", "역량"],
                "learning_culture": ["배움", "학습", "세미나", "컨퍼런스"],
                "mentorship": ["멘토", "선배", "가이드", "코칭"]
            }
            
            for result in all_results:
                content = result.document.page_content.lower()
                metadata = result.document.metadata
                
                for aspect, keywords in aspect_keywords.items():
                    if any(keyword in content for keyword in keywords):
                        growth_analysis["growth_aspects"][aspect].append({
                            "content": result.document.page_content[:150] + "...",
                            "score": result.relevance_score,
                            "position": metadata.get("position", "unknown"),
                            "has_positive_growth": metadata.get("has_positive_growth", False)
                        })
            
            # 4. 전체 성장 점수 계산
            total_mentions = sum(len(aspects) for aspects in growth_analysis["growth_aspects"].values())
            positive_growth_count = sum(
                1 for aspects in growth_analysis["growth_aspects"].values()
                for mention in aspects
                if mention.get("has_positive_growth", False)
            )
            
            if total_mentions > 0:
                growth_analysis["overall_growth_score"] = (positive_growth_count / total_mentions) * 5.0  # 5점 만점
            
            return growth_analysis
            
        except Exception as e:
            return {
                "company": company_name,
                "error": f"성장 정보 검색 중 오류: {str(e)}",
                "growth_aspects": {},
                "overall_growth_score": 0.0,
                "evidence_count": 0
            }


async def demonstrate_rag_usage():
    """RAG 시스템 사용 예제 시연"""
    
    print("🚀 BlindInsight RAG 시스템 사용 예제")
    print("=" * 50)
    
    # 1. 지식 베이스 초기화
    print("1. 지식 베이스 초기화 중...")
    kb = KnowledgeBase()
    
    # 2. 리뷰 데이터 로드 (처음 한 번만 실행)
    print("2. 리뷰 데이터 로드 중...")
    # success = await kb.load_review_data("tools/data/reviews")
    # if success:
    #     print("✅ 리뷰 데이터 로드 완료")
    # else:
    #     print("❌ 리뷰 데이터 로드 실패")
    #     return
    
    # 3. 에이전트 초기화
    print("3. 향상된 RAG 에이전트 초기화...")
    agent = EnhancedRAGAgent(kb)
    
    # 4. 문화 분석 예제
    print("\n4. 네이버 문화 분석 예제:")
    culture_result = await agent.retrieve_company_culture_info(
        company_name="네이버",
        specific_aspects=["워라밸", "분위기"]
    )
    
    print(f"   총 {culture_result['evidence_count']}개 증거 발견")
    print(f"   전체 감정: {culture_result['overall_sentiment']}")
    print(f"   문화 측면: {list(culture_result['culture_aspects'].keys())}")
    
    # 5. 연봉 분석 예제
    print("\n5. 카카오 연봉 분석 예제:")
    salary_result = await agent.retrieve_salary_info(
        company_name="카카오",
        position_filter="개발자"
    )
    
    print(f"   총 {salary_result['evidence_count']}개 증거 발견")
    print(f"   만족도: {salary_result['satisfaction_level']}")
    print(f"   연봉 언급: {len(salary_result['salary_mentions'])}건")
    
    # 6. 성장 기회 분석 예제
    print("\n6. 토스 성장 기회 분석 예제:")
    growth_result = await agent.retrieve_growth_opportunities("토스")
    
    print(f"   총 {growth_result['evidence_count']}개 증거 발견")
    print(f"   성장 점수: {growth_result['overall_growth_score']:.1f}/5.0")
    print(f"   성장 측면: {list(growth_result['growth_aspects'].keys())}")
    
    print("\n✅ RAG 시스템 사용 예제 완료!")


if __name__ == "__main__":
    # 예제 실행
    asyncio.run(demonstrate_rag_usage())