"""
BlindInsight AI - 연봉/복지 분석 전문 에이전트

🎯 전문 영역: 
- 연봉 및 보상 (급여, 보너스, 인센티브, 스톡옵션)
- 복리후생 및 혜택 (보험, 휴가, 교육지원 등)

🔍 검색 컬렉션:
- salary_benefits (전문) + general (보조)

📊 분석 내용:
- 연봉/복지의 장점과 단점을 구분하여 분석
- 구체적인 리뷰 데이터 기반 인사이트 제공
- 한국어로 자연스러운 분석 결과 생성
"""

from typing import Any, Dict, List, Optional
from ..models.user import UserProfile
from .base import BaseAgent, AgentResult, AgentConfig


class SalaryBenefitsAgent(BaseAgent):
    """
    급여 및 복지 분석 전문 에이전트
    
    🎯 주요 기능:
    - 급여 및 복지 관련 RAG 검색 수행
    - 장점/단점을 구분한 종합적 인사이트 제공
    - 한국어 기반 자연스러운 분석 결과 생성
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """급여 및 복지 에이전트 초기화"""
        super().__init__(name="salary_benefits_agent", config=config)
        
        # 전문 컬렉션 설정 (급여 및 복지 + 일반)
        self.target_collections = ["salary_benefits", "general"]
    
    async def execute(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> AgentResult:
        """
        급여 및 복지 분석 실행
        
        Args:
            query: 분석할 회사명 (예: "네이버")
            context: 추가 컨텍스트 (선택사항)
            user_profile: 사용자 프로필 (선택사항)
            
        Returns:
            AgentResult: 급여 및 복지 분석 결과
        """
        # 에러 처리와 함께 실행
        return await self._execute_with_error_handling(
            self._analyze_salary_benefits,
            query,
            context,
            user_profile
        )
    
    async def _analyze_salary_benefits(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> Dict[str, Any]:
        """
        급여 및 복지 분석 메인 로직 - 장점과 단점을 개별적으로 분석
        
        Args:
            query: 분석할 회사명
            context: 추가 컨텍스트
            user_profile: 사용자 프로필
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        company_name = context.get("company_name") if context else query
        
        # 사용자 키워드 추출 (연봉/복지 관련)
        salary_keywords = ""
        if context:
            salary_keywords = context.get("salary_keywords", "")
        
        # 1. 장점 관련 문서 검색 (긍정적 내용)
        positive_base_query = f"{company_name} 연봉 급여 월급 임금 소득 높음 좋음 만족 보너스 인센티브 성과급 스톡옵션 주식매수선택권 퇴직금 복리후생 복지 혜택 건강보험 4대보험 의료비 지원 건강검진 종합검진 식비 지원 점심 제공 간식 제공 카페테리아 통근버스 교통비 지원 차량 지원 기숙사 사택 주거지원 대출 지원 학자금 지원 자녀 학비 지원 육아휴직 출산휴가 임신출산 지원 동호회 지원 휴양시설 리조트 콘도 지원 경조사비 지원 선물 지급 문화생활 지원 도서구입비 휴가비 지원 재택근무 지원금 노트북 지급 장비 지원"
        
        if salary_keywords.strip():
            positive_documents = await self.retrieve_knowledge_with_keywords(
                base_query=positive_base_query,
                user_keywords=f"{company_name} {salary_keywords} 좋음 긍정적 장점",
                context=context,
                collections=self.target_collections,
                company_name=company_name,
                content_type_filter="pros",
                k=10
            )
        else:
            positive_documents = await self.retrieve_knowledge(
                query=positive_base_query,
                collections=self.target_collections,
                company_name=company_name,
                content_type_filter="pros",
                k=10
            )
        
        # 2. 단점 관련 문서 검색 (부정적 내용)
        negative_base_query = f"{company_name} 연봉 급여 월급 임금 소득 낮음 적음 부족 불만족 아쉬움 보너스 없음 인센티브 적음 성과급 적음 스톡옵션 없음 주식매수선택권 없음 퇴직금 적음 복리후생 부족 복지 미흡 혜택 적음 건강보험 부실 의료비 지원 부족 건강검진 부실 식비 지원 없음 점심 개인부담 간식 없음 카페테리아 없음 통근버스 없음 교통비 지원 없음 차량 지원 없음 기숙사 없음 사택 없음 주거지원 없음 대출 지원 없음 학자금 지원 없음 자녀 학비 지원 없음 육아휴직 사용 어려움 출산휴가 눈치 경조사비 지원 없음 휴양시설 없음 문화생활 지원 없음 재택근무 지원금 없음 장비 지원 부족"
        
        if salary_keywords.strip():
            negative_documents = await self.retrieve_knowledge_with_keywords(
                base_query=negative_base_query,
                user_keywords=f"{company_name} {salary_keywords} 나쁨 부정적 단점",
                context=context,
                collections=self.target_collections,
                company_name=company_name,
                content_type_filter="cons",
                k=10
            )
        else:
            negative_documents = await self.retrieve_knowledge(
                query=negative_base_query,
                collections=self.target_collections,
                company_name=company_name,
                content_type_filter="cons",
                k=10
            )
       
        # 3. 문서 존재 여부 확인
        if not positive_documents and not negative_documents:
            return {
                "company_name": company_name,
                "analysis_type": "급여 및 복지",
                "error": "관련 정보를 찾을 수 없습니다.",
                "confidence_score": 0.0
            }
        
        # 4. 장점 분석 (긍정적 문서가 있는 경우)
        strengths = []
        if positive_documents:
            strengths_prompt = f"""
다음은 {company_name}의 급여 및 복지 장점과 관련된 실제 긍정적 리뷰 데이터입니다.  
이 데이터를 바탕으로 {company_name}의 급여 및 복지 **장점**을 5~7개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.  

**출력 형식:**  
JSON 객체 형태로만 출력하세요.  

{{
  "strengths": [
    "구체적인 장점 1",
    "구체적인 장점 2",
    "구체적인 장점 3",
    ...
    "구체적인 장점 N"
  ],
  "final_summary": "위 장점들을 종합했을 때 {company_name}의 급여 및 복지 전반적인 강점 요약"
}}

**중요:**  
- 반드시 실제 리뷰 데이터에 근거한 구체적인 장점만 작성하세요.  
- 모호하거나 일반적인 표현 대신, 리뷰에서 반복적으로 나타나는 특성을 요약하세요.  
- 최종 결과는 JSON 객체 외의 불필요한 텍스트 없이 제공해야 합니다.  
"""
            try:
                strengths_response = await self.generate_response(
                    prompt=strengths_prompt,
                    context_documents=positive_documents,
                    temperature=1.0,  # 모델이 지원하는 temperature 값 사용
                    max_tokens=10000
                )
                # print(f"[salary_benefits_agent] 장점 분석 응답: {strengths_response[:200]}...")
                
                import json
                if strengths_response and strengths_response.strip():
                    try:
                        strengths = json.loads(strengths_response.strip())
                    except json.JSONDecodeError as je:
                        print(f"JSON 파싱 실패, 대체 파싱 시도: {str(je)}")
                        # JSON이 아닌 경우 대체 파싱
                        if isinstance(strengths_response, str):
                            strengths = {"strengths": [strengths_response], "final_summary": "분석 결과를 정리하지 못했습니다."}
                        else:
                            strengths = {"strengths": ["장점 분석 중 오류 발생"], "final_summary": "분석에 실패했습니다."}
                else:
                    print("빈 응답 받음")
                    strengths = {"strengths": ["응답을 받지 못했습니다."], "final_summary": "분석에 실패했습니다."}
            except Exception as e:
                print(f"장점 분석 실패 돈돈: {str(e)}")
                strengths = {"strengths": ["장점 분석 중 오류 발생"], "final_summary": "분석에 실패했습니다."}
        
        # 5. 단점 분석 (부정적 문서가 있는 경우) 
        weaknesses = []
        if negative_documents:
            weaknesses_prompt = f"""
다음은 {company_name}의 급여 및 복지 단점과 관련된 실제 부정적 리뷰 데이터입니다.  
이 데이터를 바탕으로 {company_name}의 급여 및 복지 **단점**을 5~7개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.  

**출력 형식:**  
JSON 객체 형태로만 출력하세요.  

{{
  "strengths": [
    "구체적인 단점 1",
    "구체적인 단점 2",
    "구체적인 단점 3",
    ...
    "구체적인 단점 N"
  ],
  "final_summary": "위 단점들을 종합했을 때 {company_name}의 급여 및 복지 전반적인 단점 요약"
}}

**중요:**  
- 반드시 실제 리뷰 데이터에 근거한 구체적인 단점만 작성하세요.  
- 모호하거나 일반적인 표현 대신, 리뷰에서 반복적으로 나타나는 특성을 요약하세요.  
- 최종 결과는 JSON 객체 외의 불필요한 텍스트 없이 제공해야 합니다.  

"""
            try:
                weaknesses_response = await self.generate_response(
                    prompt=weaknesses_prompt,
                    context_documents=negative_documents,
                    temperature=1.0,  # 모델이 지원하는 temperature 값 사용
                    max_tokens=10000
                )
                # print(f"[salary_benefits_agent] 단점 분석 응답: {weaknesses_response[:200]}...")
                
                import json
                if weaknesses_response and weaknesses_response.strip():
                    try:
                        weaknesses = json.loads(weaknesses_response.strip())
                    except json.JSONDecodeError as je:
                        print(f"JSON 파싱 실패, 대체 파싱 시도: {str(je)}")
                        # JSON이 아닌 경우 대체 파싱
                        if isinstance(weaknesses_response, str):
                            weaknesses = {"weaknesses": [weaknesses_response], "final_summary": "분석 결과를 정리하지 못했습니다."}
                        else:
                            weaknesses = {"weaknesses": ["단점 분석 중 오류 발생"], "final_summary": "분석에 실패했습니다."}
                else:
                    print("빈 응답 받음")
                    weaknesses = {"weaknesses": ["응답을 받지 못했습니다."], "final_summary": "분석에 실패했습니다."}
            except Exception as e:
                print(f"단점 분석 실패: {str(e)}")
                weaknesses = {"weaknesses": ["단점 분석 중 오류 발생"], "final_summary": "분석에 실패했습니다."}
        
        # 6. 최종 결과 구성 (종합 분석 제외)
        try:
            
            # 최종 결과 구성 (종합 분석 제외)
            result = {
                "company_name": company_name,
                "analysis_type": "급여 및 복지",
                "strengths": strengths,
                "weaknesses": weaknesses,
                "source_documents_count": {
                    "positive": len(positive_documents),
                    "negative": len(negative_documents),
                    "total": len(positive_documents) + len(negative_documents)
                },
                "collections_searched": self.target_collections,
                "analysis_timestamp": str(context.get("timestamp")) if context else None,
                "confidence_score": min(0.9, 0.3 + 0.3 * (len(positive_documents) + len(negative_documents)) / 10)
            }
            
            return result
            
        except Exception as e:
            return {
                "company_name": company_name,
                "analysis_type": "급여 및 복지", 
                "error": f"분석 중 오류 발생: {str(e)}",
                "strengths": strengths,
                "weaknesses": weaknesses,
                "confidence_score": 0.5
            }