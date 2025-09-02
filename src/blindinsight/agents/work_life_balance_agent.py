"""
BlindInsight AI - 업무와 삶의 균형 분석 전문 에이전트

🎯 전문 영역: 
- 워크 라이프 밸런스 (근무시간, 야근, 휴가, 업무강도)
- 근무환경 및 복지제도

🔍 검색 컬렉션:
- work_life_balance (전문) + general (보조)

📊 분석 내용:
- 업무와 삶의 균형의 장점과 단점을 구분하여 분석
- 구체적인 리뷰 데이터 기반 인사이트 제공
- 한국어로 자연스러운 분석 결과 생성
"""

from typing import Any, Dict, List, Optional
from ..models.user import UserProfile
from .base import BaseAgent, AgentResult, AgentConfig


class WorkLifeBalanceAgent(BaseAgent):
    """
    워크 라이프 밸런스 분석 전문 에이전트
    
    🎯 주요 기능:
    - 업무와 삶의 균형 관련 RAG 검색 수행
    - 장점/단점을 구분한 종합적 인사이트 제공
    - 한국어 기반 자연스러운 분석 결과 생성
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """업무와 삶의 균형 에이전트 초기화"""
        super().__init__(name="work_life_balance_agent", config=config)
        
        # 전문 컬렉션 설정 (업무와 삶의 균형 + 일반)
        self.target_collections = ["work_life_balance", "general"]
    
    async def execute(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> AgentResult:
        """
        업무와 삶의 균형 분석 실행
        
        Args:
            query: 분석할 회사명 (예: "카카오")
            context: 추가 컨텍스트 (선택사항)
            user_profile: 사용자 프로필 (선택사항)
            
        Returns:
            AgentResult: 업무와 삶의 균형 분석 결과
        """
        # 에러 처리와 함께 실행
        return await self._execute_with_error_handling(
            self._analyze_work_life_balance,
            query,
            context,
            user_profile
        )
    
    async def _analyze_work_life_balance(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> Dict[str, Any]:
        """
        업무와 삶의 균형 분석 메인 로직 - 장점과 단점을 개별적으로 분석
        
        Args:
            query: 분석할 회사명
            context: 추가 컨텍스트
            user_profile: 사용자 프로필
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        company_name = context.get("company_name") if context else query
        
        # 1. 장점 관련 문서 검색 (긍정적 내용)
        positive_documents = await self.retrieve_knowledge(
            query=f"{company_name} 워라밸 업무시간 정시퇴근 야근 없음 주말근무 없음 유연근무 재택근무 하이브리드 탄력근무 휴가 활용 쉬움 연차 자유롭게 휴가제도 좋음 개인시간 보장 프라이빗 시간 충분 개인생활 존중 가족시간 보장 취미활동 시간 여가시간 충분 스트레스 적음 업무강도 적정 일과 삶 조화 건강한 근무환경 퇴근 후 자유 업무 압박 적음 충분한 휴식 여유로운 일정 개인시간 확보",
            collections=self.target_collections,
            company_name=company_name,
            content_type_filter="pros",
            k=10
        )
        
        # 2. 단점 관련 문서 검색 (부정적 내용)
        negative_documents = await self.retrieve_knowledge(
            query=f"{company_name} 워라밸 나쁨 업무시간 길음 야근 많음 늦은 퇴근 주말근무 토요일 출근 일요일 출근 유연근무 없음 재택근무 불가 하이브리드 안됨 휴가 사용 어려움 연차 눈치 휴가제도 나쁨 개인시간 부족 프라이빗 시간 없음 개인생활 침해 가족시간 부족 취미활동 못함 여가시간 없음 스트레스 많음 업무강도 높음 일과 삶 불균형 피곤함 번아웃 과로 업무 압박 심함 휴식 부족 바쁜 일정 개인시간 확보 어려움 회식 강요 업무 연락 퇴근 후에도",
            collections=self.target_collections,
            company_name=company_name,
            content_type_filter="cons",
            k=10
        )
        

        # 3. 문서 존재 여부 확인
        if not positive_documents and not negative_documents:
            return {
                "company_name": company_name,
                "analysis_type": "업무와 삶의 균형",
                "error": "관련 정보를 찾을 수 없습니다.",
                "confidence_score": 0.0
            }
        
        # 4. 장점 분석 (긍정적 문서가 있는 경우)
        strengths = []
        if positive_documents:
            strengths_prompt = f"""
다음은 {company_name}의업무와 삶의 균형 장점과 관련된 실제 부정적 리뷰 데이터입니다.  
이 데이터를 바탕으로 {company_name}의업무와 삶의 균형 **장점**을 5~7개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.

**출력 형식:**
JSON 객체 형태로만 출력하세요.

{{
  "strengths": [
    "구체적인 장점 1",
    "구체적인 장점 2",
    ...
    "구체적인 장점 N"
  ],
  "final_summary": "위 장점들을 종합했을 때 {company_name}의업무와 삶의 균형 전반적인 장점 요약"
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
                # print(f"[management_agent] 장점 분석 응답: {strengths_response[:200]}...")
                
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
                print(f"장점 분석 실패 워라벨: {str(e)}")
                strengths = {"strengths": ["장점 분석 중 오류 발생"], "final_summary": "분석에 실패했습니다."}
        
        # 5. 단점 분석 (부정적 문서가 있는 경우) 
        weaknesses = []
        if negative_documents:
            weaknesses_prompt = f"""
다음은 {company_name}의업무와 삶의 균형 단점과 관련된 실제 부정적 리뷰 데이터입니다.  
이 데이터를 바탕으로 {company_name}의업무와 삶의 균형 **단점**을 5~7개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.

**출력 형식:**
JSON 객체 형태로만 출력하세요.

{{
  "strengths": [
    "구체적인 단점 1",
    "구체적인 단점 2",
    ...
    "구체적인 단점 N"
  ],
  "final_summary": "위 단점들을 종합했을 때 {company_name}의업무와 삶의 균형 전반적인 단점 요약"
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
                # print(f"[management_agent] 단점 분석 응답: {weaknesses_response[:200]}...")
                
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
                "analysis_type": "업무와 삶의 균형",
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
                "analysis_type": "업무와 삶의 균형", 
                "error": f"분석 중 오류 발생: {str(e)}",
                "strengths": strengths,
                "weaknesses": weaknesses,
                "confidence_score": 0.5
            }