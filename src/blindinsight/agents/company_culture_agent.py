"""
BlindInsight AI - 사내 문화 분석 전문 에이전트

🎯 전문 영역: 
- 회사 문화 (사내 문화, 조직문화, 사내분위기)
- 회사 전반적인 평가 및 인사이트

🔍 검색 컬렉션:
- company_culture (전문) + general (보조)

📊 분석 내용:
- 사내 문화의 장점과 단점을 구분하여 분석
- 구체적인 리뷰 데이터 기반 인사이트 제공
- 한국어로 자연스러운 분석 결과 생성
"""

from typing import Any, Dict, List, Optional
from ..models.user import UserProfile
from .base import BaseAgent, AgentResult, AgentConfig
import logging



class CompanyCultureAgent(BaseAgent):
    """
    사내 문화 분석 전문 에이전트
    
    🎯 주요 기능:
    - 사내 문화 관련 RAG 검색 수행
    - 장점/단점을 구분한 종합적 인사이트 제공
    - 한국어 기반 자연스러운 분석 결과 생성
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """사내 문화 에이전트 초기화"""
        super().__init__(name="company_culture_agent", config=config)
        
        # 전문 컬렉션 설정 (사내 문화 + 일반)
        self.target_collections = ["company_culture", "general"]
    
    async def execute(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> AgentResult:
        """
        사내 문화 분석 실행
        
        Args:
            query: 분석할 회사명 (예: "네이버")
            context: 추가 컨텍스트 (선택사항)
            user_profile: 사용자 프로필 (선택사항)
            
        Returns:
            AgentResult: 사내 문화 분석 결과
        """
        # 에러 처리와 함께 실행
        return await self._execute_with_error_handling(
            self._analyze_company_culture,
            query,
            context,
            user_profile
        )
    
    async def _analyze_company_culture(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> Dict[str, Any]:
        print('여기여기!!!!!!')
        
        """
        사내 문화 분석 메인 로직 - 장점과 단점을 개별적으로 분석
        
        Args:
            query: 분석할 회사명
            context: 추가 컨텍스트
            user_profile: 사용자 프로필
            
        Returns:
            Dict[str, Any]: 분석 결과
        """
        company_name = context.get("company_name") if context else query
        
        # 사용자 키워드 추출 (기업문화 관련)
        culture_keywords = ""
        if context:
            culture_keywords = context.get("culture_keywords", "")
        
        # 1. 장점 관련 문서 검색 (긍정적 내용)
        positive_base_query = f"{company_name} 사내문화 조직문화 회사분위기 좋음 분위기 좋은 직원들 사이 좋음 동료관계 화목 팀워크 좋음 소통 원활 수평적 문화 개방적 분위기 자유로운 의견개진 창의적 업무환경 혁신적 문화 도전정신 격려 실수 용인 학습문화 성장지향 다양성 존중 포용적 문화 공정한 평가 투명한 경영 신뢰문화 존중문화 협력적 분위기 즐거운 업무환경 재미있는 직장 활기찬 분위기 긍정적 에너지 친근한 상사 배려하는 동료 서로 도움 상호존중 편안한 분위기 스트레스 적은 인간적인 회사"
        
        if culture_keywords.strip():
            positive_documents = await self.retrieve_knowledge_with_keywords(
                base_query=positive_base_query,
                user_keywords=f"{company_name} {culture_keywords} 좋음 긍정적 장점",
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
        negative_base_query = f"{company_name} 사내문화 조직문화 회사분위기 나쁨 분위기 안좋음 직원들 사이 나쁨 동료관계 갈등 팀워크 안됨 소통 부족 수직적 문화 폐쇄적 분위기 경직된 문화 보수적 분위기 권위적 문화 갑질 문화 괴롭힘 따돌림 차별 편파적 평가 불투명한 경영 불신문화 경쟁적 분위기 스트레스 많은 업무환경 삭막한 직장 무기력한 분위기 부정적 에너지 까다로운 상사 이기적인 동료 도움 안줌 상호불신 경직된 분위기 스트레스 많은 인간관계 복잡 정치적 회사문화 파벌 인맥 중시"
        
        if culture_keywords.strip():
            negative_documents = await self.retrieve_knowledge_with_keywords(
                base_query=negative_base_query,
                user_keywords=f"{company_name} {culture_keywords} 나쁨 부정적 단점",
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
                "analysis_type": "사내 문화",
                "error": "관련 정보를 찾을 수 없습니다.",
                "confidence_score": 0.0
            }
        
        # 4. 장점 분석 (긍정적 문서가 있는 경우)
        strengths = []
        if positive_documents:
            strengths_prompt = f"""
다음은 {company_name}의 사내 문화 장점과 관련된 실제 긍정적 리뷰 데이터입니다.  
이 데이터를 바탕으로 {company_name}의 사내 문화 **장점**을 5~7개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.

**출력 형식:**
JSON 객체 형태로만 출력하세요.

{{
  "strengths": [
    "구체적인 장점 1",
    "구체적인 장점 2",
    ...
    "구체적인 장점 N"
  ],
  "final_summary": "위 장점들을 종합했을 때 {company_name}의 사내 문화 전반적인 강점 요약"
}}

**중요:**  
- 반드시 실제 리뷰 데이터에 근거한 구체적인 단점만 작성하세요.  
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
                print(f"장점 분석 실패 사내문화: {str(e)}")
                strengths = {"strengths": ["장점 분석 중 오류 발생"], "final_summary": "분석에 실패했습니다."}
        
        # 5. 단점 분석 (부정적 문서가 있는 경우) 
        weaknesses = []
        if negative_documents:
            weaknesses_prompt = f"""
다음은 {company_name}의 사내 문화 단점과 관련된 실제 부정적 리뷰 데이터입니다.  
이 데이터를 바탕으로 {company_name}의 사내 문화 **단점**을 5~7개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.

**출력 형식:**
JSON 객체 형태로만 출력하세요.

{{
  "strengths": [
    "구체적인 단점 1",
    "구체적인 단점 2",
    ...
    "구체적인 단점 N"
  ],
  "final_summary": "위 단점들을 종합했을 때 {company_name}의 사내 문화 전반적인 단점 요약"
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
            # 로깅 설정
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)

            # 최종 결과 구성 (종합 분석 제외)
            result = {
                "company_name": company_name,
                "analysis_type": "사내 문화",
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
                "analysis_type": "사내 문화", 
                "error": f"분석 중 오류 발생: {str(e)}",
                "strengths": strengths,
                "weaknesses": weaknesses,
                "confidence_score": 0.5
            }