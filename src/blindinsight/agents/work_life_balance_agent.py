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
import os


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
        self.target_collections = ["work_life_balance"]
    
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
        
        # 사용자 키워드 추출 (워라밸 관련)
        worklife_keywords = ""
        if context:
            worklife_keywords = context.get("worklife_keywords", "")
        
        # 1. 장점 관련 문서 검색 (긍정적 내용)
        positive_base_query = f"워라밸, 정시퇴근, 유연근무, 휴가활용, 개인시간보장"
        
        if worklife_keywords.strip():
            positive_documents = await self.retrieve_knowledge(
                query=f" {worklife_keywords}",
                collections=self.target_collections,
                company_name=company_name,
                content_type_filter="pros",
                k=int(os.getenv("DEFAULT_SEARCH_K", "10"))
            )
        else:
            positive_documents = await self.retrieve_knowledge(
                query=positive_base_query,
                collections=self.target_collections,
                company_name=company_name,
                content_type_filter="pros",
                k=int(os.getenv("DEFAULT_SEARCH_K", "10"))
            )
        
        # 2. 단점 관련 문서 검색 (부정적 내용)
        negative_base_query = f"야근, 주말근무, 휴가어려움, 개인시간부족, 번아웃"
        
        if worklife_keywords.strip():
            negative_documents = await self.retrieve_knowledge(
                query=f" {worklife_keywords}",
                collections=self.target_collections,
                company_name=company_name,
                content_type_filter="cons",
                k=int(os.getenv("DEFAULT_SEARCH_K", "10"))
            )
        else:
            negative_documents = await self.retrieve_knowledge(
                query=negative_base_query,
                collections=self.target_collections,
                company_name=company_name,
                content_type_filter="cons",
                k=int(os.getenv("DEFAULT_SEARCH_K", "10"))
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
            다음은 워라벨, 즉 일과 삶의 균형이 얼마나 잘 이루어지고 있는지, 업무과 너무 과다한것은 아닌지 등과 관련된 장점과 관련된 실제 긍정적 리뷰 데이터입니다.  
            이 데이터를 바탕으로 {company_name}의 워라벨, 즉 일과 삶의 균형이 얼마나 잘 이루어지고 있는지, 업무과 너무 과다한것은 아닌지 등의 **장점**을 5개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.  

            **중요 조건:**  
            - 입력된 context_text에는 다양한 문서가 포함될 수 있습니다.  
            - 반드시 **워라벨, 즉 일과 삶의 균형이 얼마나 잘 이루어지고 있는지, 업무과 너무 과다한것은 아닌지 등과 관련 내용이 담긴 문서들만** 선별하여 요약 및 정리하세요.  
            - 워라벨, 즉 일과 삶의 균형이 얼마나 잘 이루어지고 있는지, 업무과 너무 과다한것은 아닌지 등과 직접 관련 없는 문서는 무시하고, 나열하거나 요약하지 마세요.  
            - 반복적으로 언급되거나 리뷰에서 많이 나타나는 장점 위주로만 추려주세요.  

            목적은 사용자들이 회사에 대한 정확한 정보를 얻고, 워라벨, 즉 일과 삶의 균형이 얼마나 잘 이루어지고 있는지, 업무과 너무 과다한것은 아닌지 등과 관련 정보를 얻고 취업, 이직등에 개해 결정을 내리는 데 도움을 주는 것입니다.

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
            "final_summary": "위 장점들을 종합했을 때 {company_name}의 워라벨, 즉 일과 삶의 균형이 얼마나 잘 이루어지고 있는지, 업무과 너무 과다한것은 아닌지 등와 관련된 전반적인 장점 요약"
            }}
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
            다음은 워라벨, 즉 일과 삶의 균형이 얼마나 잘 이루어지고 있는지, 업무과 너무 과다한것은 아닌지 등과 관련된 단점과 관련된 실제 긍정적 리뷰 데이터입니다.  
            이 데이터를 바탕으로 {company_name}의 워라벨, 즉 일과 삶의 균형이 얼마나 잘 이루어지고 있는지, 업무과 너무 과다한것은 아닌지 등의 **단점**을 5개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.  

            **중요 조건:**  
            - 입력된 context_text에는 다양한 문서가 포함될 수 있습니다.  
            - 반드시 **워라벨, 즉 일과 삶의 균형이 얼마나 잘 이루어지고 있는지, 업무과 너무 과다한것은 아닌지 등과 관련 내용이 담긴 문서들만** 선별하여 요약 및 정리하세요.  
            - 워라벨, 즉 일과 삶의 균형이 얼마나 잘 이루어지고 있는지, 업무과 너무 과다한것은 아닌지 등과 직접 관련 없는 문서는 무시하고, 나열하거나 요약하지 마세요.  
            - 반복적으로 언급되거나 리뷰에서 많이 나타나는 단점 위주로만 추려주세요.  

            목적은 사용자들이 회사에 대한 정확한 정보를 얻고, 워라벨, 즉 일과 삶의 균형이 얼마나 잘 이루어지고 있는지, 업무과 너무 과다한것은 아닌지 등과 관련 정보를 얻고 취업, 이직등에 개해 결정을 내리는 데 도움을 주는 것입니다.

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
            "final_summary": "위 단점들을 종합했을 때 {company_name}의 워라벨, 즉 일과 삶의 균형이 얼마나 잘 이루어지고 있는지, 업무과 너무 과다한것은 아닌지 등와 관련된 전반적인 단점 요약"
            }}
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