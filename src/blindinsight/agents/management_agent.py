from typing import Any, Dict, List, Optional
from ..models.user import UserProfile
from .base import BaseAgent, AgentResult, AgentConfig
import os

class ManagementAgent(BaseAgent):
    """
    경영진/관리 분석 전문 에이전트 - 장점/단점 RAG 검색 및 데이터 기반 인사이트 생성
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(name="management_agent", config=config)
        self.target_collections = ["management"]

    async def execute(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> AgentResult:
        return await self._execute_with_error_handling(
            self._analyze_management,
            query,
            context,
            user_profile
        )

    async def _analyze_management(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> Dict[str, Any]:

        company_name = context.get("company_name") if context else query
        
        # 사용자 키워드 추출 (경영진 관련)
        management_keywords = ""
        if context:
            management_keywords = context.get("management_keywords", "")

        # 1. 장점 관련 문서 검색
        positive_base_query = f"경영진, 리더십, 소통, 비전, 혁신"
        
        if management_keywords.strip():
            positive_documents = await self.retrieve_knowledge(
                query=f" {management_keywords}",
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

        # 2. 단점 관련 문서 검색
        negative_base_query = f"경영진, 소통부족, 불투명, 보수적, 책임회피"
        
        if management_keywords.strip():
            negative_documents = await self.retrieve_knowledge(
                query=f" {management_keywords}",
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
                "analysis_type": "경영진",
                "error": "관련 정보를 찾을 수 없습니다.",
                "confidence_score": 0.0
            }

        strengths = []
        if positive_documents:
            strengths_prompt = f"""
            다음은 해당 기업의 경영진, 임원과 관련된 장점과 관련된 실제 긍정적 리뷰 데이터입니다.  
            이 데이터를 바탕으로 {company_name}의 해당 기업의 경영진, 임원의 **장점**을 5개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.  

            **중요 조건:**  
            - 입력된 context_text에는 다양한 문서가 포함될 수 있습니다.  
            - 반드시 **해당 기업의 경영진, 임원과 관련 내용이 담긴 문서들만** 선별하여 요약 및 정리하세요.  
            - 해당 기업의 경영진, 임원과 직접 관련 없는 문서는 무시하고, 나열하거나 요약하지 마세요.  
            - 반복적으로 언급되거나 리뷰에서 많이 나타나는 장점 위주로만 추려주세요.  

            목적은 사용자들이 회사에 대한 정확한 정보를 얻고, 해당 기업의 경영진, 임원과 관련 정보를 얻고 취업, 이직등에 개해 결정을 내리는 데 도움을 주는 것입니다.

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
            "final_summary": "위 장점들을 종합했을 때 {company_name}의 해당 기업의 경영진, 임원와 관련된 전반적인 강점 요약"
            }}
            """
            try:
                strengths_response = await self.generate_response(
                    prompt=strengths_prompt,
                    context_documents=positive_documents,
                    temperature=1.0,
                    max_tokens=10000
                )
                # print(f"[management_agent] 장점 분석 응답: {str(strengths_response)[:200]}...")

                import json
                if strengths_response and strengths_response.strip():
                    try:
                        strengths = json.loads(strengths_response.strip())
                    except json.JSONDecodeError as je:
                        print(f"JSON 파싱 실패, 대체 파싱 시도: {str(je)}")
                        if isinstance(strengths_response, str):
                            strengths = {"strengths": [strengths_response], "final_summary": "분석 결과를 정리하지 못했습니다."}
                        else:
                            strengths = {"strengths": ["장점 분석 중 오류 발생"], "final_summary": "분석에 실패했습니다."}
                else:
                    print("빈 응답 받음")
                    strengths = {"strengths": ["응답을 받지 못했습니다."], "final_summary": "분석에 실패했습니다."}
            except Exception as e:
                print(f"장점 분석 실패 경영진: {str(e)}")
                strengths = {"strengths": ["장점 분석 중 오류 발생"], "final_summary": "분석에 실패했습니다."}

        # 5. 단점 분석
        weaknesses = []
        if negative_documents:
            weaknesses_prompt = f"""
            다음은 해당 기업의 경영진, 임원과 관련된 단점과 관련된 실제 긍정적 리뷰 데이터입니다.  
            이 데이터를 바탕으로 {company_name}의 해당 기업의 경영진, 임원의 **단점**을 5개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.  

            **중요 조건:**  
            - 입력된 context_text에는 다양한 문서가 포함될 수 있습니다.  
            - 반드시 **해당 기업의 경영진, 임원과 관련 내용이 담긴 문서들만** 선별하여 요약 및 정리하세요.  
            - 해당 기업의 경영진, 임원과 직접 관련 없는 문서는 무시하고, 나열하거나 요약하지 마세요.  
            - 반복적으로 언급되거나 리뷰에서 많이 나타나는 단점 위주로만 추려주세요.  

            목적은 사용자들이 회사에 대한 정확한 정보를 얻고, 해당 기업의 경영진, 임원과 관련 정보를 얻고 취업, 이직등에 개해 결정을 내리는 데 도움을 주는 것입니다.

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
            "final_summary": "위 단점들을 종합했을 때 {company_name}의 해당 기업의 경영진, 임원와 관련된 전반적인 단점 요약"
            }}
            """
            try:
                weaknesses_response = await self.generate_response(
                    prompt=weaknesses_prompt,
                    context_documents=negative_documents,
                    temperature=1.0,
                    max_tokens=10000
                )
                # print(f"[management_agent] 단점 분석 응답: {str(weaknesses_response)[:200]}...")

                import json
                if weaknesses_response and weaknesses_response.strip():
                    try:
                        weaknesses = json.loads(weaknesses_response.strip())
                    except json.JSONDecodeError as je:
                        print(f"JSON 파싱 실패, 대체 파싱 시도: {str(je)}")
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

        # 6. 최종 결과 구성
        try:
            result = {
                "company_name": company_name,
                "analysis_type": "경영진",
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
                "analysis_type": "경영진",
                "error": f"분석 중 오류 발생: {str(e)}",
                "strengths": strengths,
                "weaknesses": weaknesses,
                "confidence_score": 0.5
            }