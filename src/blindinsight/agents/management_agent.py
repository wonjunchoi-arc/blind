from typing import Any, Dict, List, Optional
from ..models.user import UserProfile
from .base import BaseAgent, AgentResult, AgentConfig

class ManagementAgent(BaseAgent):
    """
    경영진/관리 분석 전문 에이전트 - 장점/단점 RAG 검색 및 데이터 기반 인사이트 생성
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(name="management_agent", config=config)
        self.target_collections = ["management", "general"]

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
        positive_base_query = f"{company_name} 경영진 대표 CEO 임원 이사 부사장 상무 전무 부장 팀장 관리자 리더십 좋음 우수함 뛰어남 소통 잘함 원활 개방적 의사결정 빠름 합리적 공정 투명 비전 명확 방향성 확실 전략적 사고 장기적 관점 직원 배려 인재 중시 성장 지원 혁신적 사고 변화 추진력 문제해결 능력 판단력 좋음 결단력 있음 책임감 강함 신뢰할 만함 존경스러움 역량 있음 경험 풍부 전문성 높음 카리스마 있음 동기부여 잘함"
        
        if management_keywords.strip():
            positive_documents = await self.retrieve_knowledge_with_keywords(
                base_query=positive_base_query,
                user_keywords=f"{company_name} {management_keywords} 좋음 긍정적 장점",
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

        # 2. 단점 관련 문서 검색
        negative_base_query = f"{company_name} 경영진 대표 CEO 임원 이사 부사장 상무 전무 부장 팀장 관리자 리더십 부족 나쁨 미흡 소통 안됨 부족 폐쇄적 의사결정 느림 불합리 불공정 불투명 비전 불명확 방향성 없음 근시안적 사고 단기적 관점 직원 무시 인재 경시 성장 방해 보수적 사고 변화 거부 문제해결 능력 부족 판단력 부족 우유부단 책임회피 신뢰 못함 실망스러움 역량 부족 경험 부족 전문성 낮음 카리스마 없음 동기부여 못함 갑질 권위적 독선적 편파적 정치적 파벌 만들기"
        
        if management_keywords.strip():
            negative_documents = await self.retrieve_knowledge_with_keywords(
                base_query=negative_base_query,
                user_keywords=f"{company_name} {management_keywords} 나쁨 부정적 단점",
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
                "analysis_type": "경영진",
                "error": "관련 정보를 찾을 수 없습니다.",
                "confidence_score": 0.0
            }

        strengths = []
        if positive_documents:
            strengths_prompt = f"""
다음은 {company_name}의 경영진/관리진 장점과 관련된 실제 긍정적 리뷰 데이터입니다.
이 데이터를 바탕으로 {company_name}의 경영진/관리진 **장점**을 5~7개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.

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
"final_summary": "위 장점들을 종합했을 때 {company_name}의 경영진/관리진 전반적인 강점 요약"
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
다음은 {company_name}의 경영진/관리진 단점과 관련된 실제 부정적 리뷰 데이터입니다.
이 데이터를 바탕으로 {company_name}의 경영진/관리진 **단점**을 5~7개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.

**출력 형식:**
JSON 객체 형태로만 출력하세요.

{{
  "weaknesses": [
    "구체적인 단점 1",
    "구체적인 단점 2",
    "구체적인 단점 3",
    ...
    "구체적인 단점 N"
  ],
  "final_summary": "위 단점들을 종합했을 때 {company_name}의 경영진/관리진 전반적인 단점 요약"
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