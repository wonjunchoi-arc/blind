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
import os


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
        self.target_collections = ["salary_benefits"]
    
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
        positive_base_query = f"연봉, 보너스, 복리후생, 복지혜택, 지원제도"
        
        if salary_keywords.strip():
            positive_documents = await self.retrieve_knowledge(
                query=f" {salary_keywords}",
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
        negative_base_query = f"급여낮음, 보너스없음, 복지부족, 지원없음, 혜택미흡"
        
        if salary_keywords.strip():
            negative_documents = await self.retrieve_knowledge(
                query=f" {salary_keywords}",
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
                "analysis_type": "급여 및 복지",
                "error": "관련 정보를 찾을 수 없습니다.",
                "confidence_score": 0.0
            }
        
        # 4. 장점 분석 (긍정적 문서가 있는 경우)
        strengths = []
        if positive_documents:
            strengths_prompt = f"""
            다음은 해당 기업의 연봉,급여,상여급등의 금전적인 것과 직원복지과 관련된 장점과 관련된 실제 긍정적 리뷰 데이터입니다.  
            이 데이터를 바탕으로 {company_name}의 해당 기업의 연봉,급여,상여급등의 금전적인 것과 직원복지의 **장점**을 5개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.  

            **중요 조건:**  
            - 입력된 context_text에는 다양한 문서가 포함될 수 있습니다.  
            - 반드시 **해당 기업의 연봉,급여,상여급등의 금전적인 것과 직원복지과 관련 내용이 담긴 문서들만** 선별하여 요약 및 정리하세요.  
            - 해당 기업의 연봉,급여,상여급등의 금전적인 것과 직원복지과 직접 관련 없는 문서는 무시하고, 나열하거나 요약하지 마세요.  
            - 반복적으로 언급되거나 리뷰에서 많이 나타나는 장점 위주로만 추려주세요.  

            목적은 사용자들이 회사에 대한 정확한 정보를 얻고, 해당 기업의 연봉,급여,상여급등의 금전적인 것과 직원복지과 관련 정보를 얻고 취업, 이직등에 개해 결정을 내리는 데 도움을 주는 것입니다.

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
            "final_summary": "위 장점들을 종합했을 때 {company_name}의 해당 기업의 연봉,급여,상여급등의 금전적인 것과 직원복지와 관련된 전반적인 장점 요약"
            }}
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
                import re

                if strengths_response and strengths_response.strip():
                    try:
                        # 마크다운 코드 블록이 있는지 확인
                        if '```' in strengths_response:
                            # 마크다운 코드 블록 제거
                            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', strengths_response.strip(), flags=re.MULTILINE)
                            cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
                            strengths = json.loads(cleaned.strip())
                        else:
                            # 일반 JSON 파싱
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
            다음은 해당 기업의 연봉,급여,상여급등의 금전적인 것과 직원복지과 관련된 단점과 관련된 실제 긍정적 리뷰 데이터입니다.  
            이 데이터를 바탕으로 {company_name}의 해당 기업의 연봉,급여,상여급등의 금전적인 것과 직원복지의 **단점**을 5개 정도로 구체적이고 핵심적으로 정리 및 요약해주세요.  

            **중요 조건:**  
            - 입력된 context_text에는 다양한 문서가 포함될 수 있습니다.  
            - 반드시 **해당 기업의 연봉,급여,상여급등의 금전적인 것과 직원복지과 관련 내용이 담긴 문서들만** 선별하여 요약 및 정리하세요.  
            - 해당 기업의 연봉,급여,상여급등의 금전적인 것과 직원복지과 직접 관련 없는 문서는 무시하고, 나열하거나 요약하지 마세요.  
            - 반복적으로 언급되거나 리뷰에서 많이 나타나는 단점 위주로만 추려주세요.  

            목적은 사용자들이 회사에 대한 정확한 정보를 얻고, 해당 기업의 연봉,급여,상여급등의 금전적인 것과 직원복지과 관련 정보를 얻고 취업, 이직등에 개해 결정을 내리는 데 도움을 주는 것입니다.

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
            "final_summary": "위 단점들을 종합했을 때 {company_name}의 해당 기업의 연봉,급여,상여급등의 금전적인 것과 직원복지와 관련된 전반적인 단점 요약"
            }}
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
                import re

                if weaknesses_response and weaknesses_response.strip():
                    try:
                        # 마크다운 코드 블록이 있는지 확인
                        if '```' in weaknesses_response:
                            # 마크다운 코드 블록 제거
                            cleaned = re.sub(r'^```(?:json)?\s*\n?', '', weaknesses_response.strip(), flags=re.MULTILINE)
                            cleaned = re.sub(r'\n?```\s*$', '', cleaned, flags=re.MULTILINE)
                            weaknesses = json.loads(cleaned.strip())
                        else:
                            # 일반 JSON 파싱
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