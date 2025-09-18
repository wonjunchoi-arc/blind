"""
BlindInsight AI의 기본 에이전트 클래스 및 유틸리티

🏗️ 핵심 구조:
- BaseAgent: 모든 에이전트의 추상 기본 클래스
- RAG 시스템: ChromaDB에서 관련 문서 검색
- MCP 통합: 외부 데이터 제공자와 통신
- LLM 생성: OpenAI GPT 모델을 통한 응답 생성
- 성능 추적: 실행 시간 및 성공률 모니터링
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

from langchain.schema import Document
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from ..models.base import BaseModel, settings
from ..models.user import UserProfile
from ..rag.knowledge_base import KnowledgeBase
from ..rag.retriever import RAGRetriever


class AgentResult(BaseModel):
    """
    에이전트 실행 결과를 담는 컨테이너 클래스
    
    🎯 주요 정보:
    - 실행 성공/실패 여부
    - 분석 결과 데이터
    - 실행 시간 및 신뢰도 점수
    - 메타데이터 및 오류 정보
    
    📊 사용 용도:
    - 에이전트별 실행 결과 표준화
    - 성능 모니터링 및 디버깅
    - 결과 집계 및 분석
    """
    
    agent_name: str  # 에이전트 이름 (예: "culture_analysis", "compensation_analysis")
    success: bool  # 실행 성공 여부 (True: 성공, False: 실패)
    result: Optional[Dict[str, Any]] = None  # 분석 결과 데이터 (JSON 형태)
    error_message: Optional[str] = None  # 에러 발생 시 오류 메시지
    execution_time: float = 0.0  # 실행 소요 시간 (초 단위)
    confidence_score: float = 0.0  # 결과 신뢰도 점수 (0.0~1.0)
    metadata: Dict[str, Any] = {}  # 추가 메타데이터 (컨텍스트, 설정값 등)
    
    class Config:
        """Pydantic 설정 클래스 - 스키마 예제 정의"""
        schema_extra = {
            "example": {
                "agent_name": "culture_analysis",  # 회사 문화 분석 에이전트
                "success": True,  # 성공적 실행
                "result": {"culture_score": 85},  # 문화 점수 85점
                "execution_time": 12.5,  # 12.5초 소요
                "confidence_score": 0.88  # 신뢰도 88%
            }
        }


class AgentConfig(BaseModel):
    """
    에이전트 동작 설정을 관리하는 구성 클래스
    
    🔧 설정 분야:
    - LLM 설정: 모델명, 온도, 토큰 수
    - RAG 설정: 검색 개수, 유사도 임계값
    - 성능 설정: 캐싱, 재시도, TTL
    - 품질 설정: 검증, 신뢰도 임계값
    
    💡 사용법:
    - 기본값으로 초기화하거나 필요한 값만 오버라이드
    - 에이전트별로 다른 설정을 적용 가능
    - 운영 환경에 따라 성능 최적화 가능
    """ 
    
    # LLM 설정 - OpenAI 모델 관련
    model_name: str = "gpt-4o-mini-2024-07-18"  # 사용할 GPT 모델명
    temperature: float = 0.4  # 응답 창의성 (0.0: 일관성, 1.0: 창의성)
    max_tokens: int = 4000  # 최대 생성 토큰 수
    timeout: int = 60  # API 호출 타임아웃 (초)
    
    # RAG 검색 설정 - ChromaDB 벡터 검색
    max_retrievals: int = 50  # 검색할 최대 문서 수
    relevance_threshold: float = 0.3  # 관련성 점수 임계값 (0.0~1.0)  
    enable_reranking: bool = True  # 재순위 매김 활성화
    keyword_match_threshold: float = 0.005  # 키워드 매칭 임계값 (0.0~1.0)
    
    # 성능 최적화 설정
    enable_caching: bool = True  # 결과 캐싱 활성화
    cache_ttl: int = 3600  # 캐시 유지 시간 (초, 1시간)
    max_retries: int = 3  # 실패 시 최대 재시도 횟수
    
    # 품질 관리 설정
    min_confidence_threshold: float = 0.6  # 최소 신뢰도 임계값
    require_sources: bool = True  # 출처 정보 필수 여부
    validate_outputs: bool = False  # 출력 결과 검증 활성화
    
    class Config:
        """Pydantic 설정 클래스 - 구성 예제 정의"""
        schema_extra = {
            "example": {
                "model_name": "gpt-5-mini-2025-08-07",  # GPT-5-mini 터보 모델 사용
                "temperature": 0.3,  # 낮은 창의성으로 일관된 분석
                "max_retrievals": 50,  # 최대 20개 관련 문서 검색
                "enable_caching": True  # 캐싱으로 성능 최적화
            }
        }


class BaseAgent(ABC):
    """
    모든 BlindInsight AI 에이전트의 추상 기본 클래스
    
    🎯 제공 기능:
    - RAG 기반 지식 검색: ChromaDB에서 관련 문서 찾기
    - LLM 상호작용: OpenAI GPT 모델을 통한 분석
    - MCP 서비스 호출: 외부 데이터 소스 연동
    - 결과 검증: 품질 관리 및 성능 추적
    
    🏗️ 상속받는 전문 에이전트들:
    - CultureAnalysisAgent: 회사 문화 분석
    - CompensationAnalysisAgent: 연봉 및 복리후생 분석
    - GrowthStabilityAgent: 성장성 및 안정성 분석
    - CareerPathAgent: 커리어 경로 분석
    """
    
    def __init__(
        self,
        name: str,
        config: Optional[AgentConfig] = None,
        tools: Optional[List[Tool]] = None
    ):
        """
        BaseAgent 초기화 메서드

        Args:
            name: 에이전트 이름 (예: "culture_analysis")
            config: 에이전트 설정 (기본값 사용 시 None)
            tools: 추가 도구 리스트 (선택사항)
        """
        self.name = name  # 에이전트 식별자
        self.config = config or AgentConfig()  # 설정 (기본값 또는 커스텀)
        self.tools = tools or []  # 도구 목록 초기화

        # 에이전트별 모델명 결정
        model_name = self._get_agent_model_name()

        # LLM 모델 초기화 - OpenAI ChatGPT
        self.llm = ChatOpenAI(
            model=model_name,  # 에이전트별 모델명 설정
            temperature=self.config.temperature,  # 창의성 수준
            max_tokens=self.config.max_tokens,  # 최대 토큰 수
            timeout=self.config.timeout,  # 타임아웃 설정
            api_key=settings.openai_api_key  # API 키 (환경변수에서 로드)
        )
        
        # 대화 기록 관리를 위한 메시지 리스트 (메모리 마이그레이션)
        self.chat_history: List[Union[AIMessage, HumanMessage]] = []
        
        # RAG 검색 엔진 초기화
        self.knowledge_base = KnowledgeBase()
        self.rag_retriever = RAGRetriever(
            vector_store=self.knowledge_base.vector_store,
            keyword_threshold=self.config.keyword_match_threshold
        )
        
        # MCP 클라이언트 초기화 (하위 클래스에서 설정)
        self.mcp_clients = {}
        
        # 성능 추적 변수 초기화
        self._execution_count = 0  # 총 실행 횟수
        self._total_execution_time = 0.0  # 총 실행 시간
        self._success_count = 0  # 성공 실행 횟수

    def _get_agent_model_name(self) -> str:
        """
        에이전트별 모델명 결정

        우선순위:
        1. config에서 명시적으로 설정된 모델명
        2. 에이전트별 환경변수 설정
        3. 기본 에이전트 모델
        4. 하드코딩된 기본값

        Returns:
            사용할 모델명
        """
        # 1. config에서 명시적으로 설정된 경우
        if hasattr(self.config, 'model_name') and self.config.model_name != "gpt-4o-mini-2024-07-18":
            return self.config.model_name

        # 2. 에이전트별 환경변수 설정 확인
        agent_model_map = {
            "salary_benefits_agent": settings.salary_benefits_agent_model,
            "company_culture_agent": settings.company_culture_agent_model,
            "work_life_balance_agent": settings.work_life_balance_agent_model,
            "management_agent": settings.management_agent_model,
            "career_growth_agent": settings.career_growth_agent_model,
            "quality_evaluator_agent": settings.quality_evaluator_agent_model,
        }

        if self.name in agent_model_map:
            return agent_model_map[self.name]

        # 3. 기본 에이전트 모델
        return settings.default_agent_model
    
    def add_message(self, message: Union[HumanMessage, AIMessage]):
        """메시지를 대화 기록에 추가"""
        self.chat_history.append(message)
    
    @abstractmethod
    async def execute(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> AgentResult:
        """
        에이전트의 핵심 기능을 실행하는 추상 메서드 (하위 클래스에서 구현 필수)
        
        🎯 주요 역할:
        - 사용자 질의에 대한 분석 수행
        - RAG와 MCP를 활용한 데이터 검색 및 처리
        - LLM을 통한 인사이트 생성 및 응답 구성
        
        Args:
            query: 분석할 주요 질의 (예: "네이버 회사 문화는 어때?")
            context: 추가 컨텍스트 정보 (회사명, 직무 등)
            user_profile: 개인화 분석을 위한 사용자 프로필
            
        Returns:
            AgentResult: 실행 결과를 포함한 표준화된 결과 객체
        """
        pass
    
    async def execute_as_supervisor_tool(
        self,
        user_question: str,
        supervisor_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        새로운 Supervisor Tool 워크플로우:
        1. 사용자 질문에서 키워드 추출 및 RAG 검색
        2. 문서 존재 여부 확인
        3. Supervisor 프롬프트 + RAG 결과로 LLM 응답 생성
        4. AI 품질 평가 (향후 구현)
        
        Args:
            user_question: 사용자의 원본 질문
            supervisor_prompt: Supervisor가 생성한 맞춤형 프롬프트
            context: 추가 컨텍스트 정보
            
        Returns:
            str: 사용자 친화적인 답변 텍스트
        """
        try:
            print(f"[{self.name}] ===== execute_as_supervisor_tool 시작 =====")
            print(f"[{self.name}] 사용자 질문: {user_question}")
            print(f"[{self.name}] Supervisor 프롬프트 존재: {bool(supervisor_prompt)}")
            if supervisor_prompt:
                print(f"[{self.name}] Supervisor 프롬프트: {supervisor_prompt[:200]}...")
            print(f"[{self.name}] 컨텍스트: {context}")
            
            # 1. 프론트엔드에서 전달된 필터 사용
            frontend_company = context.get("company_filter") if context else None
            frontend_year = context.get("year_filter") if context else None
            frontend_content_type = context.get("content_type") if context else None

            print(f"[{self.name}] 프론트엔드 필터 - 회사: {frontend_company}, 연도: {frontend_year}, 콘텐츠타입: {frontend_content_type}")

            # 2. RAG 검색 수행 (단순화)
            print(f"[{self.name}] RAG 검색 시작...")
            rag_documents = await self._perform_rag_search(
                user_question=user_question,
                company_filter=frontend_company,
                year_filter=frontend_year,
                content_type_filter=frontend_content_type,
                context=context
            )
            
            # 3. 문서 존재 여부 확인
            if not rag_documents:
                print(f"[{self.name}] RAG 검색 결과 없음")
                company_text = frontend_company or "해당 회사"
                return f"죄송합니다. '{company_text}'에 대한 {self.name} 관련 정보를 찾을 수 없습니다."
            
            print(f"[{self.name}] RAG 검색 완료: {len(rag_documents)}개 문서 발견")
            
            # 문서 미리보기 출력
            for i, doc in enumerate(rag_documents[:3]):
                content_preview = doc.page_content[:150].replace('\n', ' ')
                print(f"[{self.name}] 문서 {i+1}: {content_preview}...")
            
            # 4. Supervisor 프롬프트가 있으면 사용, 없으면 기본 프롬프트
            if supervisor_prompt and supervisor_prompt.strip():
                final_prompt = supervisor_prompt
                print(f"[{self.name}] Supervisor 커스텀 프롬프트 사용")
            else:
                # 기본 프롬프트 생성
                final_prompt = self._generate_default_prompt(user_question, frontend_company)
                print(f"[{self.name}] 기본 프롬프트 사용")
            
            print(f"[{self.name}] 최종 프롬프트: {final_prompt[:300]}...")
            
            # 5. LLM 응답 생성 (RAG 문서 + 프롬프트)
            print(f"[{self.name}] LLM 응답 생성 시작...")
            response = await self.generate_response(
                prompt=final_prompt,
                context_documents=rag_documents,
                temperature=0.7,
                max_tokens=4000
            )
            
            print(f"[{self.name}] LLM 응답 생성 완료: {len(response) if response else 0}자")
            if response:
                print(f"[{self.name}] 응답 미리보기: {response[:200]}...")
            
            # 6. 응답 후처리
            if response and response.strip():
                final_response = response.strip()
                print(f"[{self.name}] ===== execute_as_supervisor_tool 완료 =====")
                return final_response
            else:
                print(f"[{self.name}] 빈 응답 받음")
                return f"죄송합니다. {self.name}에서 적절한 답변을 생성할 수 없습니다."
                
        except Exception as e:
            print(f"[{self.name}] execute_as_supervisor_tool 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"분석 중 오류가 발생했습니다: {str(e)}"
    
    async def _perform_rag_search(
        self,
        user_question: str,
        company_filter: Optional[str] = None,
        year_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """RAG 검색 수행 - 프론트엔드 필터 직접 사용"""
        try:
            # 검색 쿼리는 사용자 질문 그대로 사용
            search_query = user_question
            
            # 에이전트별 전문 컬렉션 사용
            collections = getattr(self, 'target_collections', ['general'])
            
            print(f"[{self.name}] 검색 쿼리: '{search_query}'")
            print(f"[{self.name}] 대상 컬렉션: {collections}")
            print(f"[{self.name}] 회사 필터: {company_filter}")
            print(f"[{self.name}] 연도 필터: {year_filter}")
            print(f"[{self.name}] 콘텐츠타입 필터: {content_type_filter}")

            # RAG 검색 수행 (프론트엔드 필터 직접 사용)
            documents = await self.retrieve_knowledge(
                query=search_query,
                collections=collections,
                company_name=company_filter,
                year_filter=year_filter,
                content_type_filter=content_type_filter,
                k=15  # 좀 더 많은 문서 검색
            )
            
            print(f"[{self.name}] RAG 검색 결과: {len(documents)}개 문서")
            return documents
            
        except Exception as e:
            print(f"[{self.name}] RAG 검색 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _generate_default_prompt(self, user_question: str, company_name: Optional[str]) -> str:
        """기본 프롬프트 생성 (Supervisor 프롬프트가 없을 때)"""
        company_text = company_name or "해당 회사"
        
        return f"""
다음은 {company_text}에 대한 실제 리뷰 데이터입니다.
이 데이터를 바탕으로 사용자의 질문 "{user_question}"에 대해 구체적이고 도움이 되는 답변을 제공해주세요.

답변 시 다음 사항을 고려해주세요:
1. 실제 리뷰 데이터에 근거한 구체적인 내용으로 답변
2. 장점과 단점을 균형있게 제시
3. 사용자가 의사결정에 도움이 될 수 있도록 명확하게 설명
4. 한국어로 자연스럽고 이해하기 쉽게 작성

리뷰 데이터를 종합하여 답변해주세요.
"""
    
    def _format_supervisor_response(self, result: Dict[str, Any], original_question: str) -> str:
        """에이전트 결과를 사용자 친화적 텍스트로 변환"""
        try:
            # 결과에서 주요 정보 추출
            if "summary" in result:
                return result["summary"]
            elif "final_summary" in result:
                return result["final_summary"]
            elif "analysis" in result:
                return result["analysis"]
            elif isinstance(result, str):
                return result
            else:
                # 구조화된 결과를 텍스트로 변환
                parts = []
                
                if "strengths" in result:
                    strengths = result["strengths"]
                    if isinstance(strengths, dict):
                        pros = strengths.get("pros", [])
                        if pros:
                            parts.append("**장점:**")
                            for pro in pros[:3]:  # 상위 3개만
                                parts.append(f"• {pro}")
                
                if "weaknesses" in result:
                    weaknesses = result["weaknesses"]
                    if isinstance(weaknesses, dict):
                        cons = weaknesses.get("cons", [])
                        if cons:
                            parts.append("\n**단점:**")
                            for con in cons[:3]:  # 상위 3개만
                                parts.append(f"• {con}")
                
                return "\n".join(parts) if parts else f"{self.name}의 분석이 완료되었습니다."
                
        except Exception:
            return f"{self.name}의 분석 결과를 표시할 수 없습니다."
    
    async def retrieve_knowledge(
        self,
        query: str,
        collections: List[str] = None,
        company_name: Optional[str] = None,
        content_type_filter: Optional[str] = None,  # "pros", "cons"
        position_filter: Optional[str] = None,  # 직무 필터 (예: "IT 디자이너")
        year_filter: Optional[str] = None,  # 연도 필터 (예: "2024")
        k: int = None
    ) -> List[Document]:
        """
        🔍 RAG 시스템을 사용한 관련 지식 검색 (멀티 컬렉션 지원)
        
        🏗️ RAG 검색 과정:
        1. 쿼리를 임베딩 벡터로 변환 (OpenAI text-embedding-3-small)
        2. 지정된 컬렉션들에서 병렬 검색 수행
        3. 유사도 점수가 임계값 이상인 문서만 필터링
        4. 회사별 필터링 적용 (선택적)
        5. 상위 k개 문서 반환 (중복 제거)
        
        📁 검색 가능한 컬렉션:
        - company_culture: 회사 문화 리뷰 (기업문화, 조직문화)
        - work_life_balance: 워라밸 관련 (근무시간, 야근, 휴가)
        - management: 경영진/관리 관련 (상사, 리더십, 의사결정)
        - salary_benefits: 연봉 및 복리후생 (급여, 보너스, 복지)
        - career_growth: 커리어 성장 (승진, 교육, 발전기회)
        
        Args:
            query: 검색할 질문 (예: "구글 워라밸 어때?" 또는 키워드 포함 "구글 워라밸 수평적 자율적")
            collections: 검색할 컬렉션 목록 (예: ["company_culture", "general"])
            company_name: 회사명 필터 (예: "구글")
            content_type_filter: 내용 타입 필터 ("pros", "cons")
            position_filter: 직무 필터 (예: "IT 디자이너")
            year_filter: 연도 필터 (예: "2024")
            k: 검색할 문서 개수 (기본값: config.max_retrievals=20)
            
        Returns:
            List[Document]: 관련 문서 목록 (유사도 점수 포함)
        """
        if not self.rag_retriever:
            return []
        
        k = k or self.config.max_retrievals
        collections = collections or ["general"]
        
        try:
            filters = {}
            if company_name:
                filters["company"] = company_name
            if content_type_filter:
                filters["content_type"] = content_type_filter
            if position_filter:
                filters["position"] = position_filter
            if year_filter:
                filters["review_year"] = year_filter
            
            print(f"[{self.name}] retrieve_knowledge - 필터: {filters}")
            print(f"[{self.name}] retrieve_knowledge - 요청 문서수: {k}")
            
            # 멀티 컬렉션에서 검색 수행
            all_results = []
            for collection_name in collections:
                try:
                    collection_k = k // len(collections) + 2
                    print(f"[{self.name}] '{collection_name}' 컬렉션에서 {collection_k}개 문서 검색...")
                    
                    search_results = await self.rag_retriever.search(
                        query=query,
                        collection_name=collection_name,
                        k=collection_k,
                        filters=filters,
                        search_type="ensemble"
                    )
                    print(f"[{self.name}] '{collection_name}' 검색 결과: {len(search_results)}개")
                    all_results.extend(search_results)
                except Exception as e:
                    print(f"[{self.name}] '{collection_name}' 컬렉션 검색 실패: {str(e)}")
                    continue
            
            # 중복 제거 및 필터링
            documents = []
            seen_contents = set()
            sorted_results = sorted(all_results, key=lambda x: x.relevance_score, reverse=True)
            
            for result in sorted_results:
                content_hash = hash(result.document.page_content[:100])

                if result.relevance_score < self.config.relevance_threshold:
                    print(
                        f"relevance_score={result.relevance_score:.4f} "
                        f"threshold={self.config.relevance_threshold} "
                        f"collection={getattr(result.document, 'metadata', {}).get('collection', 'unknown')} "
                        f"preview={result.document.page_content[:50]!r}"
                    )

                if (content_hash not in seen_contents and 
                    result.relevance_score >= self.config.relevance_threshold):
                    seen_contents.add(content_hash)
                    documents.append(result.document)
                    if len(documents) >= k:
                        break
            
            return documents
            
        except Exception:
            return []
    
    async def call_mcp_service(
        self, 
        service_name: str, 
        method: str, 
        params: Dict[str, Any]
    ) -> Any:
        """
        🌐 MCP(Model Context Protocol) 서비스 호출
        
        🔗 MCP 서비스 종류:
        - BlindDataProvider: Blind.com 리뷰 및 연봉 정보
        - JobSiteProvider: 채용 사이트 공고 정보
        - SalaryDataProvider: 연봉 벤치마킹 데이터
        - CompanyNewsProvider: 회사 뉴스 및 동향
        
        🚀 호출 과정:
        1. 서비스 클라이언트 확인
        2. 비동기 API 호출 실행
        3. 응답 데이터 반환
        4. 오류 발생 시 로그 출력 후 None 반환
        
        Args:
            service_name: MCP 서비스 이름 (예: 'blind_data')
            method: 호출할 메서드 (예: 'get_company_reviews')
            params: 메서드 파라미터 (예: {'company': '네이버', 'limit': 10})
            
        Returns:
            Any: 서비스 응답 데이터 (서비스별로 다른 형태)
        """
        if service_name not in self.mcp_clients:
            raise ValueError(f"MCP 서비스 '{service_name}'가 설정되지 않음")
        
        try:
            client = self.mcp_clients[service_name]
            return await client.call(method, params)
        except Exception:
            return None
    
    async def generate_response(
        self, 
        prompt: str, 
        context_documents: List[Document] = None,
        **kwargs
    ) -> str:
        """
        🤖 LLM을 사용한 응답 생성 (RAG + LLM 통합) 고정형
        
        🔄 생성 과정:
        1. RAG 문서가 있으면 컨텍스트로 구성
        2. 프롬프트와 컨텍스트를 결합하여 최종 프롬프트 생성
        3. OpenAI GPT 모델로 분석 및 응답 생성
        4. 생성된 텍스트 반환
        
        💡 컨텍스트 활용:
        - 최대 5개 문서를 컨텍스트로 사용 (토큰 제한)
        - 각 문서는 "Document 1:", "Document 2:" 형태로 구성
        - 검색된 실제 데이터를 바탕으로 한 정확한 분석 제공
        
        Args:
            prompt: 입력 프롬프트 (예: "네이버의 워라밸을 분석해줘")
            context_documents: RAG로 검색된 관련 문서들
            **kwargs: 추가 LLM 파라미터 (temperature, max_tokens 등)
            
        Returns:
            str: LLM이 생성한 분석 응답
        """
        if context_documents:
            context_text = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(context_documents[:5])
            ])
            full_prompt = f"Context:\n{context_text}\n\nQuery: {prompt}"
        else:
            full_prompt = prompt
        
        try:
            response = await self.llm.ainvoke(full_prompt, **kwargs)
            return response.content
        except Exception:
            return ""
    
    async def validate_result(
        self, 
        result: Dict[str, Any], 
        required_fields: List[str] = None
    ) -> tuple[bool, List[str]]:
        """
        에이전트 실행 결과를 검증하는 메서드
        
        🔍 검증 항목:
        - 필수 필드 존재 여부
        - 신뢰도 점수 임계값
        - 출처 정보 필수 여부
        
        Args:
            result: 검증할 결과 딕셔너리
            required_fields: 필수 필드 목록 (선택사항)
            
        Returns:
            tuple[bool, List[str]]: (검증 성공 여부, 오류 목록)
        """
        errors = []  # 검증 오류 목록
        
        # 출력 검증이 비활성화된 경우 검증 건너뛰기
        if not self.config.validate_outputs:
            return True, errors
        
        # 필수 필드 존재 여부 확인
        if required_fields:
            for field in required_fields:
                if field not in result:
                    errors.append(f"필수 필드 누락: {field}")
        
        # 신뢰도 점수 임계값 확인
        confidence = result.get("confidence_score", 0.0)
        if confidence < self.config.min_confidence_threshold:
            errors.append(f"신뢰도 점수 {confidence}가 임계값 {self.config.min_confidence_threshold}보다 낮음")
        
        # 출처 정보 필수 여부 확인
        if self.config.require_sources and not result.get("sources"):
            errors.append("출처 정보가 필수이지만 제공되지 않음")
        
        return len(errors) == 0, errors  # 오류가 없으면 True, 있으면 False
    
    def update_performance_metrics(self, execution_time: float, success: bool) -> None:
        """
        에이전트 성능 지표를 업데이트하는 메서드
        
        Args:
            execution_time: 실행 시간 (초)
            success: 실행 성공 여부
        """
        self._execution_count += 1  # 총 실행 횟수 증가
        self._total_execution_time += execution_time  # 총 실행 시간 누적
        if success:
            self._success_count += 1  # 성공 횟수 증가
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        에이전트 성능 통계를 반환하는 메서드
        
        Returns:
            Dict[str, Any]: 성능 통계 정보
            - execution_count: 총 실행 횟수
            - success_rate: 성공률 (0.0~1.0)
            - average_execution_time: 평균 실행 시간
            - total_execution_time: 총 실행 시간
        """
        # 실행 기록이 없는 경우
        if self._execution_count == 0:
            return {
                "execution_count": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0
            }
        
        # 성능 통계 계산 및 반환
        return {
            "execution_count": self._execution_count,  # 총 실행 횟수
            "success_rate": self._success_count / self._execution_count,  # 성공률
            "average_execution_time": self._total_execution_time / self._execution_count,  # 평균 실행 시간
            "total_execution_time": self._total_execution_time  # 총 실행 시간
        }
    
    async def _execute_with_error_handling(
        self,
        execution_func,
        *args,
        **kwargs
    ) -> AgentResult:
        """
        포괄적인 오류 처리와 함께 에이전트 함수를 실행하는 내부 메서드
        
        🛡️ 주요 기능:
        - 실행 시간 측정 및 추적
        - 결과 검증 및 품질 관리
        - 오류 처리 및 복구
        - 성능 지표 업데이트
        
        Args:
            execution_func: 실행할 함수
            *args: 위치 인자들
            **kwargs: 키워드 인자들
            
        Returns:
            AgentResult: 실행 결과와 메타데이터를 포함한 결과 객체
        """
        start_time = time.time()  # 실행 시작 시간 기록
        
        try:
            # 메인 함수 실행
            result = await execution_func(*args, **kwargs)
            execution_time = time.time() - start_time  # 실행 시간 계산
            
            # 결과가 딕셔너리인 경우 검증 수행
            if isinstance(result, dict):
                is_valid, validation_errors = await self.validate_result(result)
                if not is_valid:
                    # 검증 실패 시 실패 결과 반환
                    self.update_performance_metrics(execution_time, False)
                    return AgentResult(
                        agent_name=self.name,
                        success=False,
                        error_message=f"검증 실패: {', '.join(validation_errors)}",
                        execution_time=execution_time
                    )
            
            # 성능 지표 업데이트 (성공)
            self.update_performance_metrics(execution_time, True)
            
            # 성공 결과 반환
            return AgentResult(
                agent_name=self.name,
                success=True,
                result=result if isinstance(result, dict) else {"output": result},
                execution_time=execution_time,
                confidence_score=result.get("confidence_score", 1.0) if isinstance(result, dict) else 1.0
            )
            
        except Exception as e:
            # 예외 발생 시 처리
            execution_time = time.time() - start_time
            self.update_performance_metrics(execution_time, False)
            
            return AgentResult(
                agent_name=self.name,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )


class AgentOrchestrator:
    """
    복잡한 분석 작업을 위해 여러 에이전트를 조율하는 오케스트레이터 클래스
    
    🎼 주요 역할:
    - 여러 에이전트의 등록 및 관리
    - 순차적/병렬 에이전트 실행
    - 실행 기록 및 성능 모니터링
    - 결과 집계 및 분석
    """
    
    def __init__(self):
        """오케스트레이터 초기화"""
        self.agents: Dict[str, BaseAgent] = {}  # 등록된 에이전트들
        self.execution_history: List[Dict[str, Any]] = []  # 실행 기록
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        에이전트를 오케스트레이터에 등록
        
        Args:
            agent: 등록할 BaseAgent 인스턴스
        """
        self.agents[agent.name] = agent
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """
        이름으로 에이전트 조회
        
        Args:
            name: 찾을 에이전트 이름
            
        Returns:
            Optional[BaseAgent]: 찾은 에이전트 또는 None
        """
        return self.agents.get(name)
    
    async def execute_agents(
        self,
        agent_names: List[str],
        query: str,
        context: Optional[Dict[str, Any]] = None,
        parallel: bool = True
    ) -> Dict[str, AgentResult]:
        """
        Execute multiple agents.
        
        Args:
            agent_names: List of agent names to execute
            query: Query to pass to agents
            context: Additional context
            parallel: Execute agents in parallel if True
            
        Returns:
            Dictionary mapping agent names to their results
        """
        
        # 유효한 에이전트만 필터링
        valid_agents = [
            (name, self.agents[name]) 
            for name in agent_names 
            if name in self.agents
        ]
        
        # 유효한 에이전트가 없는 경우 빈 결과 반환
        if not valid_agents:
            return {}
        
        start_time = time.time()  # 실행 시작 시간 기록
        
        if parallel:
            # 병렬 실행 모드
            tasks = [
                agent.execute(query, context)
                for _, agent in valid_agents
            ]
            
            # 모든 에이전트 동시 실행
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 처리
            agent_results = {}
            for (name, _), result in zip(valid_agents, results):
                if isinstance(result, Exception):
                    # 예외 발생한 에이전트는 실패 결과로 처리
                    agent_results[name] = AgentResult(
                        agent_name=name,
                        success=False,
                        error_message=str(result),
                        execution_time=time.time() - start_time
                    )
                else:
                    agent_results[name] = result
        
        else:
            # 순차 실행 모드
            agent_results = {}
            for name, agent in valid_agents:
                result = await agent.execute(query, context)
                agent_results[name] = result
        
        # 실행 기록 저장
        total_time = time.time() - start_time
        self.execution_history.append({
            "timestamp": datetime.now(),  # 실행 시간
            "agent_names": agent_names,  # 실행된 에이전트 목록
            "parallel": parallel,  # 실행 모드
            "total_time": total_time,  # 총 실행 시간
            "success_count": sum(1 for r in agent_results.values() if r.success),  # 성공 횟수
            "total_agents": len(agent_results)  # 총 에이전트 수
        })
        
        return agent_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        등록된 모든 에이전트의 성능 요약 정보를 반환
        
        Returns:
            Dict[str, Any]: 성능 요약 정보
            - agent_stats: 각 에이전트별 성능 통계
            - orchestrator_stats: 오케스트레이터 전체 통계
        """
        # 각 에이전트의 성능 통계 수집
        agent_stats = {
            name: agent.get_performance_stats()
            for name, agent in self.agents.items()
        }
        
        # 오케스트레이터 전체 통계 계산
        orchestrator_stats = {
            "total_orchestrations": len(self.execution_history),  # 총 오케스트레이션 횟수
            "average_orchestration_time": (
                sum(h["total_time"] for h in self.execution_history) / len(self.execution_history)
                if self.execution_history else 0.0  # 평균 오케스트레이션 시간
            )
        }
        
        return {
            "agent_stats": agent_stats,  # 에이전트별 통계
            "orchestrator_stats": orchestrator_stats  # 오케스트레이터 통계
        }


# 에이전트 팩토리 함수들
def create_agent(agent_type: str, config: Optional[AgentConfig] = None) -> BaseAgent:
    """
    타입별 에이전트를 생성하는 팩토리 함수
    
    🏭 지원하는 에이전트 타입:
    - culture: 회사 문화 분석 에이전트
    - compensation: 연봉 및 복리후생 분석 에이전트  
    - growth: 성장성 및 안정성 분석 에이전트
    - career: 커리어 경로 분석 에이전트
    
    Args:
        agent_type: 생성할 에이전트 타입 (culture/compensation/growth/career)
        config: 에이전트 설정 (기본값 사용 시 None)
        
    Returns:
        BaseAgent: 설정된 에이전트 인스턴스
        
    Raises:
        ValueError: 알려지지 않은 에이전트 타입인 경우
    """
    
    # 순환 import 방지를 위해 여기서 import
    # 새로운 전문 에이전트들만 import
    from .company_culture_agent import CompanyCultureAgent
    from .work_life_balance_agent import WorkLifeBalanceAgent
    from .management_agent import ManagementAgent
    from .salary_benefits_agent import SalaryBenefitsAgent
    from .career_growth_agent import CareerGrowthAgent
    
    # 에이전트 타입과 클래스 매핑 (새로운 전문 에이전트들)
    agent_classes = {
        # 전문 에이전트들 (각 컬렉션별 + general)
        "company_culture": CompanyCultureAgent,  # 기업문화 전문
        "work_life_balance": WorkLifeBalanceAgent,  # 워라밸 전문
        "management": ManagementAgent,  # 경영진 전문
        "salary_benefits": SalaryBenefitsAgent,  # 연봉/복지 전문
        "career_growth": CareerGrowthAgent,  # 커리어 성장 전문
        
        # 하위 호환성을 위한 별칭
        "culture": CompanyCultureAgent,  # 기존 culture -> company_culture
        "compensation": SalaryBenefitsAgent,  # 기존 compensation -> salary_benefits
        "growth": CareerGrowthAgent,  # 기존 growth -> career_growth  
        "career": CareerGrowthAgent  # 기존 career -> career_growth
    }
    
    # 지원되지 않는 타입 체크
    if agent_type not in agent_classes:
        raise ValueError(f"알 수 없는 에이전트 타입: {agent_type}")
    
    # 에이전트 인스턴스 생성 및 반환
    return agent_classes[agent_type](config=config)


def get_agent_by_type(agent_type: str) -> Type[BaseAgent]:
    """
    타입별 에이전트 클래스를 반환하는 함수
    
    Args:
        agent_type: 조회할 에이전트 타입
        
    Returns:
        Type[BaseAgent]: 에이전트 클래스 (없으면 None)
    """
    
    # 순환 import 방지를 위해 여기서 import
    # 새로운 전문 에이전트들만 import
    from .company_culture_agent import CompanyCultureAgent
    from .work_life_balance_agent import WorkLifeBalanceAgent
    from .management_agent import ManagementAgent
    from .salary_benefits_agent import SalaryBenefitsAgent
    from .career_growth_agent import CareerGrowthAgent
    
    # 에이전트 타입과 클래스 매핑 (새로운 전문 에이전트들)
    agent_classes = {
        # 전문 에이전트들 (각 컬렉션별 + general)
        "company_culture": CompanyCultureAgent,  # 기업문화 전문 클래스
        "work_life_balance": WorkLifeBalanceAgent,  # 워라밸 전문 클래스
        "management": ManagementAgent,  # 경영진 전문 클래스
        "salary_benefits": SalaryBenefitsAgent,  # 연봉/복지 전문 클래스
        "career_growth": CareerGrowthAgent,  # 커리어 성장 전문 클래스
        
        # 하위 호환성을 위한 별칭
        "culture": CompanyCultureAgent,  # 기존 culture -> company_culture
        "compensation": SalaryBenefitsAgent,  # 기존 compensation -> salary_benefits
        "growth": CareerGrowthAgent,  # 기존 growth -> career_growth  
        "career": CareerGrowthAgent  # 기존 career -> career_growth
    }
    
    return agent_classes.get(agent_type)  # 해당 타입의 클래스 반환 (없으면 None)