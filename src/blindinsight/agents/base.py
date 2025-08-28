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
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

from ..models.base import BaseModel, settings
from ..models.user import UserProfile


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
    model_name: str = "gpt-4-turbo"  # 사용할 GPT 모델명
    temperature: float = 0.3  # 응답 창의성 (0.0: 일관성, 1.0: 창의성)
    max_tokens: int = 4000  # 최대 생성 토큰 수
    timeout: int = 60  # API 호출 타임아웃 (초)
    
    # RAG 검색 설정 - ChromaDB 벡터 검색
    max_retrievals: int = 20  # 검색할 최대 문서 수
    similarity_threshold: float = 0.7  # 유사도 임계값 (0.0~1.0)
    enable_reranking: bool = True  # 재순위 매김 활성화
    
    # 성능 최적화 설정
    enable_caching: bool = True  # 결과 캐싱 활성화
    cache_ttl: int = 3600  # 캐시 유지 시간 (초, 1시간)
    max_retries: int = 3  # 실패 시 최대 재시도 횟수
    
    # 품질 관리 설정
    min_confidence_threshold: float = 0.6  # 최소 신뢰도 임계값
    require_sources: bool = True  # 출처 정보 필수 여부
    validate_outputs: bool = True  # 출력 결과 검증 활성화
    
    class Config:
        """Pydantic 설정 클래스 - 구성 예제 정의"""
        schema_extra = {
            "example": {
                "model_name": "gpt-4-turbo",  # GPT-4 터보 모델 사용
                "temperature": 0.3,  # 낮은 창의성으로 일관된 분석
                "max_retrievals": 20,  # 최대 20개 관련 문서 검색
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
        
        # LLM 모델 초기화 - OpenAI ChatGPT
        self.llm = ChatOpenAI(
            model=self.config.model_name,  # 모델명 설정
            temperature=self.config.temperature,  # 창의성 수준
            max_tokens=self.config.max_tokens,  # 최대 토큰 수
            timeout=self.config.timeout,  # 타임아웃 설정
            api_key=settings.openai_api_key  # API 키 (환경변수에서 로드)
        )
        
        # 대화 기록 관리 메모리 초기화
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",  # 기록 키
            return_messages=True  # 메시지 형태로 반환
        )
        
        # RAG 검색 엔진 초기화 (하위 클래스에서 설정)
        self.rag_retriever = None
        
        # MCP 클라이언트 초기화 (하위 클래스에서 설정)
        self.mcp_clients = {}
        
        # 성능 추적 변수 초기화
        self._execution_count = 0  # 총 실행 횟수
        self._total_execution_time = 0.0  # 총 실행 시간
        self._success_count = 0  # 성공 실행 횟수
    
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
    
    async def retrieve_knowledge(
        self, 
        query: str, 
        category: Optional[str] = None,
        k: int = None
    ) -> List[Document]:
        """
        🔍 RAG 시스템을 사용한 관련 지식 검색
        
        🏗️ RAG 검색 과정:
        1. 쿼리를 임베딩 벡터로 변환 (OpenAI text-embedding-3-large)
        2. ChromaDB에서 유사한 문서 벡터 검색
        3. 유사도 점수가 임계값 이상인 문서만 필터링
        4. 카테고리별 필터링 적용 (선택적)
        5. 상위 k개 문서 반환
        
        📁 검색 가능한 카테고리:
        - culture_reviews: 회사 문화 리뷰
        - salary_discussions: 연봉 관련 토론
        - career_advice: 커리어 조언
        - interview_reviews: 면접 후기
        - company_general: 회사 일반 정보
        
        Args:
            query: 검색할 질문 (예: "구글 워라밸 어때?")
            category: 카테고리 필터 (예: "culture_reviews")
            k: 검색할 문서 개수 (기본값: config.max_retrievals=20)
            
        Returns:
            List[Document]: 관련 문서 목록 (유사도 점수 포함)
        """
        if not self.rag_retriever:
            return []
        
        k = k or self.config.max_retrievals
        
        try:
            # 검색 파라미터 구성
            search_kwargs = {"k": k}  # 검색할 문서 개수
            if category:
                search_kwargs["filter"] = {"category": category}  # 카테고리 필터 적용
            
            # RAG 검색 실행 - ChromaDB에서 벡터 유사도 검색
            documents = await self.rag_retriever.aget_relevant_documents(
                query,  # 검색 질의
                **search_kwargs  # 검색 옵션
            )
            
            # 유사도 임계값 필터링 (설정된 경우)
            if hasattr(self.rag_retriever, "similarity_threshold"):
                documents = [
                    doc for doc in documents 
                    if getattr(doc, "metadata", {}).get("score", 1.0) >= self.config.similarity_threshold
                ]
            
            return documents[:k]  # 상위 k개 문서 반환
            
        except Exception as e:
            print(f"RAG 검색 실패 - 에이전트 {self.name}: {str(e)}")
            return []  # 실패 시 빈 리스트 반환
    
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
        # MCP 서비스 클라이언트 존재 여부 확인
        if service_name not in self.mcp_clients:
            raise ValueError(f"MCP 서비스 '{service_name}'가 에이전트 {self.name}에 설정되지 않음")
        
        try:
            client = self.mcp_clients[service_name]  # 서비스 클라이언트 가져오기
            return await client.call(method, params)  # 비동기 API 호출 실행
        except Exception as e:
            print(f"MCP 서비스 호출 실패 - {service_name}.{method}: {str(e)}")
            return None  # 실패 시 None 반환
    
    async def generate_response(
        self, 
        prompt: str, 
        context_documents: List[Document] = None,
        **kwargs
    ) -> str:
        """
        🤖 LLM을 사용한 응답 생성 (RAG + LLM 통합)
        
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
        # RAG 문서가 있는 경우 컨텍스트 구성
        if context_documents:
            # 최대 5개 문서로 컨텍스트 제한 (토큰 절약)
            context_text = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(context_documents[:5])
            ])
            
            # RAG 컨텍스트와 질의를 결합한 최종 프롬프트 생성
            full_prompt = f"""Context Information:
{context_text}

Query: {prompt}

Please provide a comprehensive analysis based on the context information above."""
        else:
            # RAG 문서가 없는 경우 원본 프롬프트 사용
            full_prompt = prompt
        
        try:
            # OpenAI LLM을 통한 응답 생성
            response = await self.llm.ainvoke(full_prompt, **kwargs)
            return response.content  # 생성된 텍스트 반환
        except Exception as e:
            print(f"LLM 응답 생성 실패 - 에이전트 {self.name}: {str(e)}")
            return ""  # 실패 시 빈 문자열 반환
    
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
    from .culture_agent import CultureAnalysisAgent
    from .compensation_agent import CompensationAnalysisAgent
    from .growth_agent import GrowthStabilityAgent
    from .career_agent import CareerPathAgent
    
    # 에이전트 타입과 클래스 매핑
    agent_classes = {
        "culture": CultureAnalysisAgent,  # 문화 분석
        "compensation": CompensationAnalysisAgent,  # 연봉 분석
        "growth": GrowthStabilityAgent,  # 성장성 분석
        "career": CareerPathAgent  # 커리어 분석
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
    from .culture_agent import CultureAnalysisAgent
    from .compensation_agent import CompensationAnalysisAgent
    from .growth_agent import GrowthStabilityAgent
    from .career_agent import CareerPathAgent
    
    # 에이전트 타입과 클래스 매핑
    agent_classes = {
        "culture": CultureAnalysisAgent,  # 문화 분석 클래스
        "compensation": CompensationAnalysisAgent,  # 연봉 분석 클래스
        "growth": GrowthStabilityAgent,  # 성장성 분석 클래스
        "career": CareerPathAgent  # 커리어 분석 클래스
    }
    
    return agent_classes.get(agent_type)  # 해당 타입의 클래스 반환 (없으면 None)