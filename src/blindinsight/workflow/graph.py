"""
LangGraph 워크플로우 그래프 구성 및 오케스트레이션 - 초보자 가이드

🎯 StateGraph란?
StateGraph는 LangGraph의 핵심으로, 노드들 간의 연결과 실행 순서를 정의합니다.
각 노드는 함수이고, 엣지(Edge)는 노드들 사이의 연결선입니다.

🏗️ 워크플로우 구성 요소:
1. StateGraph(WorkflowState): 상태 클래스를 지정하여 그래프 생성
2. add_node(): 노드(함수) 추가
3. add_edge(): 노드들 사이의 연결 정의
4. set_entry_point(): 시작 노드 지정
5. compile(): 실행 가능한 워크플로우로 컴파일

🔧 커스터마이징 가이드:
1. 새 노드 추가:
   - workflow.add_node("node_name", node_function)
   - 적절한 위치에 엣지 연결

2. 실행 흐름 변경:
   - 엣지 순서를 바꿔서 노드 실행 순서 변경
   - 조건부 엣지로 분기 처리 가능

3. 병렬/순차 실행 전환:
   - config.enable_parallel_execution으로 제어
   - _define_workflow_edges()에서 흐름 정의

4. 체크포인팅:
   - MemorySaver로 중간 상태 저장
   - 실패 시 중단 지점부터 재시작 가능
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..models.analysis import AnalysisRequest
from ..models.user import UserProfile
from .state import WorkflowState, WorkflowConfig
from .nodes import (
    InputValidationNode, CultureAnalysisNode, CompensationAnalysisNode,
    GrowthAnalysisNode, CareerAnalysisNode, ParallelAnalysisNode,
    SynthesisNode, ReportGenerationNode
)


class BlindInsightWorkflow:
    """
    🎯 BlindInsight AI 분석 시스템의 메인 워크플로우 오케스트레이터
    
    LangGraph 기반 워크플로우를 관리하여 여러 AI 에이전트를 조율하고
    종합적인 기업 분석을 수행합니다.
    
    🏗️ 워크플로우 아키텍처:
    
    순차 실행 모드:
    input_validation → culture_analysis → compensation_analysis → growth_analysis → career_analysis → synthesis → report_generation
    
    병렬 실행 모드:
    input_validation → parallel_analysis (culture + compensation + growth 동시 실행) → career_analysis → synthesis → report_generation
    
    🔧 커스터마이징 가이드:
    1. 새로운 분석 타입 추가:
       - 새 노드 래퍼 메서드 추가 (_your_analysis_wrapper)
       - _build_workflow()에서 노드 추가
       - _define_workflow_edges()에서 연결 정의
    
    2. 실행 모드 변경:
       - WorkflowConfig의 enable_parallel_execution으로 제어
       - 병렬 모드: 독립적인 분석들을 동시 실행하여 속도 향상
       - 순차 모드: 단계별 순서 실행으로 디버깅 용이
    
    3. 체크포인팅 활용:
       - thread_id로 세션 관리
       - 실패 시 중간부터 재시작 가능
       - 진행 상황 추적 및 복구
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.workflow_graph = None
        self.checkpointer = MemorySaver()
        self._build_workflow()
    
    def _build_workflow(self) -> None:
        """
        🏗️ LangGraph 워크플로우 구성 - 핵심 메서드
        
        🔄 구성 순서:
        1. StateGraph 생성 (상태 클래스 지정)
        2. 노드들 추가 (각각 함수로 등록)
        3. 엣지들 정의 (노드 간 연결)
        4. 진입점 설정 (시작 노드)
        5. 컴파일 (실행 가능한 그래프로 변환)
        
        💡 커스터마이징 포인트:
        - 새 노드 추가: workflow.add_node("name", wrapper_function)
        - 실행 모드: config.enable_parallel_execution으로 제어
        """
        
        # 1️⃣ 워크플로우 그래프 생성 (상태 클래스 지정)
        workflow = StateGraph(WorkflowState)
        
        # 2️⃣ 노드들 추가 - 각 노드는 비동기 함수로 등록
        workflow.add_node("input_validation", self._input_validation_wrapper)
        
        # 💡 실행 모드에 따른 노드 구성
        if self.config.enable_parallel_execution:
            # 🚀 병렬 실행: culture, compensation, growth를 한 번에
            workflow.add_node("parallel_analysis", self._parallel_analysis_wrapper)
        else:
            # 🔄 순차 실행: 각 분석을 단계적으로
            workflow.add_node("culture_analysis", self._culture_analysis_wrapper)
            workflow.add_node("compensation_analysis", self._compensation_analysis_wrapper)
            workflow.add_node("growth_analysis", self._growth_analysis_wrapper)
        
        # 🎯 공통 노드들 (실행 모드와 무관)
        workflow.add_node("career_analysis", self._career_analysis_wrapper)
        workflow.add_node("synthesis", self._synthesis_wrapper)
        workflow.add_node("report_generation", self._report_generation_wrapper)
        
        # 🔧 커스터마이징: 새 노드 추가 예시
        # workflow.add_node("risk_analysis", self._risk_analysis_wrapper)
        
        # 3️⃣ 워크플로우 연결 정의 (실행 모드에 따라 다름)
        self._define_workflow_edges(workflow)
        
        # 4️⃣ 진입점 설정 (첫 번째로 실행될 노드)
        workflow.set_entry_point("input_validation")
        
        # 5️⃣ 컴파일 - 실행 가능한 워크플로우로 변환 (체크포인팅 포함)
        self.workflow_graph = workflow.compile(checkpointer=self.checkpointer)
    
    def _define_workflow_edges(self, workflow: StateGraph) -> None:
        """
        🔗 워크플로우 엣지(연결) 정의 - 노드 간 실행 순서 결정
        
        💡 엣지(Edge)의 개념:
        - workflow.add_edge(A, B): A 노드 완료 후 B 노드 실행
        - END: 워크플로우 종료 지점
        - 엣지 순서를 바꾸면 실행 흐름이 변경됨
        
        🔧 커스터마이징 가이드:
        1. 새 노드 삽입: 기존 엣지 제거 후 새 엣지 추가
           예: A -> C 사이에 B 삽입
           기존: workflow.add_edge(A, C)
           변경: workflow.add_edge(A, B) + workflow.add_edge(B, C)
        
        2. 조건부 분기: 조건에 따라 다른 노드로 연결
        3. 병렬 vs 순차: 설정에 따라 다른 흐름 구성
        """
        
        if self.config.enable_parallel_execution:
            # 🚀 병렬 실행 흐름
            # input_validation → parallel_analysis → career_analysis → synthesis → report_generation → END
            workflow.add_edge("input_validation", "parallel_analysis")
            workflow.add_edge("parallel_analysis", "career_analysis")
        else:
            # 🔄 순차 실행 흐름
            # input_validation → culture → compensation → growth → career_analysis → synthesis → report_generation → END
            workflow.add_edge("input_validation", "culture_analysis")
            workflow.add_edge("culture_analysis", "compensation_analysis")
            workflow.add_edge("compensation_analysis", "growth_analysis")
            workflow.add_edge("growth_analysis", "career_analysis")
        
        # 📊 공통 흐름 (실행 모드와 무관)
        workflow.add_edge("career_analysis", "synthesis")
        workflow.add_edge("synthesis", "report_generation")
        workflow.add_edge("report_generation", END)
        
        # 🔧 커스터마이징: 새 노드 연결 예시
        # 리스크 분석을 synthesis 전에 추가하는 경우:
        # workflow.add_edge("career_analysis", "risk_analysis")
        # workflow.add_edge("risk_analysis", "synthesis")
    
    # Node wrapper methods for LangGraph integration
    async def _input_validation_wrapper(self, state: WorkflowState) -> WorkflowState:
        """Wrapper for input validation node."""
        node = InputValidationNode()
        return await node.execute(state)
    
    async def _culture_analysis_wrapper(self, state: WorkflowState) -> WorkflowState:
        """Wrapper for culture analysis node."""
        node = CultureAnalysisNode()
        return await node.execute(state)
    
    async def _compensation_analysis_wrapper(self, state: WorkflowState) -> WorkflowState:
        """Wrapper for compensation analysis node."""
        node = CompensationAnalysisNode()
        return await node.execute(state)
    
    async def _growth_analysis_wrapper(self, state: WorkflowState) -> WorkflowState:
        """Wrapper for growth analysis node."""
        node = GrowthAnalysisNode()
        return await node.execute(state)
    
    async def _career_analysis_wrapper(self, state: WorkflowState) -> WorkflowState:
        """Wrapper for career analysis node."""
        node = CareerAnalysisNode()
        return await node.execute(state)
    
    async def _parallel_analysis_wrapper(self, state: WorkflowState) -> WorkflowState:
        """Wrapper for parallel analysis node."""
        node = ParallelAnalysisNode()
        return await node.execute(state)
    
    async def _synthesis_wrapper(self, state: WorkflowState) -> WorkflowState:
        """Wrapper for synthesis node."""
        node = SynthesisNode()
        return await node.execute(state)
    
    async def _report_generation_wrapper(self, state: WorkflowState) -> WorkflowState:
        """Wrapper for report generation node."""
        node = ReportGenerationNode()
        return await node.execute(state)
    
    async def analyze_company(
        self,
        request: AnalysisRequest,
        user_profile: Optional[UserProfile] = None,
        thread_id: Optional[str] = None
    ) -> WorkflowState:
        """
        Execute comprehensive company analysis workflow.
        
        Args:
            request: Analysis request with company and position details
            user_profile: Optional user profile for personalized analysis
            thread_id: Optional thread ID for workflow checkpointing
            
        Returns:
            Final workflow state with comprehensive analysis results
        """
        
        # Generate unique workflow ID and thread ID
        workflow_id = str(uuid.uuid4())
        if not thread_id:
            thread_id = f"analysis_{workflow_id}"
        
        # Initialize workflow state
        initial_state = WorkflowState(
            request=request,
            user_profile=user_profile,
            workflow_id=workflow_id,
            debug_mode=self.config.enable_debug_logging
        )
        
        # Configure checkpointing
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Execute workflow
            result = await self.workflow_graph.ainvoke(
                initial_state.dict(),
                config=config
            )
            
            # Convert result back to WorkflowState
            final_state = WorkflowState(**result)
            
            return final_state
            
        except Exception as e:
            # Handle workflow execution errors
            initial_state.add_error(f"Workflow execution failed: {str(e)}")
            return initial_state
    
    async def get_workflow_state(self, thread_id: str) -> Optional[WorkflowState]:
        """Get current workflow state by thread ID."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoints = self.checkpointer.list(config)
            
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.ts)
                return WorkflowState(**latest_checkpoint.checkpoint["channel_values"])
            
            return None
            
        except Exception:
            return None
    
    def get_workflow_history(self, thread_id: str) -> List[Dict]:
        """Get workflow execution history."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            return list(self.checkpointer.list(config))
        except Exception:
            return []


class AnalysisWorkflow:
    """
    Simplified workflow interface for direct analysis execution.
    
    This class provides a simpler interface for executing analysis workflows
    without dealing with LangGraph complexities directly.
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.workflow = BlindInsightWorkflow(config)
    
    async def analyze(
        self,
        company: str,
        position: Optional[str] = None,
        user_profile: Optional[UserProfile] = None,
        analysis_type: str = "comprehensive"
    ) -> Dict:
        """
        Execute company analysis with simplified parameters.
        
        Args:
            company: Company name to analyze
            position: Optional position/role
            user_profile: Optional user profile for personalization
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary with analysis results
        """
        
        # Create analysis request
        request = AnalysisRequest(
            company=company,
            position=position,
            analysis_type=analysis_type
        )
        
        # Execute workflow
        result_state = await self.workflow.analyze_company(
            request=request,
            user_profile=user_profile
        )
        
        # Return formatted results
        return {
            "success": not result_state.has_errors(),
            "workflow_id": result_state.workflow_id,
            "progress": result_state.progress,
            "results": result_state.get_analysis_results(),
            "errors": list(result_state.errors),
            "warnings": list(result_state.warnings),
            "performance": result_state.get_performance_summary(),
            "completed_at": result_state.updated_at.isoformat()
        }


def create_analysis_workflow(
    enable_parallel: bool = True,
    enable_caching: bool = True,
    debug_mode: bool = False
) -> BlindInsightWorkflow:
    """
    Factory function to create a configured BlindInsight workflow.
    
    Args:
        enable_parallel: Enable parallel execution of analysis agents
        enable_caching: Enable result caching
        debug_mode: Enable debug logging
        
    Returns:
        Configured BlindInsightWorkflow instance
    """
    
    config = WorkflowConfig(
        enable_parallel_execution=enable_parallel,
        enable_caching=enable_caching,
        enable_debug_logging=debug_mode
    )
    
    return BlindInsightWorkflow(config)


def create_simple_workflow() -> AnalysisWorkflow:
    """Create a simple analysis workflow with default configuration."""
    return AnalysisWorkflow()


# Workflow execution utilities
async def execute_quick_analysis(
    company: str,
    position: Optional[str] = None
) -> Dict:
    """
    Execute a quick company analysis with minimal configuration.
    
    Args:
        company: Company name to analyze
        position: Optional position/role
        
    Returns:
        Analysis results dictionary
    """
    
    workflow = create_simple_workflow()
    return await workflow.analyze(
        company=company,
        position=position,
        analysis_type="comprehensive"
    )


async def execute_personalized_analysis(
    company: str,
    position: str,
    user_profile: UserProfile
) -> Dict:
    """
    Execute personalized company analysis with user profile.
    
    Args:
        company: Company name to analyze
        position: Position/role
        user_profile: User profile for personalization
        
    Returns:
        Personalized analysis results dictionary
    """
    
    workflow = create_simple_workflow()
    return await workflow.analyze(
        company=company,
        position=position,
        user_profile=user_profile,
        analysis_type="comprehensive"
    )