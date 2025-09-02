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
#graph.py
"""
LangGraph 워크플로우 - 에이전트 구조 최적화

5개 전문 에이전트에 맞춘 워크플로우:
- CompanyCultureAgent, WorkLifeBalanceAgent, ManagementAgent, SalaryBenefitsAgent, CareerGrowthAgent
- 순차 실행과 병렬 실행 모드 지원
- 불필요한 복잡성 제거 및 성능 최적화
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
    InputValidationNode,
    CompanyCultureNode,
    WorkLifeBalanceNode,
    ManagementNode,
    SalaryBenefitsNode,
    CareerGrowthNode,
    ParallelAnalysisNode,
    SynthesisNode,
    ReportGenerationNode
)


class BlindInsightWorkflow:
    """
    BlindInsight AI 분석 워크플로우 오케스트레이터
    
    5개 전문 에이전트를 조율하여 종합적인 기업 분석 수행:
    
    순차 실행 모드:
    input_validation → company_culture → work_life_balance → management → salary_benefits → career_growth → synthesis → report_generation
    
    병렬 실행 모드:
    input_validation → parallel_analysis (5개 에이전트 동시 실행) → synthesis → report_generation
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
        self.workflow_graph = None
        self.checkpointer = MemorySaver()
        self._build_workflow()
    
    def _build_workflow(self) -> None:
        """LangGraph 워크플로우 구성"""
        workflow = StateGraph(WorkflowState)
        
        # 노드 추가
        workflow.add_node("input_validation", self._input_validation_wrapper)
        
        if self.config.enable_parallel_execution:
            # 병렬 실행: 5개 에이전트를 한 번에
            workflow.add_node("parallel_analysis", self._parallel_analysis_wrapper)
        else:
            # 순차 실행: 각 에이전트를 단계적으로
            workflow.add_node("company_culture_analysis", self._company_culture_wrapper)
            workflow.add_node("work_life_balance_analysis", self._work_life_balance_wrapper)
            workflow.add_node("management_analysis", self._management_wrapper)
            workflow.add_node("salary_benefits_analysis", self._salary_benefits_wrapper)
            workflow.add_node("career_growth_analysis", self._career_growth_wrapper)
        
        # 공통 노드들
        workflow.add_node("synthesis", self._synthesis_wrapper)
        workflow.add_node("report_generation", self._report_generation_wrapper)
        
        # 워크플로우 연결 정의
        self._define_workflow_edges(workflow)
        
        # 진입점 설정
        workflow.set_entry_point("input_validation")
        
        # 컴파일
        self.workflow_graph = workflow.compile(checkpointer=self.checkpointer)
    
    def _define_workflow_edges(self, workflow: StateGraph) -> None:
        """워크플로우 엣지 정의"""
        if self.config.enable_parallel_execution:
            # 병렬 실행 흐름
            workflow.add_edge("input_validation", "parallel_analysis")
            workflow.add_edge("parallel_analysis", "synthesis")
        else:
            # 순차 실행 흐름
            workflow.add_edge("input_validation", "company_culture_analysis")
            workflow.add_edge("company_culture_analysis", "work_life_balance_analysis")
            workflow.add_edge("work_life_balance_analysis", "management_analysis")
            workflow.add_edge("management_analysis", "salary_benefits_analysis")
            workflow.add_edge("salary_benefits_analysis", "career_growth_analysis")
            workflow.add_edge("career_growth_analysis", "synthesis")
        
        # 공통 흐름
        workflow.add_edge("synthesis", "report_generation")
        workflow.add_edge("report_generation", END)
    
    # 노드 래퍼 메서드들
    async def _input_validation_wrapper(self, state: WorkflowState) -> WorkflowState:
        """입력 검증 노드 래퍼"""
        node = InputValidationNode()
        return await node.execute(state)
    
    async def _company_culture_wrapper(self, state: WorkflowState) -> WorkflowState:
        """기업문화 분석 노드 래퍼"""
        node = CompanyCultureNode()
        return await node.execute(state)
    
    async def _work_life_balance_wrapper(self, state: WorkflowState) -> WorkflowState:
        """워라밸 분석 노드 래퍼"""
        node = WorkLifeBalanceNode()
        return await node.execute(state)
    
    async def _management_wrapper(self, state: WorkflowState) -> WorkflowState:
        """경영진 분석 노드 래퍼"""
        node = ManagementNode()
        return await node.execute(state)
    
    async def _salary_benefits_wrapper(self, state: WorkflowState) -> WorkflowState:
        """연봉/복지 분석 노드 래퍼"""
        node = SalaryBenefitsNode()
        return await node.execute(state)
    
    async def _career_growth_wrapper(self, state: WorkflowState) -> WorkflowState:
        """커리어 성장 분석 노드 래퍼"""
        node = CareerGrowthNode()
        return await node.execute(state)
    
    async def _parallel_analysis_wrapper(self, state: WorkflowState) -> WorkflowState:
        """병렬 분석 노드 래퍼"""
        node = ParallelAnalysisNode()
        return await node.execute(state)
    
    async def _synthesis_wrapper(self, state: WorkflowState) -> WorkflowState:
        """결과 종합 노드 래퍼"""
        node = SynthesisNode()
        return await node.execute(state)
    
    async def _report_generation_wrapper(self, state: WorkflowState) -> WorkflowState:
        """보고서 생성 노드 래퍼"""
        node = ReportGenerationNode()
        return await node.execute(state)
    
    async def analyze_company(
        self,
        request: AnalysisRequest,
        user_profile: Optional[UserProfile] = None,
        thread_id: Optional[str] = None
    ) -> WorkflowState:
        """기업 분석 워크플로우 실행"""
        
        # 워크플로우 ID 및 스레드 ID 생성
        workflow_id = str(uuid.uuid4())
        if not thread_id:
            thread_id = f"analysis_{workflow_id}"
        
        # 초기 상태 설정
        initial_state = WorkflowState(
            request=request,
            user_profile=user_profile,
            workflow_id=workflow_id,
            debug_mode=self.config.enable_debug_logging
        )
        
        # 체크포인팅 설정
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # 워크플로우 실행
            result = await self.workflow_graph.ainvoke(
                initial_state.dict(),
                config=config
            )
            
            # 결과를 WorkflowState로 변환
            final_state = WorkflowState(**result)
            
            return final_state
            
        except Exception as e:
            # 워크플로우 실행 오류 처리
            initial_state.add_error(f"Workflow execution failed: {str(e)}")
            return initial_state
    
    async def get_workflow_state(self, thread_id: str) -> Optional[WorkflowState]:
        """스레드 ID로 현재 워크플로우 상태 조회"""
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
        """워크플로우 실행 히스토리 조회"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            return list(self.checkpointer.list(config))
        except Exception:
            return []


class AnalysisWorkflow:
    """간편 분석 워크플로우 인터페이스"""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.workflow = BlindInsightWorkflow(config)
    
    async def analyze(
        self,
        company: str,
        position: Optional[str] = None,
        user_profile: Optional[UserProfile] = None,
        analysis_type: str = "comprehensive"
    ) -> Dict:
        """기업 분석 실행 (간편 버전)"""
        
        # 분석 요청 생성
        request = AnalysisRequest(
            company=company,
            position=position,
            analysis_type=analysis_type
        )
        
        # 워크플로우 실행
        result_state = await self.workflow.analyze_company(
            request=request,
            user_profile=user_profile
        )
        
        # 포맷된 결과 반환
        return {
            "success": not result_state.has_errors(),
            "workflow_id": result_state.workflow_id,
            "progress": result_state.progress,
            "results": result_state.get_all_results(),
            "errors": list(result_state.errors),
            "warnings": list(result_state.warnings),
            "performance": result_state.get_performance_summary(),
            "completed_at": result_state.updated_at.isoformat()
        }


# 워크플로우 생성 유틸리티 함수들
def create_analysis_workflow(
    enable_parallel: bool = True,
    enable_caching: bool = True,
    debug_mode: bool = False
) -> BlindInsightWorkflow:
    """설정이 적용된 분석 워크플로우 생성"""
    config = WorkflowConfig(
        enable_parallel_execution=enable_parallel,
        enable_caching=enable_caching,
        enable_debug_logging=debug_mode
    )
    
    return BlindInsightWorkflow(config)


def create_simple_workflow() -> AnalysisWorkflow:
    """기본 설정으로 간편 워크플로우 생성"""
    return AnalysisWorkflow()


# 빠른 실행 유틸리티 함수들
async def execute_quick_analysis(
    company: str,
    position: Optional[str] = None
) -> Dict:
    """빠른 기업 분석 실행"""
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
    """개인화된 기업 분석 실행"""
    workflow = create_simple_workflow()
    return await workflow.analyze(
        company=company,
        position=position,
        user_profile=user_profile,
        analysis_type="comprehensive"
    )


async def execute_parallel_analysis(
    company: str,
    position: Optional[str] = None,
    debug_mode: bool = False
) -> Dict:
    """병렬 처리 기업 분석 실행"""
    config = WorkflowConfig(
        enable_parallel_execution=True,
        enable_debug_logging=debug_mode
    )
    workflow = AnalysisWorkflow(config)
    
    return await workflow.analyze(
        company=company,
        position=position,
        analysis_type="comprehensive"
    )


async def execute_sequential_analysis(
    company: str,
    position: Optional[str] = None,
    debug_mode: bool = False
) -> Dict:
    """순차 처리 기업 분석 실행"""
    config = WorkflowConfig(
        enable_parallel_execution=False,
        enable_debug_logging=debug_mode
    )
    workflow = AnalysisWorkflow(config)
    
    return await workflow.analyze(
        company=company,
        position=position,
        analysis_type="comprehensive"
    )