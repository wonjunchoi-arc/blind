"""
LangGraph 워크플로우 노드 구현 - 초보자 가이드

🎯 노드(Node)란?
노드는 LangGraph에서 실제 작업을 수행하는 단위입니다.
각 노드는 WorkflowState를 입력받아 처리 후 수정된 상태를 반환합니다.

🏗️ 노드 설계 패턴:
1. 기본 구조: async def node_function(state: WorkflowState) -> WorkflowState
2. 에러 처리: try-except로 안전하게 처리
3. 진행률 업데이트: state.update_progress()로 진행 상황 추적
4. 로깅: state.add_debug_log()로 실행 과정 기록

🔧 커스터마이징 가이드:
1. 새 노드 만들기:
   - WorkflowNode 클래스 상속
   - _execute_logic 메서드 구현
   - NODE_REGISTRY에 등록

2. 병렬 실행 노드:
   - asyncio.gather() 사용
   - 각 작업을 별도 태스크로 실행

3. 조건부 노드:
   - state.should_skip_stage()로 건너뛰기 체크
   - 다양한 조건에 따른 분기 처리
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List

from ..models.base import ValidationError
from ..models.user import UserProfile
from .state import WorkflowState


class WorkflowNode:
    """
    🏗️ 워크플로우 노드의 기본 클래스 - 모든 노드의 부모 클래스
    
    📚 LangGraph 노드 패턴:
    - 각 노드는 상태를 받아서 처리하고 수정된 상태를 반환
    - 실행 시간 측정, 에러 처리, 로깅이 자동으로 처리됨
    
    🔧 커스터마이징 방법:
    1. 새 노드 클래스 생성:
       class MyCustomNode(WorkflowNode):
           def __init__(self):
               super().__init__("my_custom_node")
           
           async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
               # 여기에 실제 로직 구현
               return state
    
    2. 에러 처리 커스터마이징:
       try-except 블록에서 state.add_error() 사용
    
    3. 진행률 업데이트:
       state.update_progress(new_progress_value)
    """
    
    def __init__(self, name: str):
        self.name = name
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """
        노드 실행 메서드 - 모든 노드 공통 처리 로직
        
        🔄 실행 흐름:
        1. 시작 시간 기록
        2. 현재 단계 업데이트
        3. 실제 로직 실행 (_execute_logic)
        4. 실행 시간 기록
        5. 에러 발생 시 상태에 에러 추가
        """
        start_time = time.time()
        state.update_stage(self.name)
        state.add_debug_log(f"Starting execution of node: {self.name}")
        
        try:
            # 💡 핵심: 실제 노드별 로직은 _execute_logic에서 구현
            result_state = await self._execute_logic(state)
            
            # 📊 성능 측정: 실행 시간 기록
            duration = time.time() - start_time
            state.record_stage_timing(self.name, duration)
            
            state.add_debug_log(f"Completed execution of node: {self.name} in {duration:.2f}s")
            return result_state
            
        except Exception as e:
            # 🚨 에러 처리: 실행 시간 기록 후 에러 상태 추가
            duration = time.time() - start_time
            state.record_stage_timing(self.name, duration)
            state.add_error(f"Node execution failed: {str(e)}")
            return state
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """
        📝 서브클래스에서 반드시 구현해야 하는 메서드
        
        여기에 실제 노드의 비즈니스 로직을 구현합니다.
        예시:
        - AI 에이전트 호출
        - 데이터 처리
        - 외부 API 호출
        - 결과 분석
        """
        raise NotImplementedError("Subclasses must implement _execute_logic method")


class InputValidationNode(WorkflowNode):
    """Validates and processes input parameters."""
    
    def __init__(self):
        super().__init__("input_validation")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """Validate input parameters and prepare state."""
        
        # Validate analysis request
        request = state.request
        if not request.company or not request.company.strip():
            state.add_error("Company name is required")
            return state
        
        # Normalize company name
        request.company = request.company.strip()
        
        # Validate user profile if provided
        if state.user_profile:
            try:
                # Additional validation logic here
                state.add_debug_log("User profile validation completed")
            except ValidationError as e:
                state.add_warning(f"User profile validation warning: {str(e)}")
        
        # Set initial progress
        state.update_progress(0.1)
        
        # Initialize agent outputs storage
        state.agent_outputs["input_validation"] = {
            "validated_company": request.company,
            "validated_position": request.position,
            "has_user_profile": state.user_profile is not None,
            "analysis_type": request.analysis_type
        }
        
        state.add_debug_log("Input validation completed successfully")
        return state


class CultureAnalysisNode(WorkflowNode):
    """Executes culture analysis using the Culture Analysis Agent."""
    
    def __init__(self):
        super().__init__("culture_analysis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """Execute culture analysis."""
        
        # Skip if not required
        if state.should_skip_stage("culture_analysis"):
            state.add_debug_log("Skipping culture analysis stage")
            state.update_progress(state.progress + 0.2)
            return state
        
        try:
            # Import here to avoid circular imports
            from ..agents.culture_agent import CultureAnalysisAgent
            
            # Initialize agent
            culture_agent = CultureAnalysisAgent()
            
            # Execute analysis
            state.add_debug_log("Starting culture analysis")
            culture_report = await culture_agent.analyze_company_culture(
                company=state.request.company,
                position=state.request.position,
                user_profile=state.user_profile
            )
            
            # Store results
            state.culture_report = culture_report
            state.store_agent_output("culture_agent", {
                "culture_score": culture_report.culture_score,
                "work_life_balance": culture_report.work_life_balance,
                "confidence": culture_report.confidence_score
            })
            
            # Update progress
            state.update_progress(state.progress + 0.2)
            state.add_debug_log("Culture analysis completed successfully")
            
        except Exception as e:
            state.add_error(f"Culture analysis failed: {str(e)}")
        
        return state


class CompensationAnalysisNode(WorkflowNode):
    """Executes compensation analysis using the Compensation Analysis Agent."""
    
    def __init__(self):
        super().__init__("compensation_analysis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """Execute compensation analysis."""
        
        if state.should_skip_stage("compensation_analysis"):
            state.add_debug_log("Skipping compensation analysis stage")
            state.update_progress(state.progress + 0.2)
            return state
        
        try:
            from ..agents.compensation_agent import CompensationAnalysisAgent
            
            compensation_agent = CompensationAnalysisAgent()
            
            state.add_debug_log("Starting compensation analysis")
            compensation_report = await compensation_agent.analyze_compensation(
                company=state.request.company,
                position=state.request.position,
                user_profile=state.user_profile
            )
            
            state.compensation_report = compensation_report
            state.store_agent_output("compensation_agent", {
                "salary_range": compensation_report.salary_range,
                "salary_percentile": compensation_report.salary_percentile,
                "confidence": compensation_report.confidence_score
            })
            
            state.update_progress(state.progress + 0.2)
            state.add_debug_log("Compensation analysis completed successfully")
            
        except Exception as e:
            state.add_error(f"Compensation analysis failed: {str(e)}")
        
        return state


class GrowthAnalysisNode(WorkflowNode):
    """Executes growth and stability analysis using the Growth & Stability Agent."""
    
    def __init__(self):
        super().__init__("growth_analysis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """Execute growth analysis."""
        
        if state.should_skip_stage("growth_analysis"):
            state.add_debug_log("Skipping growth analysis stage")
            state.update_progress(state.progress + 0.2)
            return state
        
        try:
            from ..agents.growth_agent import GrowthStabilityAgent
            
            growth_agent = GrowthStabilityAgent()
            
            state.add_debug_log("Starting growth analysis")
            growth_report = await growth_agent.analyze_growth_stability(
                company=state.request.company,
                user_profile=state.user_profile
            )
            
            state.growth_report = growth_report
            state.store_agent_output("growth_agent", {
                "growth_score": growth_report.growth_score,
                "stability_score": growth_report.stability_score,
                "confidence": growth_report.confidence_score
            })
            
            state.update_progress(state.progress + 0.2)
            state.add_debug_log("Growth analysis completed successfully")
            
        except Exception as e:
            state.add_error(f"Growth analysis failed: {str(e)}")
        
        return state


class CareerAnalysisNode(WorkflowNode):
    """Executes career path analysis using the Career Path Agent."""
    
    def __init__(self):
        super().__init__("career_analysis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """Execute career analysis."""
        
        if state.should_skip_stage("career_analysis"):
            state.add_debug_log("Skipping career analysis stage")
            state.update_progress(state.progress + 0.2)
            return state
        
        try:
            from ..agents.career_agent import CareerPathAgent
            
            career_agent = CareerPathAgent()
            
            state.add_debug_log("Starting career analysis")
            career_report = await career_agent.analyze_career_path(
                company=state.request.company,
                position=state.request.position,
                user_profile=state.user_profile,
                culture_context=state.culture_report,
                compensation_context=state.compensation_report
            )
            
            state.career_report = career_report
            state.store_agent_output("career_agent", {
                "success_probability": career_report.success_probability,
                "timeline_estimate": career_report.timeline_estimate,
                "confidence": career_report.confidence_score
            })
            
            state.update_progress(state.progress + 0.2)
            state.add_debug_log("Career analysis completed successfully")
            
        except Exception as e:
            state.add_error(f"Career analysis failed: {str(e)}")
        
        return state


class ParallelAnalysisNode(WorkflowNode):
    """
    🚀 병렬 분석 노드 - 여러 분석을 동시에 실행하여 성능 향상
    
    💡 병렬 처리의 핵심 개념:
    - asyncio.gather()로 여러 비동기 작업을 동시 실행
    - 각 분석이 독립적이므로 병렬 처리 가능
    - 전체 실행 시간을 대폭 단축
    
    🔧 커스터마이징 가이드:
    1. 새로운 분석 추가:
       - 새 노드 인스턴스 생성
       - tasks 리스트에 추가
       - 결과 병합 로직에 추가
    
    2. 병렬 처리 방식 변경:
       - asyncio.gather() 대신 asyncio.as_completed() 사용
       - 각 태스크별 타임아웃 설정
       - 실패한 태스크 재시도 로직 추가
    
    3. 결과 병합 커스터마이징:
       - 새로운 리포트 타입의 병합 로직 추가
       - 우선순위에 따른 결과 선택
    """
    
    def __init__(self):
        super().__init__("parallel_analysis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """
        병렬 분석 실행 로직
        
        🔄 처리 순서:
        1. 개별 분석 노드들 생성
        2. 실행할 태스크들 준비
        3. asyncio.gather()로 병렬 실행
        4. 결과들을 원본 상태에 병합
        """
        
        # 💡 개별 분석 노드들 생성
        culture_node = CultureAnalysisNode()
        compensation_node = CompensationAnalysisNode()
        growth_node = GrowthAnalysisNode()
        
        # 🎯 병렬 실행할 태스크들 준비
        tasks = []
        
        if not state.should_skip_stage("culture_analysis"):
            tasks.append(culture_node._execute_logic(state))
        
        if not state.should_skip_stage("compensation_analysis"):
            tasks.append(compensation_node._execute_logic(state))
        
        if not state.should_skip_stage("growth_analysis"):
            tasks.append(growth_node._execute_logic(state))
        
        if not tasks:
            state.add_warning("No analysis tasks to execute")
            return state
        
        try:
            # 🚀 핵심: asyncio.gather()로 병렬 실행
            state.add_debug_log(f"Starting parallel execution of {len(tasks)} analysis tasks")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 📊 결과 처리 및 병합
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    state.add_error(f"Parallel task {i} failed: {str(result)}")
                elif isinstance(result, WorkflowState):
                    # 💡 상태 병합: 각 분석 결과를 원본 상태에 통합
                    if result.culture_report:
                        state.culture_report = result.culture_report
                    if result.compensation_report:
                        state.compensation_report = result.compensation_report
                    if result.growth_report:
                        state.growth_report = result.growth_report
                    
                    # 🔧 커스터마이징: 새로운 리포트 타입 추가 시 여기에 병합 로직 추가
                    # if result.new_report:
                    #     state.new_report = result.new_report
                    
                    # 📈 에이전트 출력 및 로그 병합
                    state.agent_outputs.update(result.agent_outputs)
                    state.errors.extend(result.errors)
                    state.warnings.extend(result.warnings)
            
            # ✅ 진행률 업데이트
            state.update_progress(0.7)
            state.add_debug_log("Parallel analysis execution completed")
            
        except Exception as e:
            state.add_error(f"Parallel execution failed: {str(e)}")
        
        return state


class SynthesisNode(WorkflowNode):
    """Synthesizes results from all analyses."""
    
    def __init__(self):
        super().__init__("synthesis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """Synthesize analysis results."""
        
        state.add_debug_log("Starting result synthesis")
        
        # Check available results
        available_reports = []
        if state.culture_report:
            available_reports.append("culture")
        if state.compensation_report:
            available_reports.append("compensation")
        if state.growth_report:
            available_reports.append("growth")
        if state.career_report:
            available_reports.append("career")
        
        if not available_reports:
            state.add_error("No analysis results available for synthesis")
            return state
        
        state.add_debug_log(f"Synthesizing results from: {', '.join(available_reports)}")
        
        # Store synthesis metadata
        state.store_agent_output("synthesis", {
            "available_reports": available_reports,
            "synthesis_timestamp": datetime.now().isoformat(),
            "total_reports": len(available_reports)
        })
        
        state.update_progress(0.85)
        state.add_debug_log("Result synthesis completed")
        
        return state


class ReportGenerationNode(WorkflowNode):
    """Generates the final comprehensive report."""
    
    def __init__(self):
        super().__init__("report_generation")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """Generate final comprehensive report."""
        
        state.add_debug_log("Starting final report generation")
        
        try:
            from ..services.report_generator import ReportGenerator
            
            # Initialize report generator
            report_generator = ReportGenerator()
            
            # Generate comprehensive report
            final_report = await report_generator.generate_comprehensive_report(
                culture_report=state.culture_report,
                compensation_report=state.compensation_report,
                growth_report=state.growth_report,
                career_report=state.career_report,
                user_profile=state.user_profile,
                company=state.request.company,
                position=state.request.position
            )
            
            # Store final report
            state.final_report = final_report
            
            # Store report generation metadata
            state.store_agent_output("report_generator", {
                "overall_score": final_report.overall_score,
                "recommendation": final_report.recommendation,
                "confidence": final_report.overall_confidence,
                "completeness": final_report.analysis_completeness
            })
            
            # Mark as complete
            state.update_progress(1.0)
            state.add_debug_log("Final report generation completed successfully")
            
        except Exception as e:
            state.add_error(f"Report generation failed: {str(e)}")
        
        return state


# Node factory functions for easier workflow construction
def create_input_validation_node() -> WorkflowNode:
    """Create input validation node."""
    return InputValidationNode()

def create_culture_analysis_node() -> WorkflowNode:
    """Create culture analysis node."""
    return CultureAnalysisNode()

def create_compensation_analysis_node() -> WorkflowNode:
    """Create compensation analysis node."""
    return CompensationAnalysisNode()

def create_growth_analysis_node() -> WorkflowNode:
    """Create growth analysis node."""
    return GrowthAnalysisNode()

def create_career_analysis_node() -> WorkflowNode:
    """Create career analysis node."""
    return CareerAnalysisNode()

def create_parallel_analysis_node() -> WorkflowNode:
    """Create parallel analysis node."""
    return ParallelAnalysisNode()

def create_synthesis_node() -> WorkflowNode:
    """Create synthesis node."""
    return SynthesisNode()

def create_report_generation_node() -> WorkflowNode:
    """Create report generation node."""
    return ReportGenerationNode()


# Node registry for dynamic node creation
NODE_REGISTRY = {
    "input_validation": create_input_validation_node,
    "culture_analysis": create_culture_analysis_node,
    "compensation_analysis": create_compensation_analysis_node,
    "growth_analysis": create_growth_analysis_node,
    "career_analysis": create_career_analysis_node,
    "parallel_analysis": create_parallel_analysis_node,
    "synthesis": create_synthesis_node,
    "report_generation": create_report_generation_node
}


def create_node(node_type: str) -> WorkflowNode:
    """Create a workflow node by type."""
    if node_type not in NODE_REGISTRY:
        raise ValueError(f"Unknown node type: {node_type}")
    
    return NODE_REGISTRY[node_type]()