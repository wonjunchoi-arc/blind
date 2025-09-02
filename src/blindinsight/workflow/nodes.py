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


"""
#node.py
"""
LangGraph 워크플로우 노드 - 에이전트 구조 최적화

전문 에이전트 5개에 맞춘 노드 설계:
- 각 노드는 해당 전문 에이전트를 호출
- 표준화된 에이전트 결과 처리
- 불필요한 복잡성 제거
"""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List

from ..models.base import ValidationError
from ..models.user import UserProfile
from .state import WorkflowState


class WorkflowNode:
    """워크플로우 노드 기본 클래스"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """노드 실행 - 공통 처리 로직"""
        start_time = time.time()
        state.update_stage(self.name)
        state.add_debug_log(f"Starting {self.name}")
        
        try:
            result_state = await self._execute_logic(state)
            duration = time.time() - start_time
            state.record_stage_timing(self.name, duration)
            state.add_debug_log(f"Completed {self.name} in {duration:.2f}s")
            return result_state
            
        except Exception as e:
            duration = time.time() - start_time
            state.record_stage_timing(self.name, duration)
            state.add_error(f"Node execution failed: {str(e)}")
            return state
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """서브클래스에서 구현할 실제 로직"""
        raise NotImplementedError


class InputValidationNode(WorkflowNode):
    """입력 검증 노드"""
    
    def __init__(self):
        super().__init__("input_validation")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """입력 파라미터 검증"""
        request = state.request
        
        if not request.company or not request.company.strip():
            state.add_error("Company name is required")
            return state
        
        # 회사명 정규화
        request.company = request.company.strip()
        
        # 사용자 프로필 검증
        if state.user_profile:
            try:
                state.add_debug_log("User profile validation completed")
            except ValidationError as e:
                state.add_warning(f"User profile validation warning: {str(e)}")
        
        # 초기 진행률 설정
        state.update_progress(0.1)
        
        # 검증 결과 저장
        state.store_agent_output("input_validation", {
            "validated_company": request.company,
            "validated_position": request.position,
            "has_user_profile": state.user_profile is not None,
            "analysis_type": request.analysis_type
        })
        
        state.add_debug_log("Input validation completed successfully")
        return state


class CompanyCultureNode(WorkflowNode):
    """기업문화 분석 노드"""
    
    def __init__(self):
        super().__init__("company_culture_analysis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """기업문화 분석 실행"""
        if state.should_skip_stage("company_culture_analysis"):
            state.add_debug_log("Skipping company culture analysis")
            state.update_progress(state.progress + 0.16)
            return state
        
        try:
            from ..agents.company_culture_agent import CompanyCultureAgent
            
            agent = CompanyCultureAgent()
            state.add_debug_log("Starting company culture analysis")
            
            context = {
                "company_name": state.request.company,
                "position": state.request.position,
                "timestamp": datetime.now(),
                "analysis_type": "comprehensive"
            }
            
            agent_result = await agent.execute(
                query=state.request.company,
                context=context,
                user_profile=state.user_profile
            )
            
            if agent_result.success:
                state.store_agent_result("company_culture", agent_result.result)
                state.store_agent_output("company_culture_agent", {
                    "analysis_data": agent_result.result,
                    "confidence": agent_result.confidence_score,
                    "execution_time": agent_result.execution_time
                })
                state.update_progress(state.progress + 0.16)
                state.add_debug_log("Company culture analysis completed")
            else:
                state.add_error(f"Company culture analysis failed: {agent_result.error_message}")
            
        except Exception as e:
            state.add_error(f"Company culture analysis failed: {str(e)}")
        
        return state


class WorkLifeBalanceNode(WorkflowNode):
    """워라밸 분석 노드"""
    
    def __init__(self):
        super().__init__("work_life_balance_analysis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """워라밸 분석 실행"""
        if state.should_skip_stage("work_life_balance_analysis"):
            state.add_debug_log("Skipping work life balance analysis")
            state.update_progress(state.progress + 0.16)
            return state
        
        try:
            from ..agents.work_life_balance_agent import WorkLifeBalanceAgent
            
            agent = WorkLifeBalanceAgent()
            state.add_debug_log("Starting work life balance analysis")
            
            context = {
                "company_name": state.request.company,
                "position": state.request.position,
                "timestamp": datetime.now(),
                "analysis_type": "comprehensive"
            }
            
            agent_result = await agent.execute(
                query=state.request.company,
                context=context,
                user_profile=state.user_profile
            )
            
            if agent_result.success:
                state.store_agent_result("work_life_balance", agent_result.result)
                state.store_agent_output("work_life_balance_agent", {
                    "analysis_data": agent_result.result,
                    "confidence": agent_result.confidence_score,
                    "execution_time": agent_result.execution_time
                })
                state.update_progress(state.progress + 0.16)
                state.add_debug_log("Work life balance analysis completed")
            else:
                state.add_error(f"Work life balance analysis failed: {agent_result.error_message}")
            
        except Exception as e:
            state.add_error(f"Work life balance analysis failed: {str(e)}")
        
        return state


class ManagementNode(WorkflowNode):
    """경영진 분석 노드"""
    
    def __init__(self):
        super().__init__("management_analysis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """경영진 분석 실행"""
        if state.should_skip_stage("management_analysis"):
            state.add_debug_log("Skipping management analysis")
            state.update_progress(state.progress + 0.16)
            return state
        
        try:
            from ..agents.management_agent import ManagementAgent
            
            agent = ManagementAgent()
            state.add_debug_log("Starting management analysis")
            
            context = {
                "company_name": state.request.company,
                "position": state.request.position,
                "timestamp": datetime.now(),
                "analysis_type": "comprehensive"
            }
            
            agent_result = await agent.execute(
                query=state.request.company,
                context=context,
                user_profile=state.user_profile
            )
            
            if agent_result.success:
                state.store_agent_result("management", agent_result.result)
                state.store_agent_output("management_agent", {
                    "analysis_data": agent_result.result,
                    "confidence": agent_result.confidence_score,
                    "execution_time": agent_result.execution_time
                })
                state.update_progress(state.progress + 0.16)
                state.add_debug_log("Management analysis completed")
            else:
                state.add_error(f"Management analysis failed: {agent_result.error_message}")
            
        except Exception as e:
            state.add_error(f"Management analysis failed: {str(e)}")
        
        return state


class SalaryBenefitsNode(WorkflowNode):
    """연봉/복지 분석 노드"""
    
    def __init__(self):
        super().__init__("salary_benefits_analysis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """연봉/복지 분석 실행"""
        if state.should_skip_stage("salary_benefits_analysis"):
            state.add_debug_log("Skipping salary benefits analysis")
            state.update_progress(state.progress + 0.16)
            return state
        
        try:
            from ..agents.salary_benefits_agent import SalaryBenefitsAgent
            
            agent = SalaryBenefitsAgent()
            state.add_debug_log("Starting salary benefits analysis")
            
            context = {
                "company_name": state.request.company,
                "position": state.request.position,
                "timestamp": datetime.now(),
                "analysis_type": "comprehensive"
            }
            
            agent_result = await agent.execute(
                query=state.request.company,
                context=context,
                user_profile=state.user_profile
            )
            
            if agent_result.success:
                state.store_agent_result("salary_benefits", agent_result.result)
                state.store_agent_output("salary_benefits_agent", {
                    "analysis_data": agent_result.result,
                    "confidence": agent_result.confidence_score,
                    "execution_time": agent_result.execution_time
                })
                state.update_progress(state.progress + 0.16)
                state.add_debug_log("Salary benefits analysis completed")
            else:
                state.add_error(f"Salary benefits analysis failed: {agent_result.error_message}")
            
        except Exception as e:
            state.add_error(f"Salary benefits analysis failed: {str(e)}")
        
        return state


class CareerGrowthNode(WorkflowNode):
    """커리어 성장 분석 노드"""
    
    def __init__(self):
        super().__init__("career_growth_analysis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """커리어 성장 분석 실행"""
        if state.should_skip_stage("career_growth_analysis"):
            state.add_debug_log("Skipping career growth analysis")
            state.update_progress(state.progress + 0.16)
            return state
        
        try:
            from ..agents.career_growth_agent import CareerGrowthAgent
            
            agent = CareerGrowthAgent()
            state.add_debug_log("Starting career growth analysis")
            
            context = {
                "company_name": state.request.company,
                "position": state.request.position,
                "timestamp": datetime.now(),
                "analysis_type": "comprehensive"
            }
            
            agent_result = await agent.execute(
                query=state.request.company,
                context=context,
                user_profile=state.user_profile
            )
            
            if agent_result.success:
                state.store_agent_result("career_growth", agent_result.result)
                state.store_agent_output("career_growth_agent", {
                    "analysis_data": agent_result.result,
                    "confidence": agent_result.confidence_score,
                    "execution_time": agent_result.execution_time
                })
                state.update_progress(state.progress + 0.16)
                state.add_debug_log("Career growth analysis completed")
            else:
                state.add_error(f"Career growth analysis failed: {agent_result.error_message}")
            
        except Exception as e:
            state.add_error(f"Career growth analysis failed: {str(e)}")
        
        return state


class ParallelAnalysisNode(WorkflowNode):
    """병렬 분석 노드 - 5개 에이전트를 동시에 실행"""
    
    def __init__(self):
        super().__init__("parallel_analysis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """5개 전문 에이전트 병렬 실행"""
        
        # 개별 노드 인스턴스 생성
        nodes = [
            CompanyCultureNode(),
            WorkLifeBalanceNode(), 
            ManagementNode(),
            SalaryBenefitsNode(),
            CareerGrowthNode()
        ]
        
        # 실행할 태스크 준비
        tasks = []
        for node in nodes:
            if not state.should_skip_stage(node.name):
                tasks.append(node._execute_logic(state))
        
        if not tasks:
            state.add_warning("No analysis tasks to execute")
            return state
        
        try:
            state.add_debug_log(f"Starting parallel execution of {len(tasks)} analysis tasks")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 병합
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    state.add_error(f"Parallel task {i} failed: {str(result)}")
                elif isinstance(result, WorkflowState):
                    # 분석 결과 병합
                    if result.company_culture_result:
                        state.company_culture_result = result.company_culture_result
                    if result.work_life_balance_result:
                        state.work_life_balance_result = result.work_life_balance_result
                    if result.management_result:
                        state.management_result = result.management_result
                    if result.salary_benefits_result:
                        state.salary_benefits_result = result.salary_benefits_result
                    if result.career_growth_result:
                        state.career_growth_result = result.career_growth_result
                    
                    # 메타데이터 병합
                    state.agent_outputs.update(result.agent_outputs)
                    state.errors.extend(result.errors)
                    state.warnings.extend(result.warnings)
            
            state.update_progress(0.9)
            state.add_debug_log("Parallel analysis execution completed")
            
        except Exception as e:
            state.add_error(f"Parallel execution failed: {str(e)}")
        
        return state


class SynthesisNode(WorkflowNode):
    """결과 종합 노드"""
    
    def __init__(self):
        super().__init__("synthesis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """분석 결과 종합"""
        state.add_debug_log("Starting result synthesis")
        
        completed_analyses = state.get_completed_analyses()
        
        if not completed_analyses:
            state.add_error("No analysis results available for synthesis")
            return state
        
        state.add_debug_log(f"Synthesizing results from: {', '.join(completed_analyses)}")
        
        # 종합 결과 생성
        synthesis_result = {
            "company_name": state.request.company,
            "position": state.request.position,
            "completed_analyses": completed_analyses,
            "analysis_summary": {},
            "overall_insights": [],
            "synthesis_timestamp": datetime.now().isoformat()
        }
        
        # 각 분석 결과를 요약에 포함
        all_results = state.get_all_results()
        for analysis_type, result in all_results.items():
            if result:
                synthesis_result["analysis_summary"][analysis_type] = {
                    "strengths_count": len(result.get("strengths", {}).get("strengths", [])) if isinstance(result.get("strengths"), dict) else 0,
                    "weaknesses_count": len(result.get("weaknesses", {}).get("strengths", [])) if isinstance(result.get("weaknesses"), dict) else 0,
                    "confidence_score": result.get("confidence_score", 0.0)
                }
        
        # 종합 결과 저장
        state.comprehensive_result = synthesis_result
        
        state.store_agent_output("synthesis", {
            "completed_analyses": completed_analyses,
            "total_analyses": len(completed_analyses),
            "synthesis_timestamp": datetime.now().isoformat()
        })
        
        state.update_progress(0.95)
        state.add_debug_log("Result synthesis completed")
        
        return state


class ReportGenerationNode(WorkflowNode):
    """최종 보고서 생성 노드"""
    
    def __init__(self):
        super().__init__("report_generation")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        """최종 보고서 생성"""
        state.add_debug_log("Starting final report generation")
        
        if not state.comprehensive_result:
            state.add_error("No synthesis result available for report generation")
            return state
        
        try:
            # 간단한 보고서 생성 (실제로는 ReportGenerator 사용)
            final_report = {
                "company": state.request.company,
                "position": state.request.position,
                "analysis_date": datetime.now().isoformat(),
                "completed_analyses": state.get_completed_analyses(),
                "results": state.get_all_results(),
                "performance": state.get_performance_summary(),
                "overall_recommendation": "분석 완료됨"
            }
            
            # 최종 보고서를 comprehensive_result에 업데이트
            state.comprehensive_result.update({
                "final_report": final_report,
                "report_generated_at": datetime.now().isoformat()
            })
            
            state.store_agent_output("report_generator", {
                "report_sections": len(final_report),
                "total_analyses": len(state.get_completed_analyses()),
                "generation_timestamp": datetime.now().isoformat()
            })
            
            state.update_progress(1.0)
            state.add_debug_log("Final report generation completed")
            
        except Exception as e:
            state.add_error(f"Report generation failed: {str(e)}")
        
        return state


# 노드 생성 함수들
def create_input_validation_node() -> WorkflowNode:
    return InputValidationNode()

def create_company_culture_node() -> WorkflowNode:
    return CompanyCultureNode()

def create_work_life_balance_node() -> WorkflowNode:
    return WorkLifeBalanceNode()

def create_management_node() -> WorkflowNode:
    return ManagementNode()

def create_salary_benefits_node() -> WorkflowNode:
    return SalaryBenefitsNode()

def create_career_growth_node() -> WorkflowNode:
    return CareerGrowthNode()

def create_parallel_analysis_node() -> WorkflowNode:
    return ParallelAnalysisNode()

def create_synthesis_node() -> WorkflowNode:
    return SynthesisNode()

def create_report_generation_node() -> WorkflowNode:
    return ReportGenerationNode()


# 노드 등록 - 동적 생성을 위한 레지스트리
NODE_REGISTRY = {
    "input_validation": create_input_validation_node,
    "company_culture_analysis": create_company_culture_node,
    "work_life_balance_analysis": create_work_life_balance_node,
    "management_analysis": create_management_node,
    "salary_benefits_analysis": create_salary_benefits_node,
    "career_growth_analysis": create_career_growth_node,
    "parallel_analysis": create_parallel_analysis_node,
    "synthesis": create_synthesis_node,
    "report_generation": create_report_generation_node
}


def create_node(node_type: str) -> WorkflowNode:
    """타입별 워크플로우 노드 생성"""
    if node_type not in NODE_REGISTRY:
        raise ValueError(f"Unknown node type: {node_type}")
    
    return NODE_REGISTRY[node_type]()