"""
LangGraph 워크플로우 상태 관리 - 초보자 가이드

🔥 LangGraph의 핵심 개념:
LangGraph는 상태 기반 워크플로우를 구성하는 프레임워크입니다.
각 노드는 WorkflowState를 입력받아 수정된 상태를 반환합니다.

📚 주요 개념들:
1. StateGraph: 전체 워크플로우의 그래프 구조를 정의
2. WorkflowState: 모든 노드 간에 공유되는 상태 객체
3. Node: 상태를 처리하는 개별 함수/클래스
4. Edge: 노드들 간의 연결과 실행 순서 정의


"""

#state.py


import operator
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Sequence
from pydantic import Field

from ..models.base import BaseModel
from ..models.analysis import AnalysisRequest
from ..models.user import UserProfile


class WorkflowState(BaseModel):
    """
    전문 에이전트 기반 워크플로우 상태 관리
    
    5개 전문 에이전트에 최적화:
    - CompanyCultureAgent (company_culture)
    - WorkLifeBalanceAgent (work_life_balance) 
    - ManagementAgent (management)
    - SalaryBenefitsAgent (salary_benefits)
    - CareerGrowthAgent (career_growth)
    """
    
    # 입력 파라미터
    request: AnalysisRequest = Field(..., description="원본 분석 요청 정보")
    user_profile: Optional[UserProfile] = Field(None, description="사용자 프로필")
    
    # 에이전트 분석 결과 - Dict 형태로 표준화
    company_culture_result: Optional[Dict[str, Any]] = Field(None, description="기업문화 분석 결과")
    work_life_balance_result: Optional[Dict[str, Any]] = Field(None, description="워라밸 분석 결과") 
    management_result: Optional[Dict[str, Any]] = Field(None, description="경영진 분석 결과")
    salary_benefits_result: Optional[Dict[str, Any]] = Field(None, description="연봉/복지 분석 결과")
    career_growth_result: Optional[Dict[str, Any]] = Field(None, description="커리어 성장 분석 결과")
    
    # 최종 종합 결과
    comprehensive_result: Optional[Dict[str, Any]] = Field(None, description="종합 분석 결과")
    
    # 워크플로우 제어
    current_stage: str = Field("initialized", description="현재 실행 단계")
    completed_stages: List[str] = Field(default_factory=list, description="완료된 단계들")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="진행률")
    
    # 에러 및 로깅 - LangGraph 병렬 처리 안전
    errors: Annotated[Sequence[str], operator.add] = Field(default_factory=list)
    warnings: Annotated[Sequence[str], operator.add] = Field(default_factory=list)
    debug_logs: Annotated[Sequence[str], operator.add] = Field(default_factory=list)
    
    # 에이전트 실행 정보
    agent_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="에이전트별 상세 출력")
    stage_timings: Dict[str, float] = Field(default_factory=dict, description="단계별 실행 시간")
    
    # 워크플로우 메타데이터
    workflow_id: str = Field(..., description="워크플로우 고유 ID")
    started_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    # 설정 플래그
    skip_stages: List[str] = Field(default_factory=list, description="건너뛸 단계들")
    debug_mode: bool = Field(False, description="디버그 모드")
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
    
    # 에러 및 로깅 메서드
    def add_error(self, error: str, stage: Optional[str] = None) -> None:
        """에러 추가"""
        error_msg = f"[{stage or self.current_stage}] {error}"
        self.errors.append(error_msg)
        self.updated_at = datetime.now()
    
    def add_warning(self, warning: str, stage: Optional[str] = None) -> None:
        """경고 추가"""
        warning_msg = f"[{stage or self.current_stage}] {warning}"
        self.warnings.append(warning_msg)
        self.updated_at = datetime.now()
    
    def add_debug_log(self, message: str, stage: Optional[str] = None) -> None:
        """디버그 로그 추가"""
        if self.debug_mode:
            log_msg = f"[{stage or self.current_stage}] {message}"
            self.debug_logs.append(log_msg)
    
    # 진행 상태 관리
    def update_stage(self, new_stage: str) -> None:
        """현재 단계 업데이트"""
        if self.current_stage != "initialized":
            self.completed_stages.append(self.current_stage)
        self.current_stage = new_stage
        self.updated_at = datetime.now()
    
    def update_progress(self, progress: float) -> None:
        """진행률 업데이트"""
        self.progress = max(0.0, min(1.0, progress))
        self.updated_at = datetime.now()
    
    def record_stage_timing(self, stage: str, duration: float) -> None:
        """단계별 실행 시간 기록"""
        self.stage_timings[stage] = duration
    
    # 에이전트 결과 저장
    def store_agent_result(self, agent_type: str, result: Dict[str, Any]) -> None:
        """에이전트 결과 저장"""
        if agent_type == "company_culture":
            self.company_culture_result = result
        elif agent_type == "work_life_balance":
            self.work_life_balance_result = result
        elif agent_type == "management":
            self.management_result = result
        elif agent_type == "salary_benefits":
            self.salary_benefits_result = result
        elif agent_type == "career_growth":
            self.career_growth_result = result
        
        self.add_debug_log(f"Stored result for {agent_type} agent")
    
    def store_agent_output(self, agent_name: str, output: Dict[str, Any]) -> None:
        """에이전트 상세 출력 저장"""
        self.agent_outputs[agent_name] = output
        self.add_debug_log(f"Stored output from {agent_name}")
    
    # 상태 확인 메서드
    def has_errors(self) -> bool:
        """에러 존재 여부 확인"""
        return len(self.errors) > 0
    
    def is_stage_completed(self, stage: str) -> bool:
        """특정 단계 완료 여부 확인"""
        return stage in self.completed_stages
    
    def should_skip_stage(self, stage: str) -> bool:
        """단계 건너뛰기 여부 확인"""
        return stage in self.skip_stages
    
    # 결과 조회 메서드
    def get_all_results(self) -> Dict[str, Any]:
        """모든 분석 결과 반환"""
        results = {}
        
        if self.company_culture_result:
            results["company_culture"] = self.company_culture_result
        if self.work_life_balance_result:
            results["work_life_balance"] = self.work_life_balance_result
        if self.management_result:
            results["management"] = self.management_result
        if self.salary_benefits_result:
            results["salary_benefits"] = self.salary_benefits_result
        if self.career_growth_result:
            results["career_growth"] = self.career_growth_result
        if self.comprehensive_result:
            results["comprehensive"] = self.comprehensive_result
            
        return results
    
    def get_completed_analyses(self) -> List[str]:
        """완료된 분석 타입 목록 반환"""
        completed = []
        if self.company_culture_result:
            completed.append("company_culture")
        if self.work_life_balance_result:
            completed.append("work_life_balance")
        if self.management_result:
            completed.append("management")
        if self.salary_benefits_result:
            completed.append("salary_benefits")
        if self.career_growth_result:
            completed.append("career_growth")
        return completed
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보 반환"""
        return {
            "total_execution_time": sum(self.stage_timings.values()),
            "stage_timings": self.stage_timings,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "completed_analyses": self.get_completed_analyses(),
            "progress": self.progress
        }


class WorkflowConfig(BaseModel):
    """워크플로우 실행 설정"""
    
    # 실행 설정
    max_execution_time: int = Field(120, description="최대 실행 시간(초)")
    enable_parallel_execution: bool = Field(True, description="병렬 실행 활성화")
    max_retries: int = Field(3, description="최대 재시도 횟수")
    
    # 품질 설정
    min_confidence_threshold: float = Field(0.6, description="최소 신뢰도 임계값")
    
    # 성능 설정
    enable_caching: bool = Field(True, description="결과 캐싱 활성화")
    cache_ttl: int = Field(3600, description="캐시 TTL(초)")
    
    # 디버그 설정
    enable_debug_logging: bool = Field(False, description="디버그 로깅 활성화")
    
    class Config:
        schema_extra = {
            "example": {
                "max_execution_time": 120,
                "enable_parallel_execution": True,
                "min_confidence_threshold": 0.6,
                "enable_caching": True,
                "enable_debug_logging": False
            }
        }