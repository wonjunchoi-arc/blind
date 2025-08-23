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

💡 커스터마이징 포인트:
- 새로운 분석 타입 추가하려면: 새 Report 필드와 해당 노드 추가
- 진행률 계산 방식 변경하려면: update_progress 메서드 수정
- 에러 처리 방식 변경하려면: add_error, add_warning 메서드 수정
"""

import operator
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Sequence
from pydantic import Field

from ..models.base import BaseModel
from ..models.analysis import (
    AnalysisRequest, CultureReport, CompensationReport, 
    GrowthReport, CareerReport, ComprehensiveReport
)
from ..models.user import UserProfile


class WorkflowState(BaseModel):
    """
    🎯 LangGraph 워크플로우의 중앙 상태 클래스
    
    모든 노드가 공유하는 상태 객체입니다. 각 노드는 이 객체를 받아서 수정하고 반환합니다.
    
    🔄 LangGraph 상태 전이 패턴:
    Node1(state) -> modified_state -> Node2(modified_state) -> ...
    
    📝 커스터마이징 가이드:
    1. 새 필드 추가하기:
       new_field: Optional[YourType] = Field(None, description="설명")
    
    2. 리스트 필드 추가하기 (여러 노드에서 안전하게 추가):
       new_list: Annotated[Sequence[YourType], operator.add] = Field(default_factory=list)
    
    3. 딕셔너리 필드 추가하기:
       new_dict: Dict[str, Any] = Field(default_factory=dict)
    
    ⚠️ 중요: Annotated[Sequence, operator.add]는 여러 노드에서 동시에 리스트에 추가할 때 사용
    """
    
    # 📥 입력 파라미터 - 워크플로우 시작 시 설정되는 필수 정보
    request: AnalysisRequest = Field(..., description="원본 분석 요청 정보")
    user_profile: Optional[UserProfile] = Field(None, description="개인화를 위한 사용자 프로필")
    
    # 📊 분석 결과 - 각각의 전문 에이전트에 의해 채워짐
    # 💡 커스터마이징: 새로운 분석 타입을 추가하려면 여기에 새 Report 필드 추가
    # 예시: risk_report: Optional[RiskReport] = Field(None, description="리스크 분석 결과")
    culture_report: Optional[CultureReport] = Field(None, description="기업 문화 분석 결과")
    compensation_report: Optional[CompensationReport] = Field(None, description="보상 체계 분석 결과") 
    growth_report: Optional[GrowthReport] = Field(None, description="성장성 분석 결과")
    career_report: Optional[CareerReport] = Field(None, description="커리어 경로 분석 결과")
    final_report: Optional[ComprehensiveReport] = Field(None, description="최종 종합 리포트")
    
    # 🎛️ 워크플로우 제어 - LangGraph 실행 상태 관리
    # 💡 커스터마이징: 진행률 계산을 변경하려면 update_progress() 메서드 수정
    current_stage: str = Field("initialized", description="현재 실행 중인 노드 이름")
    completed_stages: List[str] = Field(default_factory=list, description="완료된 워크플로우 단계들")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="전체 진행률 (0.0-1.0)")
    
    # 🚨 에러 처리 및 로깅 - operator.add 패턴으로 여러 노드에서 안전하게 추가
    # ⚠️ 중요: Annotated[Sequence[str], operator.add]는 병렬 노드에서도 안전하게 리스트에 추가
    # 💡 커스터마이징: 새로운 로그 타입을 추가하려면 비슷한 패턴으로 필드 추가
    # 예시: performance_logs: Annotated[Sequence[str], operator.add] = Field(default_factory=list)
    errors: Annotated[Sequence[str], operator.add] = Field(default_factory=list)
    warnings: Annotated[Sequence[str], operator.add] = Field(default_factory=list)
    debug_logs: Annotated[Sequence[str], operator.add] = Field(default_factory=list)
    
    # Data and cache management
    retrieved_documents: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="RAG retrieved documents by category"
    )
    external_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="External data from MCP services"
    )
    agent_outputs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Raw outputs from individual agents"
    )
    
    # Performance metrics
    stage_timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Execution time for each stage in seconds"
    )
    api_call_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="API call counts by service"
    )
    token_usage: Dict[str, int] = Field(
        default_factory=dict,
        description="Token usage by model/service"
    )
    
    # Workflow metadata
    workflow_id: str = Field(..., description="Unique workflow execution identifier")
    started_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    
    # Configuration and flags
    skip_stages: List[str] = Field(default_factory=list, description="Stages to skip")
    force_refresh: bool = Field(False, description="Force refresh of cached data")
    debug_mode: bool = Field(False, description="Enable debug mode")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True
    
    def add_error(self, error: str, stage: Optional[str] = None) -> None:
        """Add an error to the workflow state."""
        error_msg = f"[{stage or self.current_stage}] {error}"
        # Note: Direct list modification works with Annotated[Sequence, operator.add]
        self.errors.append(error_msg)
        self.updated_at = datetime.now()
    
    def add_warning(self, warning: str, stage: Optional[str] = None) -> None:
        """Add a warning to the workflow state."""
        warning_msg = f"[{stage or self.current_stage}] {warning}"
        self.warnings.append(warning_msg)
        self.updated_at = datetime.now()
    
    def add_debug_log(self, message: str, stage: Optional[str] = None) -> None:
        """Add a debug log entry."""
        if self.debug_mode:
            log_msg = f"[{stage or self.current_stage}] {message}"
            self.debug_logs.append(log_msg)
    
    def update_stage(self, new_stage: str) -> None:
        """Update the current workflow stage."""
        if self.current_stage != "initialized":
            self.completed_stages.append(self.current_stage)
        self.current_stage = new_stage
        self.updated_at = datetime.now()
    
    def update_progress(self, progress: float) -> None:
        """Update the workflow progress."""
        self.progress = max(0.0, min(1.0, progress))
        self.updated_at = datetime.now()
    
    def record_stage_timing(self, stage: str, duration: float) -> None:
        """Record the execution time for a workflow stage."""
        self.stage_timings[stage] = duration
    
    def increment_api_calls(self, service: str, count: int = 1) -> None:
        """Increment API call count for a service."""
        self.api_call_counts[service] = self.api_call_counts.get(service, 0) + count
    
    def add_token_usage(self, model: str, tokens: int) -> None:
        """Add token usage for a model."""
        self.token_usage[model] = self.token_usage.get(model, 0) + tokens
    
    def store_retrieved_documents(self, category: str, documents: List[Dict[str, Any]]) -> None:
        """Store retrieved documents for a category."""
        self.retrieved_documents[category] = documents
        self.add_debug_log(f"Stored {len(documents)} documents for category: {category}")
    
    def store_external_data(self, source: str, data: Any) -> None:
        """Store external data from MCP services."""
        self.external_data[source] = data
        self.add_debug_log(f"Stored external data from source: {source}")
    
    def store_agent_output(self, agent_name: str, output: Dict[str, Any]) -> None:
        """Store raw output from an agent."""
        self.agent_outputs[agent_name] = output
        self.add_debug_log(f"Stored output from agent: {agent_name}")
    
    def has_errors(self) -> bool:
        """Check if the workflow has any errors."""
        return len(self.errors) > 0
    
    def get_completion_estimate(self) -> Optional[datetime]:
        """Get estimated completion time based on current progress."""
        if self.progress > 0 and self.started_at:
            elapsed = (datetime.now() - self.started_at).total_seconds()
            estimated_total = elapsed / self.progress
            return self.started_at + datetime.timedelta(seconds=estimated_total)
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of workflow performance metrics."""
        total_execution_time = sum(self.stage_timings.values())
        total_api_calls = sum(self.api_call_counts.values())
        total_tokens = sum(self.token_usage.values())
        
        return {
            "total_execution_time": total_execution_time,
            "total_api_calls": total_api_calls,
            "total_tokens": total_tokens,
            "stage_timings": self.stage_timings,
            "api_calls_by_service": self.api_call_counts,
            "tokens_by_model": self.token_usage,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }
    
    def is_stage_completed(self, stage: str) -> bool:
        """Check if a specific stage has been completed."""
        return stage in self.completed_stages
    
    def should_skip_stage(self, stage: str) -> bool:
        """Check if a stage should be skipped."""
        return stage in self.skip_stages
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """Get all available analysis results."""
        results = {}
        
        if self.culture_report:
            results["culture"] = self.culture_report.dict()
        if self.compensation_report:
            results["compensation"] = self.compensation_report.dict()
        if self.growth_report:
            results["growth"] = self.growth_report.dict()
        if self.career_report:
            results["career"] = self.career_report.dict()
        if self.final_report:
            results["comprehensive"] = self.final_report.dict()
        
        return results
    
    def validate_state(self) -> bool:
        """Validate the current workflow state."""
        # Check required fields
        if not self.request:
            self.add_error("Missing analysis request")
            return False
        
        if not self.workflow_id:
            self.add_error("Missing workflow ID")
            return False
        
        # Check progress consistency
        if self.progress < 0 or self.progress > 1:
            self.add_error(f"Invalid progress value: {self.progress}")
            return False
        
        # Check if final report exists when progress is complete
        if self.progress >= 1.0 and not self.final_report:
            self.add_warning("Progress indicates completion but no final report available")
        
        return not self.has_errors()


class WorkflowConfig(BaseModel):
    """Configuration for workflow execution."""
    
    # Execution settings
    max_execution_time: int = Field(120, description="Maximum execution time in seconds")
    enable_parallel_execution: bool = Field(True, description="Enable parallel agent execution")
    max_retries: int = Field(3, description="Maximum retries for failed stages")
    
    # Quality settings
    min_confidence_threshold: float = Field(0.7, description="Minimum confidence threshold")
    require_all_analyses: bool = Field(False, description="Require all analysis types to complete")
    
    # Performance settings
    enable_caching: bool = Field(True, description="Enable result caching")
    cache_ttl: int = Field(3600, description="Cache TTL in seconds")
    
    # Debug settings
    enable_debug_logging: bool = Field(False, description="Enable debug logging")
    save_intermediate_results: bool = Field(False, description="Save intermediate results")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "max_execution_time": 120,
                "enable_parallel_execution": True,
                "min_confidence_threshold": 0.7,
                "enable_caching": True
            }
        }