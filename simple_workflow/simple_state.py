"""
🎯 LangGraph 상태 관리 - 초보자용 간단 버전

상태(State)는 워크플로우 전체에서 공유되는 데이터 저장소입니다.
각 노드가 이 상태를 받아서 수정하고 다음 노드로 전달합니다.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SimpleState(BaseModel):
    """
    🏗️ 간단한 워크플로우 상태 클래스
    
    실제 프로젝트의 WorkflowState는 복잡하지만, 
    여기서는 핵심 개념만 보여드립니다.
    
    📝 상태에 포함되는 정보:
    1. 사용자 입력 (회사 이름)
    2. 처리 결과들 (각 단계별 결과)
    3. 현재 상태 (어느 단계까지 완료했는지)
    """
    
    # 📥 사용자 입력
    company_name: str = Field("", description="분석할 회사 이름")
    
    # 📊 각 단계별 결과 (None이면 아직 처리 안됨)
    culture_score: Optional[int] = Field(None, description="문화 점수 (1-10)")
    salary_info: Optional[str] = Field(None, description="연봉 정보")
    growth_score: Optional[int] = Field(None, description="성장 점수 (1-10)")
    
    # 📈 최종 결과
    final_report: Optional[str] = Field(None, description="최종 분석 보고서")
    
    # 🎛️ 진행 상황 추적
    current_step: str = Field("시작", description="현재 진행 단계")
    completed_steps: List[str] = Field(default_factory=list, description="완료된 단계들")
    
    # 📝 로그 (간단한 메시지들)
    logs: List[str] = Field(default_factory=list, description="처리 과정 로그")
    
    def add_log(self, message: str):
        """로그 메시지 추가"""
        self.logs.append(f"[{self.current_step}] {message}")
    
    def move_to_step(self, step_name: str):
        """다음 단계로 이동"""
        if self.current_step != "시작":
            self.completed_steps.append(self.current_step)
        self.current_step = step_name
        self.add_log(f"{step_name} 단계 시작")
    
    def is_completed(self) -> bool:
        """모든 단계가 완료되었는지 확인"""
        return self.final_report is not None


# 💡 사용 예시
if __name__ == "__main__":
    # 상태 생성
    state = SimpleState(company_name="구글")
    
    # 상태 변경
    state.move_to_step("문화 분석")
    state.culture_score = 9
    
    state.move_to_step("연봉 분석")
    state.salary_info = "연봉 평균 1억원"
    
    print("현재 상태:")
    print(f"회사: {state.company_name}")
    print(f"현재 단계: {state.current_step}")
    print(f"완료된 단계: {state.completed_steps}")
    print(f"문화 점수: {state.culture_score}")
    print(f"로그: {state.logs}")