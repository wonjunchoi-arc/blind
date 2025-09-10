"""
ModernSupervisorAgent - LangGraph StateGraph + Command-based Handoffs 멀티 에이전트 조율자

리팩토링 완료:
- state.py: SupervisorState와 관련 유틸리티 분리
- workflow.py: handoff tools와 워크플로우 로직 분리
- modern_supervisor.py: 메인 클래스와 공개 API만 유지
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

# 분리된 모듈들 import
from .state import SupervisorState, create_initial_state
from .workflow import (
    SupervisorWorkflow, 
    create_handoff_tools, 
    extract_final_response_from_messages,
    get_chat_history_from_checkpointer
)

# 외부 의존(예시): 프로젝트 설정/유저 모델
try:
    from ..models.base import settings  # type: ignore
    from ..models.user import UserProfile  # type: ignore
except Exception:
    class _S:
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    settings = _S()
    class UserProfile: ...

class ModernSupervisorAgent:
    """
    LangGraph StateGraph + Command handoffs 기반 멀티 에이전트 조율자
    - 리팩토링된 구조: state.py, workflow.py 사용
    - Supervisor: 도구 호출을 통해 직접 라우팅
    - Worker: supervisor_prompt(task_description)만 참고해 실행
    """

    def __init__(self):
        self.model = ChatOpenAI(
            model=settings.supervisor_model,
            temperature=0.7,
            openai_api_key=settings.openai_api_key,
            parallel_tool_calls=False,  # ✅ 병렬 도구 호출 금지
        )
        self.checkpointer = MemorySaver()
        self.agents = self._create_specialized_agents()
        self.handoff_tools = create_handoff_tools()
        
        # 워크플로우 관리자 생성
        self.workflow_manager = SupervisorWorkflow(
            agents=self.agents,
            model=self.model,
            handoff_tools=self.handoff_tools,
            checkpointer=self.checkpointer
        )
        self.supervisor_workflow = self.workflow_manager.create_workflow()

    # -------------------------
    # Specialized Agents Creation
    # -------------------------

    def _create_specialized_agents(self) -> Dict[str, Any]:
        """전문화된 에이전트들을 실제 에이전트 인스턴스로 생성"""
        try:
            print(f"[ModernSupervisor] 전문 에이전트 생성 시작...")
            
            from ..agents.salary_benefits_agent import SalaryBenefitsAgent
            from ..agents.company_culture_agent import CompanyCultureAgent
            from ..agents.work_life_balance_agent import WorkLifeBalanceAgent
            from ..agents.management_agent import ManagementAgent
            from ..agents.career_growth_agent import CareerGrowthAgent
            from ..agents.quality_evaluator_agent import QualityEvaluatorAgent

            agents = {
                "salary_benefits_expert": SalaryBenefitsAgent(),
                "company_culture_expert": CompanyCultureAgent(),
                "work_life_balance_expert": WorkLifeBalanceAgent(), 
                "management_expert": ManagementAgent(),
                "career_growth_expert": CareerGrowthAgent(),
                "quality_evaluator": QualityEvaluatorAgent(),
            }
            
            print(f"[ModernSupervisor] 전문 에이전트 생성 완료: {list(agents.keys())}")
            # 인터페이스 점검
            for name, agent in agents.items():
                has_method = hasattr(agent, 'execute_as_supervisor_tool')
                print(f"[ModernSupervisor] {name}: execute_as_supervisor_tool 메서드 존재 = {has_method}")
            
            return agents
        except Exception as e:
            print(f"[ModernSupervisor] 에이전트 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return {}

    # -------------------------
    # Public APIs
    # -------------------------
    async def chat(self, user_question: str, session_id: str = "default", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """사용자 질문 처리 및 최종 응답 반환"""
        try:
            initial_state = create_initial_state(user_question)
            config = {"configurable": {"thread_id": session_id}}
            result = await self.supervisor_workflow.ainvoke(initial_state, config=config)

            final_messages = result.get("messages", [])
            response = extract_final_response_from_messages(final_messages)
            
            if response:
                return {
                    "success": True,
                    "response": response,
                    "session_id": session_id,
                    "metadata": {
                        "total_messages": len(final_messages),
                        "workflow_state": "completed",
                        "selected_expert": result.get("selected_expert"),
                        "quality_score": result.get("quality_score", 0.0),
                        "retry_count": result.get("retry_count", 0),
                    },
                }

            return {
                "success": False,
                "response": "죄송합니다. 적절한 답변을 생성할 수 없습니다.",
                "session_id": session_id,
                "error": "No valid response found in workflow result",
            }

        except Exception as e:
            return {
                "success": False,
                "response": "시스템 오류로 인해 응답을 생성할 수 없습니다.",
                "session_id": session_id,
                "error": str(e),
            }

    async def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """채팅 기록 조회 (워크플로우 모듈에 위임)"""
        return await get_chat_history_from_checkpointer(self.checkpointer, session_id)

# -----------------------------
# Quick helpers (예시)
# -----------------------------
async def quick_chat_with_modern_supervisor(question: str, session_id: str = "default") -> str:
    supervisor = ModernSupervisorAgent()
    result = await supervisor.chat(question, session_id)
    return result.get("response", "응답을 생성할 수 없습니다.")

async def test_new_workflow() -> bool:
    print("🚀 Modern Supervisor(Command handoff) 테스트 시작!")
    test_question = "삼성전자 연봉 어때?"
    try:
        supervisor = ModernSupervisorAgent()
        result = await supervisor.chat(test_question)
        print(f"✅ 결과: {result.get('success', False)} | 답변: {result.get('response', '')[:120]}...")
        return True
    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_new_workflow())
