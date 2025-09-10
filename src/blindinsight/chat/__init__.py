"""
BlindInsight AI Chat System - 리팩토링된 Modern Supervisor 기반

리팩토링 완료:
- state.py: SupervisorState와 상태 관리 유틸리티
- workflow.py: handoff tools와 워크플로우 로직
- modern_supervisor.py: 메인 클래스와 공개 API

LangGraph Supervisor 패턴을 사용한 멀티 에이전트 채팅 시스템
"""

# 메인 클래스들 import
from .modern_supervisor import ModernSupervisorAgent
from .state import SupervisorState, create_initial_state
from .workflow import SupervisorWorkflow, create_handoff_tools

# Modern Supervisor를 기본 SupervisorAgent로 사용
SupervisorAgent = ModernSupervisorAgent

# 주요 클래스들을 직접 import 가능하도록 export
__all__ = [
    # Core Classes
    'ModernSupervisorAgent',
    'SupervisorAgent',  # ModernSupervisorAgent의 별칭
    
    # State Management
    'SupervisorState',
    'create_initial_state',
    
    # Workflow Management
    'SupervisorWorkflow',
    'create_handoff_tools',
]

# 편의성을 위한 기본 인스턴스들
def create_default_supervisor():
    """기본 Modern Supervisor 생성"""
    return ModernSupervisorAgent()

# 빠른 시작을 위한 함수들
async def quick_chat(question: str, session_id: str = "default"):
    """
    빠른 질문-답변 함수
    
    Args:
        question: 사용자 질문
        session_id: 세션 ID
    
    Returns:
        str: AI 답변
    """
    supervisor = create_default_supervisor()
    result = await supervisor.chat(question, session_id)
    return result.get("response", "응답을 생성할 수 없습니다.")

# 버전 정보 
__version__ = "2.1.0-refactored"
__supervisor_type__ = "modern-refactored"
__description__ = "리팩토링된 LangGraph Supervisor 패턴 기반 멀티 에이전트 채팅 시스템"