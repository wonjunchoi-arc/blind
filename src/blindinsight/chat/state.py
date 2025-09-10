"""
Chat State Management - Modern Supervisor 상태 관리
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any, TypedDict, Annotated
from langchain_core.messages import AnyMessage

# -----------------------------
# SupervisorState 정의
# -----------------------------

class SupervisorState(TypedDict):
    """Supervisor 워크플로우 상태"""
    messages: Annotated[list[AnyMessage], "대화 메시지들"]
    current_agent: str
    retry_count: int
    quality_score: float
    selected_expert: Optional[str]
    supervisor_prompt: Optional[str]  # handoff 시 저장되는 간결 task_description
    # 품질평가 관련 필드 추가
    should_retry: bool  # 품질평가 결과 재시도 여부
    improvement_suggestions: Optional[List[str]]  # 개선 제안사항
    evaluation_result: Optional[Dict[str, Any]]  # 품질평가 전체 결과
    max_retry_count: int  # 최대 재시도 횟수
    last_worker_agent: Optional[str]  # 마지막으로 실행된 워커 에이전트

# -----------------------------
# 유틸리티 함수들
# -----------------------------

MAX_TASK_LEN = 160

def sanitize_task_description(text: Optional[str]) -> str:
    """간결한 task_description 보장(1~2문장, 160자)
    - 줄바꿈/마크다운 기호 제거
    - 최대 두 문장 클리핑
    - 길이 상한 적용
    """
    import re
    if not text:
        return ""
    t = re.sub(r"[\r\n]+", " ", text)
    t = re.sub(r"[*_`>#>-]+", " ", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    parts = re.split(r"([.!?？！。])", t)
    clipped = "".join(parts[:4]) if parts else t
    if len(clipped) > MAX_TASK_LEN:
        clipped = clipped[:MAX_TASK_LEN].rstrip()
    return clipped

def create_initial_state(user_question: str) -> SupervisorState:
    """초기 Supervisor 상태 생성"""
    from langchain_core.messages import HumanMessage
    
    return {
        "messages": [HumanMessage(content=user_question)],
        "current_agent": "supervisor",
        "retry_count": 0,
        "quality_score": 0.0,
        "selected_expert": None,
        "supervisor_prompt": None,
        # 품질평가 관련 초기값
        "should_retry": False,
        "improvement_suggestions": None,
        "evaluation_result": None,
        "max_retry_count": 2,
        "last_worker_agent": None,
    }

# -----------------------------
# 메시지 추출 함수들
# -----------------------------

def get_user_question_from_state(state: SupervisorState) -> Optional[str]:
    """상태에서 사용자 질문 추출"""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and msg.content and not str(msg.content).startswith("handoff -> "):
            if "HumanMessage" in str(type(msg)):
                return msg.content
    return None

def get_last_ai_response_from_state(state: SupervisorState) -> Optional[str]:
    """상태에서 마지막 AI 응답 추출 (품질평가 제외)"""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and msg.content and not str(msg.content).startswith("handoff -> "):
            content = str(msg.content).strip()
            # JSON 형식의 품질평가 메시지 제외
            if not (content.startswith('{') and '"evaluation_type"' in content):
                if "AIMessage" in str(type(msg)):
                    return msg.content
    return None

def get_last_user_message_from_state(state: SupervisorState) -> Optional[str]:
    """상태에서 마지막 사용자 메시지 추출"""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "content") and msg.content and not str(msg.content).startswith("handoff -> "):
            return msg.content
    return None

def filter_handoff_messages(messages: List[AnyMessage]) -> List[AnyMessage]:
    """handoff 메시지들 필터링"""
    filtered = []
    for msg in messages:
        if hasattr(msg, "content") and msg.content:
            if not str(msg.content).startswith("handoff -> "):
                filtered.append(msg)
        else:
            filtered.append(msg)
    return filtered

def filter_quality_evaluation_messages(messages: List[AnyMessage]) -> List[AnyMessage]:
    """품질평가 JSON 메시지들 필터링"""
    from langchain_core.messages import AIMessage
    
    filtered = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content.strip()
            # JSON 형식의 품질평가 메시지 제외
            if not (content.startswith('{') and '"evaluation_type"' in content):
                filtered.append(msg)
        else:
            filtered.append(msg)
    return filtered