"""
Chat Workflow - Modern Supervisor 워크플로우 관리
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Any, Annotated
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from .state import SupervisorState, sanitize_task_description, get_user_question_from_state

# -----------------------------
# Handoff Tools
# -----------------------------

def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """Command 기반 handoff 도구: 간결한 task_description을 state에 저장한 후 target agent로 이동"""
    name = f"transfer_to_{agent_name}"
    description = description or f"Hand off to {agent_name} with a concise task description."

    @tool(name, description=description)
    def handoff_tool(
        request: str,
        task_description: str = "",
        state: Annotated[MessagesState, InjectedState] = None,
        tool_call_id: Annotated[str, InjectedToolCallId] = "",
    ) -> Command:
        td = sanitize_task_description(task_description or request)
        tool_message = {
            "role": "tool",
            "name": name,
            "tool_call_id": tool_call_id,
            "content": f"handoff -> {agent_name} | task: {td}",
        }
        updated = {**state, "messages": state["messages"] + [tool_message], "supervisor_prompt": td}
        return Command(goto=agent_name, update=updated, graph=Command.PARENT)

    return handoff_tool

def create_handoff_tools():
    """전체 handoff 도구 생성"""
    return [
            create_handoff_tool(
                agent_name="salary_benefits_expert",
                description="급여/연봉/복리후생/보너스/협상 관련 질문을 전문가에게 이전",
            ),
            create_handoff_tool(
                agent_name="company_culture_expert",
                description="회사 문화/조직 분위기/팀워크/사내 정치 관련 질문 이전",
            ),
            create_handoff_tool(
                agent_name="work_life_balance_expert",
                description="근무시간/야근/워라밸/휴가/유연근무 관련 질문 이전",
            ),
            create_handoff_tool(
                agent_name="management_expert",
                description="상사/경영진/리더십/관리 체계 관련 질문 이전",
            ),
            create_handoff_tool(
                agent_name="career_growth_expert",
                description="승진/교육/성장 기회/커리어 발전 관련 질문 이전",
            ),
            # NEW: 품질 평가 에이전트로 핸드오프
            create_handoff_tool(
                agent_name="quality_evaluator",
                description="직전 응답을 품질 기준으로 평가하고 개선 제안을 생성",
            ),
    ]

# -----------------------------
# Workflow 관리 클래스
# -----------------------------

class SupervisorWorkflow:
    """Modern Supervisor 워크플로우 관리 클래스"""
    
    def __init__(self, agents: Dict[str, Any], model: ChatOpenAI, handoff_tools: List[Any], checkpointer: MemorySaver):
        self.agents = agents
        self.model = model
        self.handoff_tools = handoff_tools
        self.checkpointer = checkpointer
    
    def create_workflow(self):
        """StateGraph 워크플로우 생성"""
        workflow = StateGraph(SupervisorState)

        # Supervisor 노드: 도구 호출(transfer_to_*) 처리 + 간결한 task_description 생성 + 품질평가 결과 처리 추가
        supervisor_agent = self._create_supervisor_node()

        workflow.add_node("supervisor", supervisor_agent)
        for agent_name in self.agents.keys():
            workflow.add_node(agent_name, self._create_agent_node(agent_name))

        workflow.add_edge(START, "supervisor")
        
        # 워크플로우 순서: worker → quality_evaluator → supervisor
        for agent_name in self.agents.keys():
            if agent_name != "quality_evaluator":  # quality_evaluator 제외
                workflow.add_edge(agent_name, "quality_evaluator")
        
        # quality_evaluator → supervisor 연결
        workflow.add_edge("quality_evaluator", "supervisor")

        # 참고: conditional_edges 없음. 대신 handoff tool의 Command(goto=...)로 동적
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _create_supervisor_node(self):
        """Supervisor 노드 생성 - 품질평가 결과 처리 및 재시도 로직 포함"""
        
        # 기본 supervisor agent
        base_supervisor = create_react_agent(
            model=self.model,
            tools=self.handoff_tools,
            name="supervisor",
            prompt=(
               "당신은 팀 슈퍼바이저입니다. 당신의 임무:\n"
                "1) 초기 요청: 적합한 전문가 선택 + 간결한 작업 설명(task_description, 1~2문장, 160자) 작성\n"
                "2) 품질평가 후: 재시도 여부 판단 + 개선된 작업 설명 생성\n\n"
                "작업 설명 규칙:\n"
                "- 금지: 불릿/체크리스트/장문/메타지시\n"
                "- 포함: 핵심 목표 + 최소 컨텍스트\n\n"
                "중요 규칙:\n"
                "- 한 번에 정확히 하나의 transfer_to_* 도구만 호출하세요. 병렬/복수 호출 금지.\n"
                "반드시 transfer_to_* 도구를 호출하세요."
            ),
        )
        
        async def supervisor_node(state: SupervisorState) -> SupervisorState:
            print(f"[Supervisor] ===== supervisor 노드 시작 =====")
            print(f"[Supervisor] current_agent: {state.get('current_agent')}")
            print(f"[Supervisor] retry_count: {state.get('retry_count', 0)}")
            print(f"[Supervisor] should_retry: {state.get('should_retry', False)}")
            
            # 1. 품질평가 결과 처리 (quality_evaluator로부터 돌아온 경우)
            if state.get("current_agent") == "supervisor" and state.get("selected_expert"):
                evaluation_result = self._parse_quality_evaluation(state)
                if evaluation_result:
                    return self._handle_quality_evaluation(state, evaluation_result, base_supervisor)
            
            # 2. 초기 요청 처리 (사용자 질문을 처음 받은 경우)
            print(f"[Supervisor] 초기 요청 처리 - base_supervisor 호출")
            
            # 일반적 base_supervisor로 전달해 handoff 도구를 선택하도록 함
            # LangGraph 내부에서 Command 반환이 자동 처리
            result = await base_supervisor.ainvoke(state)
            print(f"[Supervisor] base_supervisor 호출 완료")
            return result
        
        return supervisor_node
    
    def _parse_quality_evaluation(self, state: SupervisorState) -> Optional[Dict[str, Any]]:
        """메시지에서 품질평가 결과 파싱"""
        try:
            # 최신 AI 메시지에서 JSON 검색
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content.strip()
                    
                    # JSON 파싱 시도
                    try:
                        if content.startswith('{') and content.endswith('}'):
                            evaluation = json.loads(content)
                            
                            # 유효성 검사 통과
                            if ("evaluation_type" in evaluation and 
                                evaluation.get("evaluation_type") == "quality_assessment" and
                                "should_retry" in evaluation):
                                
                                print(f"[Supervisor] 품질평가 결과 파싱 성공: should_retry={evaluation.get('should_retry')}")
                                return evaluation
                    except json.JSONDecodeError:
                        continue
            
            print(f"[Supervisor] 품질평가 결과를 찾을 수 없음")
            return None
            
        except Exception as e:
            print(f"[Supervisor] 품질평가 파싱 오류: {str(e)}")
            return None
    
    def _handle_quality_evaluation(self, state: SupervisorState, evaluation: Dict[str, Any], base_supervisor) -> SupervisorState:
        """품질평가 결과 처리 및 재시도 로직"""
        should_retry = evaluation.get("should_retry", False)
        retry_count = state.get("retry_count", 0)
        max_retry = state.get("max_retry_count", 2)
        improvement_suggestions = evaluation.get("improvement_suggestions", [])
        
        print(f"[Supervisor] 품질평가 처리 - should_retry: {should_retry}, retry_count: {retry_count}/{max_retry}")
        
        # 재시도 필요 + 재시도 횟수가 남은 경우
        if should_retry and retry_count < max_retry:
            print(f"[Supervisor] 재시도 실행 - 개선사항: {improvement_suggestions}")
            
            # 개선된 프롬프트 생성
            improved_prompt = self._create_improved_prompt(state, improvement_suggestions)
            
            # 재시도 상태 업데이트
            retry_state = {
                **state,
                "retry_count": retry_count + 1,
                "supervisor_prompt": improved_prompt,
                "should_retry": True,
                "improvement_suggestions": improvement_suggestions,
                "evaluation_result": evaluation,
                "current_agent": "supervisor",
            }
            
            # 이전 worker agent로 재전송
            last_worker = state.get("last_worker_agent")
            if last_worker and last_worker in self.agents:
                print(f"[Supervisor] {last_worker}로 재전송")
                
                # 개선된 handoff 메시지 생성
                handoff_msg = f"handoff -> {last_worker} | improved_task: {improved_prompt}"
                tool_message = {
                    "role": "tool", 
                    "name": f"transfer_to_{last_worker}",
                    "tool_call_id": "retry_handoff",
                    "content": handoff_msg,
                }
                
                return {
                    **retry_state,
                    "messages": state["messages"] + [tool_message],
                    "current_agent": last_worker,
                }
        
        # 재시도 불필요 또는 재시도 횟수 초과 - 최종 응답 반환
        print(f"[Supervisor] 워크플로우 완료 - 최종 응답 반환")
        
        # quality_evaluator 메시지만 제거하고 실제 worker 응답은 유지
        from .state import filter_quality_evaluation_messages
        filtered_messages = filter_quality_evaluation_messages(state["messages"])
        
        return {
            **state,
            "messages": filtered_messages,
            "current_agent": "supervisor", 
            "should_retry": False,
            "evaluation_result": evaluation,
        }
    
    def _create_improved_prompt(self, state: SupervisorState, suggestions: List[str]) -> str:
        """개선사항을 반영한 프롬프트 생성"""
        
        # 원본 사용자 질문 추출
        user_question = get_user_question_from_state(state)
        
        # 개선사항을 반영한 새 프롬프트
        if suggestions:
            main_suggestion = suggestions[0] if suggestions else ""
            improved_prompt = f"{user_question} (개선요구: {main_suggestion})"
        else:
            improved_prompt = f"{user_question} (더 상세하고 구체적인 답변 필요)"
        
        # 길이 제한 적용
        return sanitize_task_description(improved_prompt)

    def _create_agent_node(self, agent_name: str):
        """개별 에이전트 노드 생성"""
        async def agent_node(state: SupervisorState) -> SupervisorState:
            try:
                print(f"[SupervisorWorkflow] 에이전트 노드 '{agent_name}' 실행 시작")
                agent = self.agents.get(agent_name)
                if not agent:
                    return {
                        **state,
                        "messages": state["messages"] + [AIMessage(content=f"에이전트 '{agent_name}'을 찾을 수 없습니다.")],
                        "current_agent": "supervisor",
                    }

                # 필요한 정보 추출(사용자 질문)
                from .state import get_last_user_message_from_state, get_last_ai_response_from_state
                
                user_message = get_last_user_message_from_state(state)
                if not user_message:
                    return {
                        **state,
                        "messages": state["messages"] + [AIMessage(content="사용자 질문을 찾을 수 없습니다.")],
                        "current_agent": "supervisor",
                    }

                supervisor_prompt = state.get("supervisor_prompt")
                # 이전 AI 응답(재시도 에이전트 호출시)을 컨텍스트로 활용할 수도 있음
                last_ai_response = get_last_ai_response_from_state(state)

                response_content = await agent.execute_as_supervisor_tool(
                    user_question=user_message,
                    supervisor_prompt=supervisor_prompt,
                    context={"agent_name": agent_name, "last_ai_response": last_ai_response},
                )

                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content=response_content)],
                    "current_agent": "quality_evaluator" if agent_name != "quality_evaluator" else "supervisor",
                    "quality_score": state.get("quality_score", 0.0),
                    "selected_expert": agent_name,
                    "last_worker_agent": agent_name if agent_name != "quality_evaluator" else state.get("last_worker_agent"),
                }

            except Exception as e:
                return {
                    **state,
                    "messages": state["messages"] + [AIMessage(content=f"에이전트 실행 오류: {str(e)}")],
                    "current_agent": "supervisor",
                }

        return agent_node

# -----------------------------
# 유틸리티 함수들
# -----------------------------

def extract_final_response_from_messages(messages: List[AnyMessage]) -> Optional[str]:
    """메시지에서 최종 응답 추출"""
    if not messages:
        return None
        
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not str(msg.content).startswith("handoff -> "):
            return msg.content
    return None

async def get_chat_history_from_checkpointer(checkpointer: MemorySaver, session_id: str) -> List[Dict[str, Any]]:
    """체크포인터에서 채팅 기록 추출"""
    try:
        config = {"configurable": {"thread_id": session_id}}
        checkpoints = list(checkpointer.list(config))
        history: List[Dict[str, Any]] = []
        for cp in sorted(checkpoints, key=lambda x: x.ts):
            channel_values = cp.checkpoint.get("channel_values", {})
            messages = channel_values.get("messages", [])
            for msg in messages:
                if hasattr(msg, "content") and msg.content:
                    if not str(msg.content).startswith("handoff -> "):
                        history.append({
                            "role": "user" if "HumanMessage" in str(type(msg)) else "assistant",
                            "content": msg.content,
                            "timestamp": cp.ts,
                        })
        return history
    except Exception:
        return []