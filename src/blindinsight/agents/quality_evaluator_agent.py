"""
BlindInsight AI - 품질 평가 전문 에이전트

🎯 전문 영역:
- AI 에이전트 응답의 품질 평가 및 개선 제안
- 사용자 질문과 응답 간의 적합성 분석
- 재시도 여부 판단 및 개선 방향 제시

🔍 핵심 기능:
- 정확성, 완성도, 유용성, 구체성, 균형성 평가
- JSON 형태의 구조화된 평가 결과 제공
- LLM 기반 순수 품질 판단 (폴백 없음)

📊 평가 기준:
- 9-10점: excellent (재시도 불필요)
- 7-8점: good (재시도 불필요)  
- 5-6점: fair (재시도 고려)
- 0-4점: poor (재시도 필요)
"""

from typing import Any, Dict, List, Optional
import json
from ..models.user import UserProfile
from .base import BaseAgent, AgentResult, AgentConfig


class QualityEvaluatorAgent(BaseAgent):
    """
    품질 평가 전문 에이전트
    
    🎯 주요 기능:
    - AI 에이전트 응답의 종합적 품질 평가
    - 사용자 질문과 응답 간의 적합성 분석
    - 개선 방향 제시 및 재시도 여부 판단
    - 순수 LLM 기반 평가 (RAG 검색 없음)
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """품질 평가 에이전트 초기화"""
        super().__init__(name="quality_evaluator_agent", config=config)
        
        # 품질 평가 에이전트는 RAG 검색을 사용하지 않음
        self.target_collections = []
    
    async def execute(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> AgentResult:
        """
        품질 평가 실행
        
        Args:
            query: 사용자의 원본 질문
            context: 평가 대상 응답 및 에이전트 정보
            user_profile: 사용자 프로필 (선택사항)
            
        Returns:
            AgentResult: 품질 평가 결과
        """
        # 에러 처리와 함께 실행
        return await self._execute_with_error_handling(
            self._evaluate_response_quality,
            query,
            context,
            user_profile
        )
    
    async def execute_as_supervisor_tool(
        self,
        user_question: str,
        supervisor_prompt: Optional[str],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Supervisor 도구로 실행 - BaseAgent 메서드 오버라이드
        품질 평가 전용 에이전트는 RAG 검색 없이 품질 평가만 수행
        
        Args:
            user_question: 사용자의 원본 질문
            supervisor_prompt: Supervisor가 제공한 작업 설명 (사용하지 않음)
            context: 평가 대상 응답 및 에이전트 정보
            
        Returns:
            str: JSON 형태의 품질 평가 결과 (supervisor가 파싱 가능한 형태)
        """
        try:
            print(f"[{self.name}] ===== execute_as_supervisor_tool 시작 =====")
            print(f"[{self.name}] 사용자 질문: {user_question}")
            print(f"[{self.name}] Supervisor 프롬프트 존재: {bool(supervisor_prompt)}")
            print(f"[{self.name}] 컨텍스트: {context}")
            
            # RAG 검색 없이 바로 품질 평가 수행
            result = await self._evaluate_response_quality(
                user_question,
                context,
                None
            )
            
            # 결과 포맷팅 - JSON 구조로 반환 (supervisor 파싱용)
            if isinstance(result, dict) and 'evaluation' in result:
                evaluation = result['evaluation']
                
                # Supervisor가 파싱할 수 있는 JSON 형태로 반환
                supervisor_result = {
                    "evaluation_type": "quality_assessment",
                    "should_retry": evaluation.get('should_retry', False),
                    "overall_score": evaluation.get('overall_score', 0),
                    "quality_level": evaluation.get('quality_level', 'fair'),
                    "improvement_suggestions": evaluation.get('improvement_suggestions', []),
                    "detailed_scores": {
                        "accuracy_score": evaluation.get('accuracy_score', 0),
                        "completeness_score": evaluation.get('completeness_score', 0),
                        "usefulness_score": evaluation.get('usefulness_score', 0),
                        "specificity_score": evaluation.get('specificity_score', 0),
                        "balance_score": evaluation.get('balance_score', 0)
                    },
                    "original_response": context.get("last_ai_response", "")[:200] + "..." if context and context.get("last_ai_response", "") and len(context.get("last_ai_response", "")) > 200 else context.get("last_ai_response", "") if context else "",
                    "agent_type": context.get("agent_name", "unknown") if context else "unknown",
                    "confidence_score": result.get('confidence_score', 0.9)
                }
                
                # JSON 문자열로 직렬화하여 반환
                import json
                json_result = json.dumps(supervisor_result, ensure_ascii=False, indent=2)
                
                print(f"[{self.name}] 품질 평가 완료: {evaluation.get('overall_score', 0)}/10 ({evaluation.get('quality_level', 'fair')}), should_retry: {evaluation.get('should_retry', False)}")
                
                return json_result
            
            # 오류 케이스 - JSON 형태로 반환
            elif isinstance(result, dict) and 'error' in result:
                error_result = {
                    "evaluation_type": "quality_assessment",
                    "should_retry": False,
                    "overall_score": 0,
                    "quality_level": "unknown",
                    "improvement_suggestions": [],
                    "error": result['error'],
                    "confidence_score": 0.0
                }
                return json.dumps(error_result, ensure_ascii=False, indent=2)
            
            # 예외 케이스 - JSON 형태로 반환
            else:
                fallback_result = {
                    "evaluation_type": "quality_assessment",
                    "should_retry": False,
                    "overall_score": 0,
                    "quality_level": "unknown",
                    "improvement_suggestions": [],
                    "error": f"예상치 못한 결과 형태: {str(result)}",
                    "confidence_score": 0.0
                }
                return json.dumps(fallback_result, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"[{self.name}] 품질 평가 중 오류 발생: {str(e)}")
            
            # 오류 시에도 JSON 형태로 반환
            error_result = {
                "evaluation_type": "quality_assessment", 
                "should_retry": False,
                "overall_score": 0,
                "quality_level": "error",
                "improvement_suggestions": [],
                "error": f"품질 평가 중 오류가 발생했습니다: {str(e)}",
                "confidence_score": 0.0
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)
    
    async def _evaluate_response_quality(
        self,
        user_question: str,
        context: Optional[Dict[str, Any]] = None,
        user_profile: Optional[UserProfile] = None
    ) -> Dict[str, Any]:
        """
        AI 응답 품질 평가 메인 로직
        
        Args:
            user_question: 사용자의 원본 질문
            context: 평가 대상 응답 및 에이전트 정보
            user_profile: 사용자 프로필 (사용하지 않음)
            
        Returns:
            Dict[str, Any]: 품질 평가 결과
        """
        if not context:
            return {
                "error": "평가할 컨텍스트가 제공되지 않았습니다.",
                "confidence_score": 0.0
            }
        
        # 평가 대상 응답과 에이전트 정보 추출
        response_content = context.get("last_ai_response", "")
        agent_type = context.get("agent_name", "unknown")
        
        if not response_content:
            return {
                "error": "평가할 응답이 제공되지 않았습니다.",
                "confidence_score": 0.0
            }
        
        # 품질 평가 프롬프트 생성
        evaluation_prompt = self._create_evaluation_prompt(
            user_question, 
            response_content, 
            agent_type
        )
        
        try:
            # LLM을 사용하여 품질 평가 수행
            evaluation_response = await self.generate_response(
                prompt=evaluation_prompt,
                context_documents=[],  # RAG 검색 사용하지 않음
                temperature=0.0,  # 일관된 평가를 위해 낮은 temperature 사용
                max_tokens=1500
            )
            
            if not evaluation_response:
                return {
                    "error": "품질 평가 응답을 받지 못했습니다.",
                    "confidence_score": 0.0
                }
            
            # JSON 파싱 시도
            try:
                evaluation_json = json.loads(evaluation_response.strip())
                
                # 필수 필드 검증
                required_fields = [
                    "accuracy_score", "completeness_score", "usefulness_score",
                    "specificity_score", "balance_score", "overall_score",
                    "quality_level", "improvement_suggestions", "should_retry"
                ]
                
                for field in required_fields:
                    if field not in evaluation_json:
                        evaluation_json[field] = self._get_default_value(field)
                
                # 점수 유효성 검증 (0-10 범위)
                score_fields = [
                    "accuracy_score", "completeness_score", "usefulness_score",
                    "specificity_score", "balance_score", "overall_score"
                ]
                
                for field in score_fields:
                    score = evaluation_json.get(field, 0)
                    if not isinstance(score, (int, float)) or score < 0 or score > 10:
                        evaluation_json[field] = 5  # 기본값으로 설정
                
                return {
                    "user_question": user_question,
                    "evaluated_response": response_content[:500] + "..." if len(response_content) > 500 else response_content,
                    "agent_type": agent_type,
                    "evaluation": evaluation_json,
                    "analysis_timestamp": context.get("timestamp"),
                    "confidence_score": 0.9  # 품질 평가는 높은 신뢰도
                }
                
            except json.JSONDecodeError as je:
                print(f"품질 평가 JSON 파싱 실패: {str(je)}")
                print(f"원본 응답: {evaluation_response}")
                
                # JSON 파싱 실패 시 기본 평가 결과 반환
                return {
                    "user_question": user_question,
                    "evaluated_response": response_content[:500] + "..." if len(response_content) > 500 else response_content,
                    "agent_type": agent_type,
                    "evaluation": {
                        "accuracy_score": 5,
                        "completeness_score": 5,
                        "usefulness_score": 5,
                        "specificity_score": 5,
                        "balance_score": 5,
                        "overall_score": 5,
                        "quality_level": "fair",
                        "improvement_suggestions": ["JSON 형태로 응답을 생성하지 못했습니다."],
                        "should_retry": False
                    },
                    "parsing_error": str(je),
                    "confidence_score": 0.3
                }
        
        except Exception as e:
            print(f"품질 평가 중 오류 발생: {str(e)}")
            return {
                "error": f"품질 평가 중 오류 발생: {str(e)}",
                "confidence_score": 0.0
            }
    
    def _create_evaluation_prompt(
        self, 
        user_question: str, 
        response_content: str, 
        agent_type: str
    ) -> str:
        """품질 평가 프롬프트 생성"""
        
        return f"""
당신은 AI 응답의 품질을 평가하는 전문가입니다.
다음은 사용자의 질문과 AI 에이전트의 답변입니다.
이 답변의 품질을 객관적이고 공정하게 평가해주세요.

**사용자 질문:** {user_question}

**에이전트 유형:** {agent_type}

**AI 답변:** {response_content}

다음 5가지 기준으로 각각 0-10점으로 평가해주세요:

1. **정확성 (Accuracy)**: 답변이 사용자 질문에 적절하게 대답하는가?
   - 질문의 핵심을 이해하고 올바른 정보를 제공하는지
   - 사실관계의 정확성 및 오해를 불러일으킬 소지가 없는지

2. **완성도 (Completeness)**: 답변이 충분히 상세하고 완전한가?
   - 질문에 대한 충분한 정보와 설명이 포함되어 있는지
   - 추가로 알아야 할 중요한 정보가 빠지지 않았는지

3. **유용성 (Usefulness)**: 사용자에게 실질적으로 도움이 되는가?
   - 실제 의사결정이나 문제해결에 도움이 되는 내용인지
   - 실행 가능하고 구체적인 정보를 제공하는지

4. **구체성 (Specificity)**: 구체적인 사례나 데이터가 포함되어 있는가?
   - 일반적인 설명보다는 구체적인 예시나 수치가 포함되었는지
   - 추상적이지 않고 명확하고 구체적인 내용인지

5. **균형성 (Balance)**: 장점과 단점이 균형있게 제시되었는가?
   - 편향되지 않고 객관적인 관점을 유지하는지
   - 긍정적/부정적 측면을 적절히 고려했는지

**전체 점수**는 위 5개 점수의 평균으로 계산하세요.

**품질 등급**:
- 9-10점: excellent (재시도 불필요)
- 7-8점: good (재시도 불필요)  
- 5-6점: fair (재시도 고려)
- 0-4점: poor (재시도 필요)

**개선사항**에는 구체적이고 실행 가능한 피드백을 포함해주세요.
만약 답변이 사용자 질문의 핵심을 벗어나거나 중요한 정보가 부족하다면, 
새로운 프롬프트나 접근 방법을 제안해주세요.

**should_retry**는 전체 점수가 6점 미만이거나 질문의 핵심을 벗어난 경우 true로 설정하세요.

**출력 형식 (반드시 JSON만 출력하세요):**
{{
  "accuracy_score": 점수(0-10),
  "completeness_score": 점수(0-10),
  "usefulness_score": 점수(0-10),
  "specificity_score": 점수(0-10),
  "balance_score": 점수(0-10),
  "overall_score": 전체점수(0-10),
  "quality_level": "excellent|good|fair|poor",
  "improvement_suggestions": ["구체적인 개선사항1", "구체적인 개선사항2", "..."],
  "should_retry": true/false
}}
"""
    
    def _get_default_value(self, field: str):
        """필수 필드의 기본값 반환"""
        if field in ["accuracy_score", "completeness_score", "usefulness_score", 
                     "specificity_score", "balance_score", "overall_score"]:
            return 5
        elif field == "quality_level":
            return "fair"
        elif field == "improvement_suggestions":
            return ["평가 결과를 생성하지 못했습니다."]
        elif field == "should_retry":
            return False
        else:
            return None