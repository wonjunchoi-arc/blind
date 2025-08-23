"""
LangGraph 워크플로우 오케스트레이션 - BlindInsight AI 모듈 구조 가이드

🎯 이 모듈의 역할:
BlindInsight AI의 LangGraph 기반 워크플로우 시스템을 구현합니다.
여러 AI 에이전트들을 조율하여 기업 분석을 체계적으로 수행합니다.

📂 모듈 구성:
├── state.py     : 워크플로우 상태 관리 (WorkflowState, WorkflowConfig)
├── nodes.py     : 개별 노드 구현 (각 분석 에이전트 래퍼)
├── graph.py     : 워크플로우 그래프 구성 (StateGraph, 실행 흐름)
└── __init__.py  : 모듈 인터페이스 (주요 클래스/함수 export)

🚀 주요 사용 패턴:
```python
# 1. 간단한 분석 실행
from blindinsight.workflow import execute_quick_analysis
result = await execute_quick_analysis("회사명", "직무")

# 2. 개인화된 분석
from blindinsight.workflow import AnalysisWorkflow
workflow = AnalysisWorkflow()
result = await workflow.analyze("회사명", "직무", user_profile)

# 3. 커스텀 워크플로우
from blindinsight.workflow import BlindInsightWorkflow, WorkflowConfig
config = WorkflowConfig(enable_parallel_execution=True)
workflow = BlindInsightWorkflow(config)
result = await workflow.analyze_company(request, user_profile)
```

🔧 커스터마이징 가이드:
1. 새로운 분석 타입 추가:
   - state.py에 새 Report 필드 추가
   - nodes.py에 새 Node 클래스 구현
   - graph.py에 워크플로우 연결 추가

2. 실행 흐름 변경:
   - WorkflowConfig로 설정 조정
   - graph.py의 _define_workflow_edges 수정

3. 에러 처리 개선:
   - state.py의 에러 처리 메서드 확장
   - nodes.py의 try-catch 블록 커스터마이징
"""

from .graph import *
from .nodes import *
from .state import *

__all__ = [
    "BlindInsightWorkflow",
    "AnalysisWorkflow", 
    "WorkflowState",
    "create_analysis_workflow",
    "WorkflowNode",
    "validate_input_node",
    "culture_analysis_node",
    "compensation_analysis_node",
    "growth_analysis_node", 
    "career_analysis_node",
    "synthesis_node",
    "report_generation_node"
]