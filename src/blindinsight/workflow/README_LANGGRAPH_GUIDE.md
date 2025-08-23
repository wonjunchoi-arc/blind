# LangGraph 초보자를 위한 BlindInsight 워크플로우 가이드

## 🎯 BlindInsight 워크플로우 전체 구조 분석

### 📁 파일별 역할 및 에이전트 노드 구분

#### 1. **state.py** - 워크플로우 상태 관리
- **핵심 역할**: 모든 노드 간 데이터 공유를 위한 중앙 상태 저장소
- **주요 클래스**: 
  - `WorkflowState`: 분석 결과, 에러, 진행 상황 등 모든 정보 보관
  - `WorkflowConfig`: 워크플로우 실행 설정 (병렬/순차, 캐싱 등)

#### 2. **nodes.py** - 에이전트 노드 구현
**기본 노드 클래스**:
- `WorkflowNode`: 모든 노드의 부모 클래스 (공통 실행 로직)

**분석 에이전트 노드들**:
- `CultureAnalysisNode`: 기업 문화 분석 (워라밸, 회사 분위기)
- `CompensationAnalysisNode`: 보상 체계 분석 (연봉, 복리후생)
- `GrowthAnalysisNode`: 성장성 분석 (회사 성장 가능성, 안정성)
- `CareerAnalysisNode`: 커리어 경로 분석 (승진 가능성, 개인 적합도)

**특수 노드들**:
- `InputValidationNode`: 입력 데이터 검증
- `ParallelAnalysisNode`: 여러 분석을 동시 실행
- `SynthesisNode`: 분석 결과 종합
- `ReportGenerationNode`: 최종 보고서 생성

#### 3. **graph.py** - 워크플로우 구성
- **핵심 역할**: 노드들 간의 연결과 실행 순서 정의
- **주요 클래스**:
  - `BlindInsightWorkflow`: 메인 워크플로우 오케스트레이터
  - `AnalysisWorkflow`: 간편한 분석 실행 인터페이스

### 🔄 워크플로우 실행 흐름

#### **순차 실행 모드**:
```
input_validation → culture_analysis → compensation_analysis → growth_analysis → career_analysis → synthesis → report_generation
```

#### **병렬 실행 모드** (성능 최적화):
```
input_validation → parallel_analysis (문화+보상+성장 동시 실행) → career_analysis → synthesis → report_generation
```

### 🎯 에이전트 노드 vs 워크플로우 노드 구분

#### **에이전트 노드** (AI 분석 수행):
1. `CultureAnalysisAgent` → `CultureAnalysisNode`
2. `CompensationAnalysisAgent` → `CompensationAnalysisNode`
3. `GrowthStabilityAgent` → `GrowthAnalysisNode`
4. `CareerPathAgent` → `CareerAnalysisNode`

#### **워크플로우 노드** (시스템 처리):
- `InputValidationNode`: 데이터 검증
- `ParallelAnalysisNode`: 병렬 처리 제어
- `SynthesisNode`: 결과 종합
- `ReportGenerationNode`: 보고서 생성

## 🔧 커스터마이징 시나리오별 가이드

### **1. 새로운 분석 타입 추가 (예: 리스크 분석)**

**Step 1: state.py 수정**
```python
# 새 리포트 필드 추가
risk_report: Optional[RiskReport] = Field(None, description="리스크 분석 결과")
```

**Step 2: nodes.py에 새 노드 추가**
```python
class RiskAnalysisNode(WorkflowNode):
    def __init__(self):
        super().__init__("risk_analysis")
    
    async def _execute_logic(self, state: WorkflowState) -> WorkflowState:
        # 리스크 분석 에이전트 호출
        # 결과를 state.risk_report에 저장
        return state
```

**Step 3: graph.py에 워크플로우 연결**
```python
# 노드 추가
workflow.add_node("risk_analysis", self._risk_analysis_wrapper)

# 엣지 연결 (career_analysis 후, synthesis 전)
workflow.add_edge("career_analysis", "risk_analysis")
workflow.add_edge("risk_analysis", "synthesis")
```

### **2. 병렬 처리 범위 확장**
현재는 culture + compensation + growth만 병렬 처리됩니다.
career 분석도 포함하려면 `ParallelAnalysisNode`에서 `career_node`를 tasks에 추가하고 결과 병합 로직을 확장하면 됩니다.

### **3. 조건부 분기 추가**
특정 조건에 따라 다른 분석을 수행하려면:
```python
# 조건부 엣지 함수 정의
def should_run_detailed_analysis(state):
    return "detailed" if state.request.analysis_type == "comprehensive" else "basic"

# 조건부 엣지 추가
workflow.add_conditional_edges(
    "input_validation",
    should_run_detailed_analysis,
    {"detailed": "comprehensive_analysis", "basic": "basic_analysis"}
)
```

### **4. 진행률 계산 커스터마이징**
`state.py`의 각 노드에서 `update_progress()` 호출 부분을 수정하여 진행률 계산 방식을 변경할 수 있습니다.

```python
# 현재 방식 (고정 증가)
state.update_progress(state.progress + 0.2)

# 커스텀 방식 (동적 계산)
completed_stages = len(state.completed_stages)
total_stages = 7  # 전체 단계 수
state.update_progress(completed_stages / total_stages)
```

## 💡 LangGraph 핵심 개념 정리

### **1. 상태 중심 설계 (State-Centric)**
- 모든 데이터는 `WorkflowState`에 저장
- 각 노드는 상태를 받아 수정된 상태를 반환
- `Annotated[Sequence, operator.add]` 패턴으로 안전한 리스트 병합

### **2. 함수형 노드 패턴**
```python
async def node_function(state: WorkflowState) -> WorkflowState:
    # 작업 수행
    # 상태 수정
    return modified_state
```

### **3. 선언적 그래프 구성**
```python
workflow = StateGraph(WorkflowState)
workflow.add_node("node_name", node_function)
workflow.add_edge("from_node", "to_node")
workflow.set_entry_point("start_node")
graph = workflow.compile()
```

### **4. 비동기 실행**
- 모든 노드는 `async/await` 패턴 사용
- `asyncio.gather()`로 병렬 처리
- 체크포인팅으로 중간 상태 저장 및 복구

## 🚀 실습 예제

### **간단한 사용법**
```python
from blindinsight.workflow import execute_quick_analysis

# 빠른 분석 실행
result = await execute_quick_analysis("구글", "소프트웨어 엔지니어")
print(f"분석 완료: {result['success']}")
print(f"전체 점수: {result['results']['comprehensive']['overall_score']}")
```

### **커스텀 설정 사용법**
```python
from blindinsight.workflow import BlindInsightWorkflow, WorkflowConfig

# 병렬 실행 + 디버그 모드
config = WorkflowConfig(
    enable_parallel_execution=True,
    enable_debug_logging=True,
    max_execution_time=180
)

workflow = BlindInsightWorkflow(config)
result = await workflow.analyze_company(request, user_profile)
```

## 🔍 디버깅 및 모니터링

### **1. 디버그 모드 활성화**
```python
config = WorkflowConfig(enable_debug_logging=True)
```

### **2. 성능 모니터링**
```python
performance = result_state.get_performance_summary()
print(f"총 실행 시간: {performance['total_execution_time']}초")
print(f"API 호출 수: {performance['total_api_calls']}")
print(f"토큰 사용량: {performance['total_tokens']}")
```

### **3. 에러 확인**
```python
if result_state.has_errors():
    for error in result_state.errors:
        print(f"에러: {error}")
```

## 🎓 학습 순서 추천

1. **기본 개념 이해**: state.py의 `WorkflowState` 구조 파악
2. **노드 구현 학습**: nodes.py의 `WorkflowNode` 패턴 이해
3. **그래프 구성 실습**: graph.py의 엣지 연결 방식 학습
4. **실제 커스터마이징**: 새로운 분석 타입 추가해보기
5. **고급 기능 활용**: 조건부 분기, 병렬 처리 최적화

## 📚 추가 학습 자료

- LangGraph 공식 문서: https://python.langchain.com/docs/langgraph
- 비동기 프로그래밍: Python asyncio 문서
- Pydantic 모델: https://docs.pydantic.dev/

---
*이 가이드는 LangGraph 초보자를 위해 작성되었습니다. 각 파일의 주석과 함께 참고하여 학습하시기 바랍니다.*