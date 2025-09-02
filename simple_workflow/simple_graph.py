"""
🎯 LangGraph 워크플로우 구성 - 초보자용 간단 버전

그래프(Graph)는 노드들을 연결해서 실행 순서를 정의합니다.
StateGraph를 사용해서 워크플로우를 만들고 실행합니다.

⚠️ 주의: 이 파일을 실행하려면 langgraph 패키지가 필요합니다.
pip install langgraph 또는 uv pip install langgraph
"""

# LangGraph 관련 임포트 (실제 프로젝트에서 사용)
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("⚠️ langgraph 패키지가 설치되지 않았습니다.")
    print("설치 방법: pip install langgraph")
    LANGGRAPH_AVAILABLE = False

from simple_state import SimpleState
from simple_nodes import (
    input_validation_node,
    culture_analysis_node,
    salary_analysis_node,
    growth_analysis_node,
    report_generation_node
)


class SimpleWorkflow:
    """
    🏗️ 간단한 워크플로우 클래스
    
    실제 BlindInsightWorkflow는 복잡하지만,
    여기서는 핵심 개념만 보여드립니다.
    """
    
    def __init__(self):
        self.graph = None
        if LANGGRAPH_AVAILABLE:
            self._build_graph()
    
    def _build_graph(self):
        """
        📊 LangGraph로 워크플로우 구성
        
        🔄 실행 순서:
        입력검증 → 문화분석 → 연봉분석 → 성장분석 → 보고서생성 → 끝
        """
        
        # 1️⃣ StateGraph 생성 (상태 타입 지정)
        workflow = StateGraph(SimpleState)
        
        # 2️⃣ 노드들 추가 (이름과 함수를 매핑)
        workflow.add_node("입력검증", input_validation_node)
        workflow.add_node("문화분석", culture_analysis_node)  
        workflow.add_node("연봉분석", salary_analysis_node)
        workflow.add_node("성장분석", growth_analysis_node)
        workflow.add_node("보고서생성", report_generation_node)
        
        # 3️⃣ 엣지(연결선) 추가 - 실행 순서 정의
        workflow.add_edge("입력검증", "문화분석")    # 입력검증 → 문화분석
        workflow.add_edge("문화분석", "연봉분석")    # 문화분석 → 연봉분석  
        workflow.add_edge("연봉분석", "성장분석")    # 연봉분석 → 성장분석
        workflow.add_edge("성장분석", "보고서생성")  # 성장분석 → 보고서생성
        workflow.add_edge("보고서생성", END)        # 보고서생성 → 끝
        
        # 4️⃣ 시작점 설정
        workflow.set_entry_point("입력검증")
        
        # 5️⃣ 실행 가능한 그래프로 컴파일
        self.graph = workflow.compile()
    
    async def run_analysis(self, company_name: str) -> SimpleState:
        """
        🚀 워크플로우 실행
        
        회사 이름을 받아서 전체 분석을 실행합니다.
        """
        if not LANGGRAPH_AVAILABLE:
            print("❌ LangGraph가 없어서 수동 실행합니다...")
            return self._manual_run(company_name)
        
        # 초기 상태 생성
        initial_state = SimpleState(company_name=company_name)
        
        print(f"🎯 {company_name} 분석 시작...")
        print("=" * 50)
        
        # LangGraph로 워크플로우 실행
        final_state = await self.graph.ainvoke(initial_state)
        
        return final_state
    
    def _manual_run(self, company_name: str) -> SimpleState:
        """
        🔧 수동 워크플로우 실행 (LangGraph 없이)
        
        LangGraph가 없을 때 노드들을 순서대로 실행합니다.
        실제로는 이렇게 수동으로 하지 않고 LangGraph가 자동으로 처리합니다.
        """
        print(f"🎯 {company_name} 수동 분석 시작...")
        print("=" * 50)
        
        # 초기 상태 생성
        state = SimpleState(company_name=company_name)
        
        # 각 노드를 순서대로 실행
        print("1️⃣ 입력 검증 중...")
        state = input_validation_node(state)
        print(f"   상태: {state.current_step}")
        
        print("\n2️⃣ 문화 분석 중...")
        state = culture_analysis_node(state)
        print(f"   결과: 문화 점수 {state.culture_score}/10")
        
        print("\n3️⃣ 연봉 분석 중...")
        state = salary_analysis_node(state)
        print(f"   결과: {state.salary_info}")
        
        print("\n4️⃣ 성장성 분석 중...")
        state = growth_analysis_node(state)
        print(f"   결과: 성장 점수 {state.growth_score}/10")
        
        print("\n5️⃣ 보고서 생성 중...")
        state = report_generation_node(state)
        print("   완료!")
        
        return state


# 🧪 테스트 및 실행 함수들
async def test_langgraph_workflow():
    """LangGraph를 사용한 워크플로우 테스트"""
    if not LANGGRAPH_AVAILABLE:
        print("❌ LangGraph가 설치되지 않아 테스트를 건너뜁니다.")
        return
    
    workflow = SimpleWorkflow()
    result = await workflow.run_analysis("네이버")
    
    print("\n📋 최종 결과:")
    print("=" * 50)
    print(result.final_report)
    
    print(f"\n📝 실행 로그:")
    for log in result.logs:
        print(f"  {log}")


def test_manual_workflow():
    """수동 워크플로우 테스트 (LangGraph 없이)"""
    workflow = SimpleWorkflow()
    result = workflow._manual_run("토스")
    
    print("\n📋 최종 결과:")
    print("=" * 50)
    print(result.final_report)
    
    print(f"\n📝 실행 로그:")
    for log in result.logs:
        print(f"  {log}")


def compare_with_original():
    """
    🔍 원본 복잡한 버전과 비교 설명
    
    이 간단한 버전과 실제 src/blindinsight/workflow의 차이점을 설명합니다.
    """
    print("🔍 간단한 버전 vs 실제 복잡한 버전")
    print("=" * 60)
    
    print("📊 간단한 버전 (여기):")
    print("  - 상태: 회사명, 점수들, 간단한 로그")
    print("  - 노드: 5개의 기본적인 분석 단계")
    print("  - 그래프: 순차적 실행만")
    print("  - 실행: 동기화, 간단한 결과")
    
    print("\n🎯 실제 복잡한 버전:")
    print("  - 상태: 40+ 필드, 복잡한 타입, 에러 처리, 성능 메트릭")
    print("  - 노드: 8개 노드, 병렬 처리, 에러 복구, 조건부 분기")  
    print("  - 그래프: 병렬/순차 모드, 체크포인팅, 재시작 기능")
    print("  - 실행: 비동기, AI 에이전트, RAG, MCP 통합")
    
    print("\n💡 학습 순서 추천:")
    print("  1. 이 간단한 버전으로 LangGraph 개념 이해")
    print("  2. 노드를 하나씩 복잡하게 만들어보기")
    print("  3. 병렬 처리 추가해보기")
    print("  4. 실제 AI 에이전트 연동하기")
    print("  5. 복잡한 상태 관리 추가하기")


if __name__ == "__main__":
    print("🎯 LangGraph 간단 워크플로우 테스트")
    print("=" * 50)
    
    # 1. 원본과의 비교 설명
    compare_with_original()
    
    print("\n" + "="*50)
    
    # 2. 실제 실행 테스트
    if LANGGRAPH_AVAILABLE:
        print("🚀 LangGraph 버전으로 실행:")
        import asyncio
        asyncio.run(test_langgraph_workflow())
    else:
        print("🔧 수동 버전으로 실행:")
        test_manual_workflow()