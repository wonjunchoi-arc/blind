"""
🎯 LangGraph 노드 구현 - 초보자용 간단 버전

노드(Node)는 실제 작업을 수행하는 함수입니다.
각 노드는 상태를 받아서 처리하고, 수정된 상태를 반환합니다.

패턴: state를 받아서 → 작업 수행 → 수정된 state 반환
"""

import time
import random
from simple_state import SimpleState


# 🏗️ 노드 함수들 - LangGraph에서는 이런 함수들이 노드가 됩니다

def input_validation_node(state: SimpleState) -> SimpleState:
    """
    📥 입력 검증 노드
    
    사용자가 입력한 회사 이름이 유효한지 확인합니다.
    실제로는 데이터베이스 확인, API 호출 등을 할 수 있습니다.
    """
    state.move_to_step("입력 검증")
    
    if not state.company_name or len(state.company_name) < 2:
        state.add_log("❌ 잘못된 회사 이름")
        return state
    
    state.add_log(f"✅ '{state.company_name}' 회사 이름 확인됨")
    
    # 실제로는 시간이 걸리는 작업을 시뮬레이션
    time.sleep(0.5)
    
    return state


def culture_analysis_node(state: SimpleState) -> SimpleState:
    """
    🏢 문화 분석 노드
    
    회사 문화를 분석하고 점수를 매깁니다.
    실제로는 AI 에이전트가 리뷰 데이터를 분석할 것입니다.
    """
    state.move_to_step("문화 분석")
    
    # 간단한 모의 분석 (실제로는 복잡한 AI 분석)
    state.add_log("🔍 회사 리뷰 데이터 분석 중...")
    time.sleep(1)  # 분석 시간 시뮬레이션
    
    # 무작위로 점수 생성 (실제로는 AI가 계산)
    culture_score = random.randint(6, 10)
    state.culture_score = culture_score
    
    state.add_log(f"📊 문화 점수: {culture_score}/10")
    
    if culture_score >= 8:
        state.add_log("😊 좋은 회사 문화!")
    elif culture_score >= 6:
        state.add_log("😐 평범한 회사 문화")
    else:
        state.add_log("😞 아쉬운 회사 문화...")
    
    return state


def salary_analysis_node(state: SimpleState) -> SimpleState:
    """
    💰 연봉 분석 노드
    
    회사의 연봉 정보를 분석합니다.
    """
    state.move_to_step("연봉 분석")
    
    state.add_log("💰 연봉 데이터 수집 중...")
    time.sleep(1)
    
    # 간단한 모의 연봉 정보
    salary_ranges = [
        "신입: 4000-5000만원, 3년차: 6000-7000만원",
        "신입: 5000-6000만원, 5년차: 8000-1억원",
        "신입: 3500-4500만원, 경력: 6000-8000만원"
    ]
    
    state.salary_info = random.choice(salary_ranges)
    state.add_log(f"💼 연봉 정보: {state.salary_info}")
    
    return state


def growth_analysis_node(state: SimpleState) -> SimpleState:
    """
    📈 성장성 분석 노드
    
    회사의 성장 가능성을 분석합니다.
    """
    state.move_to_step("성장성 분석")
    
    state.add_log("📈 회사 성장성 분석 중...")
    time.sleep(1)
    
    growth_score = random.randint(5, 10)
    state.growth_score = growth_score
    
    state.add_log(f"🚀 성장 점수: {growth_score}/10")
    
    return state


def report_generation_node(state: SimpleState) -> SimpleState:
    """
    📝 최종 보고서 생성 노드
    
    모든 분석 결과를 종합해서 최종 보고서를 만듭니다.
    """
    state.move_to_step("보고서 생성")
    
    state.add_log("📝 최종 보고서 작성 중...")
    time.sleep(1)
    
    # 모든 분석 결과를 종합
    total_score = (state.culture_score or 0) + (state.growth_score or 0)
    avg_score = total_score / 2 if total_score > 0 else 0
    
    # 최종 보고서 작성
    report = f"""
🏢 {state.company_name} 분석 보고서
{'='*50}

📊 분석 결과:
- 문화 점수: {state.culture_score}/10
- 성장 점수: {state.growth_score}/10
- 종합 점수: {avg_score:.1f}/10

💰 연봉 정보:
{state.salary_info}

🎯 종합 평가:
"""
    
    if avg_score >= 8:
        report += "⭐ 추천! 매우 좋은 회사입니다."
    elif avg_score >= 6:
        report += "👍 괜찮은 회사입니다."
    else:
        report += "🤔 신중하게 고려해보세요."
    
    state.final_report = report
    state.add_log("✅ 최종 보고서 완성!")
    
    return state


# 💡 노드 테스트 함수
def test_single_node():
    """개별 노드 테스트"""
    print("🧪 개별 노드 테스트")
    
    # 상태 생성
    state = SimpleState(company_name="카카오")
    
    # 각 노드 순서대로 실행
    print("\n1️⃣ 입력 검증:")
    state = input_validation_node(state)
    print(f"현재 단계: {state.current_step}")
    
    print("\n2️⃣ 문화 분석:")
    state = culture_analysis_node(state)
    print(f"문화 점수: {state.culture_score}")
    
    print("\n3️⃣ 연봉 분석:")
    state = salary_analysis_node(state)
    print(f"연봉 정보: {state.salary_info}")
    
    print("\n4️⃣ 성장성 분석:")
    state = growth_analysis_node(state)
    print(f"성장 점수: {state.growth_score}")
    
    print("\n5️⃣ 보고서 생성:")
    state = report_generation_node(state)
    print("최종 보고서:")
    print(state.final_report)


if __name__ == "__main__":
    test_single_node()