"""
BlindInsight AI 아키텍처 시각화 다이어그램 생성 스크립트
텍스트 기반 ASCII 다이어그램 + Mermaid 다이어그램 생성
"""

import os
from typing import Dict, List, Tuple

def generate_ascii_diagrams():
    """ASCII 아트 다이어그램 생성"""
    
    print("🎯 BlindInsight AI Architecture Visualization")
    print("=" * 60)
    
    # 1. LangGraph StateGraph Workflow
    print("\n1. 📊 LangGraph StateGraph Workflow")
    print("-" * 40)
    print("""
    ┌─────────────────┐
    │ Input Validation │
    └─────────┬───────┘
              │
              ▼
         ┌─────────┐
         │Parallel?│◄─── Decision Diamond
         │  Mode?  │
         └────┬────┘
              │
         ┌────┴────┐
         │   Yes   │   No
         ▼         ▼
    ┌─────────┐   ┌─────────────────┐
    │Parallel │   │Culture Analysis │
    │Analysis │   └─────────┬───────┘
    │         │             ▼
    │ ┌─────┐ │   ┌─────────────────┐
    │ │Cult │ │   │Compensation     │
    │ │Comp │ │   │Analysis         │
    │ │Grow │ │   └─────────┬───────┘
    │ └─────┘ │             ▼
    └────┬────┘   ┌─────────────────┐
         │        │Growth Analysis  │
         │        └─────────┬───────┘
         │                  │
         └──────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │Career Analysis  │
         └─────────┬───────┘
                   │
                   ▼
         ┌─────────────────┐
         │   Synthesis     │
         └─────────┬───────┘
                   │
                   ▼
         ┌─────────────────┐
         │Report Generation│
         └─────────────────┘
    """)
    
    # 2. Agent Hierarchy
    print("\n2. 🏗️ Agent Hierarchy Structure")
    print("-" * 40)
    print("""
                    ┌─────────────────────┐
                    │   BaseAgent (ABC)   │
                    │                     │
                    │ • RAG 검색 기능     │
                    │ • MCP 서비스 연동   │
                    │ • LLM 응답 생성     │
                    │ • 결과 검증         │
                    │ • 오류 처리         │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │Culture        │ │Compensation   │ │Growth         │
    │AnalysisAgent  │ │AnalysisAgent  │ │StabilityAgent │
    │               │ │               │ │               │
    │• 회사 문화    │ │• 연봉 정보    │ │• 성장성 분석  │
    │• 워라밸 평가  │ │• 복리후생     │ │• 안정성 평가  │
    │• 조직 특성    │ │• 보상 수준    │ │• 미래 전망    │
    └───────────────┘ └───────────────┘ └───────────────┘
                               │
                               ▼
                    ┌───────────────────┐
                    │  CareerPathAgent  │
                    │                   │
                    │ • 커리어 경로     │
                    │ • 개인화 적합도   │
                    │ • 발전 가능성     │
                    └───────────────────┘
    """)
    
    # 3. Data Flow Architecture
    print("\n3. 🔄 Data Flow Architecture")
    print("-" * 40)
    print("""
    사용자 요청 ──────┐
                    │
                    ▼
    AnalysisRequest 생성 ──────► WorkflowState 초기화
                                        │
                                        ▼
                               LangGraph StateGraph 실행
                                        │
                        ┌───────────────┼───────────────┐
                        │               │               │
                        ▼               ▼               ▼
                ┌──────────────┐ ┌─────────────┐ ┌─────────────┐
                │  RAG 검색    │ │ MCP 서비스  │ │ LLM 처리    │
                │ (ChromaDB)   │ │   호출      │ │(OpenAI GPT-4)│
                │              │ │             │ │             │
                │• culture_    │ │• BlindData  │ │• 분석       │
                │  reviews     │ │• JobSite    │ │• 생성       │
                │• salary_     │ │• SalaryData │ │• 합성       │
                │  discussions │ │• CompanyNews│ │             │
                │• career_     │ │             │ │             │
                │  advice      │ │             │ │             │
                └──────┬───────┘ └──────┬──────┘ └──────┬──────┘
                       │                │               │
                       └────────────────┼───────────────┘
                                        │
                                        ▼
                               각 에이전트 결과
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
            ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
            │Culture      │    │Compensation │    │Growth       │
            │Report       │    │Report       │    │Report       │
            └─────────────┘    └─────────────┘    └─────────────┘
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │Career Report    │
                               └─────────┬───────┘
                                         │
                                         ▼
                               ┌─────────────────┐
                               │   Synthesis     │
                               └─────────┬───────┘
                                         │
                                         ▼
                               ┌─────────────────┐
                               │Final Report     │
                               └─────────────────┘
    """)
    
    # 4. 3-Tab UI Structure
    print("\n4. 💻 3-Tab UI Structure")
    print("-" * 40)
    print("""
    ┌─────────────────────────────────────────────────────────┐
    │              Company Analysis Page                       │
    │                                                         │
    │  if company_name:                                       │
    │    tab1, tab2, tab3 = st.tabs([...])                  │
    └─────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    ┌──────────┐   ┌─────────────┐   ┌─────────────┐
    │📋 Tab 1  │   │ℹ️ Tab 2     │   │🤖 Tab 3     │
    │커뮤니티   │   │회사 정보    │   │AI 분석      │
    │          │   │             │   │             │
    │• 직원 리뷰│   │• 기본 정보  │   │• RAG 검색   │
    │• 면접 후기│   │• 채용 공고  │   │• MCP 데이터 │
    │• 회사 평점│   │• 회사 소개  │   │• 에이전트   │
    │• 커뮤니티 │   │• 외부 링크  │   │  분석       │
    │  데이터   │   │             │   │• 종합 리포트│
    └──────────┘   └─────────────┘   └─────┬───────┘
                                           │
                                           ▼
                              ┌─────────────────────┐
                              │  LangGraph 워크플로우 │
                              │      실행           │
                              └─────────────────────┘
    """)
    
    print("\n✨ Architecture Overview Complete!")
    print("📋 Key Components:")
    print("   • LangGraph StateGraph: 워크플로우 오케스트레이션")
    print("   • Multi-Agent System: 전문화된 AI 에이전트들")
    print("   • RAG + MCP Integration: 내부 지식베이스 + 외부 API")
    print("   • 3-Tab Streamlit UI: 사용자 친화적 인터페이스")

def generate_mermaid_diagrams():
    """Mermaid 다이어그램 코드 생성"""
    
    workflow_mermaid = """
graph TB
    A[Input Validation] --> B{Parallel Mode?}
    B -->|Yes| C[Parallel Analysis]
    B -->|No| D[Culture Analysis]
    
    C --> E[Culture Analysis ⚡]
    C --> F[Compensation Analysis ⚡]
    C --> G[Growth Analysis ⚡]
    
    D --> H[Compensation Analysis]
    H --> I[Growth Analysis]
    
    E --> J[Career Analysis]
    F --> J
    G --> J
    I --> J
    
    J --> K[Synthesis]
    K --> L[Report Generation]
    L --> M[END]
    
    classDef inputNode fill:#e3f2fd,stroke:#333,stroke-width:2px
    classDef processNode fill:#fff3e0,stroke:#333,stroke-width:2px
    classDef decisionNode fill:#f3e5f5,stroke:#333,stroke-width:2px
    classDef parallelNode fill:#fce4ec,stroke:#333,stroke-width:2px
    classDef outputNode fill:#e8f5e8,stroke:#333,stroke-width:2px
    
    class A inputNode
    class B decisionNode
    class C parallelNode
    class D,H,I,J,K processNode
    class E,F,G parallelNode
    class L,M outputNode
    """
    
    agent_hierarchy_mermaid = """
graph TD
    A[BaseAgent - ABC] --> B[CultureAnalysisAgent]
    A --> C[CompensationAnalysisAgent]
    A --> D[GrowthStabilityAgent]
    A --> E[CareerPathAgent]
    
    A1[RAG 검색 기능] --> A
    A2[MCP 서비스 연동] --> A
    A3[LLM 응답 생성] --> A
    A4[결과 검증 및 성능 추적] --> A
    A5[오류 처리 및 재시도 로직] --> A
    
    B --> B1[회사 문화 분석<br/>워라밸 평가<br/>조직 특성 진단]
    C --> C1[연봉 정보 분석<br/>복리후생 평가<br/>보상 수준 비교]
    D --> D1[성장성 분석<br/>안정성 평가<br/>미래 전망 예측]
    E --> E1[커리어 경로 분석<br/>개인화된 적합도<br/>발전 가능성]
    
    classDef baseClass fill:#e3f2fd,stroke:#333,stroke-width:3px
    classDef agentClass fill:#fff3e0,stroke:#333,stroke-width:2px
    classDef featureClass fill:#f0f0f0,stroke:#666,stroke-width:1px
    classDef specClass fill:#e8f5e8,stroke:#333,stroke-width:1px
    
    class A baseClass
    class B,C,D,E agentClass
    class A1,A2,A3,A4,A5 featureClass
    class B1,C1,D1,E1 specClass
    """
    
    data_flow_mermaid = """
graph TD
    A[사용자 요청] --> B[AnalysisRequest 생성]
    B --> C[WorkflowState 초기화]
    C --> D[LangGraph StateGraph 실행]
    
    D --> E[RAG 검색 - ChromaDB]
    D --> F[MCP 서비스 호출]
    
    E --> E1[culture_reviews]
    E --> E2[salary_discussions]
    E --> E3[career_advice]
    E --> E4[interview_reviews]
    E --> E5[company_general]
    
    F --> F1[BlindDataProvider]
    F --> F2[JobSiteProvider]
    F --> F3[SalaryDataProvider]
    F --> F4[CompanyNewsProvider]
    
    E1 --> G[LLM 분석 - OpenAI GPT-4]
    E2 --> G
    E3 --> G
    E4 --> G
    E5 --> G
    
    F1 --> G
    F2 --> G
    F3 --> G
    F4 --> G
    
    G --> H[각 에이전트 결과]
    H --> H1[Culture Report]
    H --> H2[Compensation Report]
    H --> H3[Growth Report]
    H --> H4[Career Report]
    
    H1 --> I[SynthesisNode - 결과 통합]
    H2 --> I
    H3 --> I
    H4 --> I
    
    I --> J[ReportGenerationNode]
    J --> K[최종 분석 보고서]
    
    classDef userClass fill:#e3f2fd,stroke:#333,stroke-width:2px
    classDef processClass fill:#fff3e0,stroke:#333,stroke-width:2px
    classDef dataClass fill:#f3e5f5,stroke:#333,stroke-width:2px
    classDef llmClass fill:#fce4ec,stroke:#333,stroke-width:2px
    classDef resultClass fill:#e8f5e8,stroke:#333,stroke-width:2px
    
    class A,B userClass
    class C,D,I,J processClass
    class E,F,E1,E2,E3,E4,E5,F1,F2,F3,F4 dataClass
    class G llmClass
    class H,H1,H2,H3,H4,K resultClass
    """
    
    ui_structure_mermaid = """
graph TD
    A[Company Analysis Page] --> B{company_name 존재?}
    B -->|Yes| C[3-Tab 구조 생성]
    B -->|No| D[회사명 입력 요청]
    
    C --> E[📋 Tab 1: 커뮤니티]
    C --> F[ℹ️ Tab 2: 회사 정보]
    C --> G[🤖 Tab 3: AI 분석]
    
    E --> E1[직원 리뷰<br/>면접 후기<br/>회사 평점<br/>커뮤니티 데이터]
    
    F --> F1[기본 정보<br/>채용 공고<br/>회사 소개<br/>외부 링크]
    
    G --> G1[AI 분석 시작 버튼]
    G1 --> G2[사용자 프로필 입력]
    G2 --> G3[LangGraph 워크플로우 실행]
    
    G3 --> H[RAG + MCP 데이터 수집]
    H --> I[멀티 에이전트 분석]
    I --> J[결과 시각화]
    
    I --> I1[CultureAnalysisAgent]
    I --> I2[CompensationAnalysisAgent]
    I --> I3[GrowthStabilityAgent]
    I --> I4[CareerPathAgent]
    
    J --> J1[문화 분석 결과]
    J --> J2[연봉 분석 결과]
    J --> J3[성장성 분석 결과]
    J --> J4[커리어 분석 결과]
    J --> J5[종합 점수 및 추천]
    
    classDef pageClass fill:#e3f2fd,stroke:#333,stroke-width:3px
    classDef tabClass fill:#fff3e0,stroke:#333,stroke-width:2px
    classDef aiClass fill:#e8f5e8,stroke:#333,stroke-width:2px
    classDef agentClass fill:#f3e5f5,stroke:#333,stroke-width:2px
    classDef resultClass fill:#fce4ec,stroke:#333,stroke-width:2px
    
    class A pageClass
    
    class C,E,F,G tabClass
    class G1,G2,G3,H,I,J aiClass
    class I1,I2,I3,I4 agentClass
    class E1,F1,J1,J2,J3,J4,J5 resultClass
    """
    
    return {
        'workflow': workflow_mermaid,
        'agent_hierarchy': agent_hierarchy_mermaid,
        'data_flow': data_flow_mermaid,
        'ui_structure': ui_structure_mermaid
    }

def generate_mermaid_files():
    """Mermaid 다이어그램 파일 생성"""
    
    # diagrams 디렉토리 생성
    diagrams_dir = 'docs/diagrams'
    os.makedirs(diagrams_dir, exist_ok=True)
    
    diagrams = generate_mermaid_diagrams()
    
    files_created = []
    for name, content in diagrams.items():
        filename = f'{diagrams_dir}/{name}_diagram.md'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# {name.replace('_', ' ').title()} Diagram\n\n")
            f.write("```mermaid\n")
            f.write(content.strip())
            f.write("\n```\n")
        files_created.append(filename)
        print(f"✅ Created: {filename}")
    
    return files_created

def create_summary_document():
    """종합 시각화 문서 생성"""
    
    summary_content = """# BlindInsight AI - 아키텍처 시각화 종합 문서

## 📋 개요

본 문서는 BlindInsight AI의 LangGraph 기반 멀티 에이전트 아키텍처를 시각적으로 표현합니다.

## 🏗️ 아키텍처 구성 요소

### 1. LangGraph StateGraph 워크플로우
- **순차 실행 모드**: Culture → Compensation → Growth → Career 순서로 실행
- **병렬 실행 모드**: Culture, Compensation, Growth를 병렬로 실행 후 Career 분석
- **상태 관리**: WorkflowState를 통한 에이전트 간 데이터 공유

### 2. 멀티 에이전트 계층
```
BaseAgent (추상 클래스)
├── RAG 검색 기능
├── MCP 서비스 연동
├── LLM 응답 생성
├── 결과 검증 및 성능 추적
└── 오류 처리 및 재시도 로직

전문 에이전트들
├── CultureAnalysisAgent - 회사 문화 분석
├── CompensationAnalysisAgent - 연봉 정보 분석
├── GrowthStabilityAgent - 성장성 분석
└── CareerPathAgent - 커리어 경로 분석
```

### 3. 데이터 통합 시스템

#### RAG (Retrieval Augmented Generation)
- **Vector Database**: ChromaDB
- **컬렉션들**:
  - culture_reviews: 회사 문화 리뷰
  - salary_discussions: 연봉 정보
  - career_advice: 커리어 조언
  - interview_reviews: 면접 후기
  - company_general: 일반 회사 정보

#### MCP (Model Context Protocol)
- **BlindDataProvider**: Blind.com 리뷰 데이터
- **JobSiteProvider**: 채용 사이트 정보
- **SalaryDataProvider**: 연봉 정보
- **CompanyNewsProvider**: 뉴스 및 동향

### 4. 3-Tab 사용자 인터페이스

#### Tab 1: 📋 커뮤니티
- 직원 리뷰 및 면접 후기
- 회사 평점 정보
- 실시간 커뮤니티 데이터

#### Tab 2: ℹ️ 회사 정보
- 기본 회사 정보
- 채용 공고
- 회사 소개 및 외부 링크

#### Tab 3: 🤖 AI 분석
- **핵심 기능**: RAG + MCP 통합 AI 분석
- **워크플로우**:
  1. 사용자 프로필 입력
  2. LangGraph 워크플로우 실행
  3. RAG 검색 및 MCP 데이터 수집
  4. 멀티 에이전트 병렬/순차 분석
  5. 결과 시각화 및 종합 리포트

## 🔄 데이터 흐름

1. **사용자 입력** → AnalysisRequest 생성
2. **WorkflowState 초기화** → LangGraph StateGraph 실행
3. **데이터 수집**:
   - RAG 벡터 검색 (ChromaDB)
   - MCP 서비스 호출 (외부 API)
   - LLM 처리 (OpenAI GPT-4)
4. **에이전트 실행**:
   - CultureAnalysisAgent
   - CompensationAnalysisAgent
   - GrowthStabilityAgent
   - CareerPathAgent
5. **결과 통합** → SynthesisNode
6. **최종 보고서 생성** → ReportGenerationNode

## 🎯 핵심 특징

- **LangGraph StateGraph**: 워크플로우 상태 관리 및 오케스트레이션
- **멀티 에이전트 협업**: 전문화된 에이전트들의 협업적 분석
- **RAG + MCP 통합**: 내부 지식베이스와 외부 API 연동
- **병렬/순차 실행**: 성능과 안정성 사이의 선택 가능
- **모듈화 설계**: 새로운 에이전트와 데이터 소스 쉽게 추가
- **포괄적 분석**: 문화, 연봉, 성장성, 커리어 등 종합 분석
- **개인화**: 사용자 프로필 기반 맞춤형 분석
- **실시간 UI**: Streamlit 기반 인터랙티브 웹 인터페이스

## 📊 시각화 파일들

### Mermaid 다이어그램들
- `workflow_diagram.md`: LangGraph StateGraph 워크플로우
- `agent_hierarchy_diagram.md`: 에이전트 계층 구조
- `data_flow_diagram.md`: 데이터 흐름도
- `ui_structure_diagram.md`: 3-Tab UI 구조

### 사용 방법
1. **온라인 시각화**: https://mermaid.live/ 에서 Mermaid 코드 복사-붙여넣기
2. **GitHub/GitLab**: 마크다운에서 자동 렌더링
3. **문서 도구**: Mermaid 지원 도구에서 통합

---

*본 문서는 BlindInsight AI의 LangGraph 기반 멀티 에이전트 아키텍처를 완전히 문서화한 것입니다.*
"""
    
    with open('docs/BlindInsight_Architecture_Summary.md', 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    print("✅ Created: docs/BlindInsight_Architecture_Summary.md")

def main():
    """메인 실행 함수"""
    print("🎯 BlindInsight AI - Architecture Diagram Generator")
    print("=" * 60)
    
    # ASCII 다이어그램 출력
    generate_ascii_diagrams()
    
    print("\n" + "=" * 60)
    print("📝 Generating Mermaid diagram files...")
    
    # Mermaid 파일 생성
    try:
        files = generate_mermaid_files()
        print(f"\n✅ Generated {len(files)} Mermaid diagram files:")
        for file in files:
            print(f"   📄 {file}")
        
        # 종합 문서 생성
        create_summary_document()
        
        print("\n💡 Usage Instructions:")
        print("   1. Copy Mermaid code to https://mermaid.live/ for visualization")
        print("   2. Use in GitHub/GitLab markdown for automatic rendering")
        print("   3. Integrate with documentation tools that support Mermaid")
        
    except Exception as e:
        print(f"❌ Error generating Mermaid files: {str(e)}")
    
    print("\n🎉 Architecture visualization complete!")

if __name__ == "__main__":
    main()