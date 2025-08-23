"""
BlindInsight AI 메인 애플리케이션

Streamlit 기반의 메인 웹 애플리케이션으로,
회사 분석, 커리어 상담, 데이터 시각화 기능을 제공합니다.
"""

import streamlit as st
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import sys

# BlindInsight 모듈 임포트
sys.path.append(str(Path(__file__).parent.parent.parent))

from blindinsight.workflow.graph import BlindInsightWorkflow
from blindinsight.models.analysis import AnalysisRequest
from blindinsight.models.user import UserProfile, create_default_user_profile
from blindinsight.rag.knowledge_base import KnowledgeBase
from blindinsight.mcp.client import MCPClient
from blindinsight.mcp.providers import (
    BlindDataProvider, JobSiteProvider, 
    SalaryDataProvider, CompanyNewsProvider, DataProviderConfig
)
from blindinsight.models.base import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionManager:
    """
    세션 상태 관리 클래스
    
    Streamlit 세션 상태를 관리하고 사용자 데이터를 유지합니다.
    """
    
    @staticmethod
    def initialize_session():
        """세션 상태 초기화"""
        
        # 기본 세션 변수들 초기화
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.user_profile = create_default_user_profile()
            st.session_state.analysis_history = []
            st.session_state.current_analysis = None
            st.session_state.chat_messages = []
            st.session_state.selected_companies = []
            st.session_state.app_initialized = False
            
            logger.info("Streamlit 세션 상태 초기화 완료")
    
    @staticmethod
    def get_user_profile() -> UserProfile:
        """현재 사용자 프로필 반환"""
        return st.session_state.get("user_profile", create_default_user_profile())
    
    @staticmethod
    def update_user_profile(profile: UserProfile):
        """사용자 프로필 업데이트"""
        st.session_state.user_profile = profile
    
    @staticmethod
    def add_analysis_to_history(analysis_result: Dict):
        """분석 결과를 히스토리에 추가"""
        if "analysis_history" not in st.session_state:
            st.session_state.analysis_history = []
        
        st.session_state.analysis_history.append({
            "timestamp": datetime.now(),
            "result": analysis_result
        })
    
    @staticmethod
    def get_analysis_history() -> List[Dict]:
        """분석 히스토리 반환"""
        return st.session_state.get("analysis_history", [])


class BlindInsightApp:
    """
    BlindInsight AI 메인 애플리케이션 클래스
    
    Streamlit 기반의 웹 인터페이스를 제공하며,
    회사 분석, 커리어 상담, 데이터 시각화 기능을 통합합니다.
    """
    
    def __init__(self):
        """애플리케이션 초기화"""
        
        # 세션 관리자 초기화
        SessionManager.initialize_session()
        
        # 핵심 구성 요소들 (지연 초기화)
        self.knowledge_base: Optional[KnowledgeBase] = None
        self.workflow: Optional[BlindInsightWorkflow] = None
        self.mcp_client: Optional[MCPClient] = None
        
        # UI 상태
        self.current_page = "홈"
        
        logger.info("BlindInsight 애플리케이션 초기화")
    
    async def initialize_components(self):
        """핵심 구성 요소들 초기화 (비동기)"""
        
        if st.session_state.get("app_initialized", False):
            return
        
        try:
            with st.spinner("BlindInsight AI를 초기화하는 중..."):
                # 지식 베이스 초기화
                self.knowledge_base = KnowledgeBase()
                
                # MCP 클라이언트 초기화
                self.mcp_client = MCPClient(self.knowledge_base)
                
                # 워크플로우 초기화
                self.workflow = BlindInsightWorkflow()
                
                # 샘플 데이터 로드 (실제 환경에서는 실제 데이터 사용)
                await self._load_sample_data()
                
                st.session_state.app_initialized = True
                logger.info("핵심 구성 요소 초기화 완료")
                
        except Exception as e:
            st.error(f"초기화 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"초기화 오류: {str(e)}")
    
    async def _load_sample_data(self):
        """샘플 데이터 로드 (데모용)"""
        
        try:
            # 실제 환경에서는 MCP를 통해 실제 데이터를 로드합니다
            # 여기서는 데모용 샘플 데이터를 생성합니다
            
            sample_companies = ["네이버", "카카오", "쿠팡", "토스", "당근마켓"]
            
            from blindinsight.mcp.providers import BlindDataProvider, DataProviderConfig
            from blindinsight.mcp.handlers import DataHandler
            
            # 샘플 데이터 제공자 설정
            config = DataProviderConfig(
                name="sample_blind",
                base_url="https://api.blind.com",  # 샘플 URL
                enabled=True
            )
            
            provider = BlindDataProvider(config, self.mcp_client)
            handler = DataHandler(self.knowledge_base)
            
            # 각 회사별로 샘플 데이터 생성 및 저장
            for company in sample_companies[:2]:  # 처음 2개 회사만 로드 (시간 절약)
                sample_data = await provider.fetch_company_data(company)
                if sample_data:
                    await handler.process_provider_data("blind", sample_data)
            
            logger.info("샘플 데이터 로드 완료")
            
        except Exception as e:
            logger.error(f"샘플 데이터 로드 실패: {str(e)}")
    
    def run(self):
        """애플리케이션 실행"""
        
        # 페이지 설정
        st.set_page_config(
            page_title="BlindInsight AI",
            page_icon="🔍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 비동기 초기화 실행
        if not st.session_state.get("app_initialized", False):
            asyncio.run(self.initialize_components())
        
        # 메인 UI 렌더링
        self._render_header()
        self._render_sidebar()
        self._render_main_content()
    
    def _render_header(self):
        """헤더 영역 렌더링"""
        
        st.markdown("""
        <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;'>
            <h1 style='color: white; margin: 0;'>🔍 BlindInsight AI</h1>
            <p style='color: #f0f0f0; margin: 0.5rem 0 0 0;'>AI 기반 회사 분석 및 커리어 상담 플랫폼</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """사이드바 렌더링"""
        
        with st.sidebar:
            st.markdown("### 📋 메뉴")
            
            # 페이지 선택
            self.current_page = st.selectbox(
                "페이지 선택",
                ["홈", "회사 분석", "커리어 상담", "데이터 탐색", "설정"],
                index=0
            )
            
            st.markdown("---")
            
            # 사용자 프로필 간단 표시
            user_profile = SessionManager.get_user_profile()
            st.markdown("### 👤 사용자 정보")
            st.write(f"**이름**: {user_profile.name or '미설정'}")
            st.write(f"**경력**: {user_profile.experience_years}년")
            st.write(f"**관심 직무**: {user_profile.target_position or '미설정'}")
            
            st.markdown("---")
            
            # 최근 분석 결과
            history = SessionManager.get_analysis_history()
            if history:
                st.markdown("### 📊 최근 분석")
                for i, item in enumerate(history[-3:]):  # 최근 3개만 표시
                    company = item.get("result", {}).get("company_name", "알 수 없음")
                    timestamp = item["timestamp"].strftime("%m/%d %H:%M")
                    st.write(f"• {company} ({timestamp})")
            
            st.markdown("---")
            
            # 시스템 상태 (개발자용)
            if st.checkbox("시스템 상태 표시"):
                if self.knowledge_base:
                    stats = self.knowledge_base.get_statistics()
                    st.json(stats)
    
    def _render_main_content(self):
        """메인 콘텐츠 영역 렌더링"""
        
        if self.current_page == "홈":
            self._render_home_page()
        elif self.current_page == "회사 분석":
            self._render_company_analysis_page()
        elif self.current_page == "커리어 상담":
            self._render_career_consultation_page()
        elif self.current_page == "데이터 탐색":
            self._render_data_exploration_page()
        elif self.current_page == "설정":
            self._render_settings_page()
    
    def _render_home_page(self):
        """홈 페이지 렌더링"""
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("## 🎯 BlindInsight AI에 오신 것을 환영합니다!")
            
            st.markdown("""
            **BlindInsight AI**는 익명 직장 리뷰 데이터를 분석하여 
            데이터 기반의 커리어 인사이트를 제공하는 AI 플랫폼입니다.
            
            ### 🚀 주요 기능
            
            #### 🏢 회사 분석
            - **문화 분석**: 실제 직장인들의 솔직한 회사 문화 리뷰 분석
            - **연봉 정보**: 포지션별, 경력별 실제 연봉 데이터 제공
            - **성장 기회**: 회사의 안정성과 성장 가능성 분석
            - **면접 정보**: 실제 면접 후기와 준비 팁 제공
            
            #### 💼 개인화된 커리어 상담
            - **맞춤형 추천**: 개인 프로필 기반 최적 회사 추천
            - **스킬 갭 분석**: 목표 포지션을 위한 필요 역량 분석
            - **커리어 로드맵**: 단계별 경력 발전 전략 제시
            - **시장 동향 분석**: 업계 트렌드와 기회 분석
            """)
        
        with col2:
            st.markdown("### 📈 시스템 현황")
            
            # 시스템 통계 (샘플 데이터)
            if self.knowledge_base:
                stats = self.knowledge_base.get_statistics()
                
                # 메트릭 카드들
                st.metric("분석 가능한 회사", len(st.session_state.get("selected_companies", [])) + 50)
                st.metric("수집된 리뷰", stats.get("knowledge_base", {}).get("total_documents", 0))
                st.metric("처리된 분석 요청", stats.get("performance", {}).get("search_queries", 0))
            
            st.markdown("---")
            
            # 빠른 시작 버튼들
            st.markdown("### 🎬 빠른 시작")
            
            if st.button("🔍 회사 분석 시작", use_container_width=True):
                st.session_state.current_page = "회사 분석"
                st.rerun()
            
            if st.button("💬 커리어 상담 시작", use_container_width=True):
                st.session_state.current_page = "커리어 상담"
                st.rerun()
            
            if st.button("📊 데이터 탐색", use_container_width=True):
                st.session_state.current_page = "데이터 탐색"
                st.rerun()
    
    def _render_company_analysis_page(self):
        """회사 분석 페이지 렌더링 - 카카오톡 사진과 같은 3탭 구조"""
        
        st.markdown("## 🏢 회사 분석")
        
        # 회사 검색 인터페이스
        company_name = st.text_input(
            "분석할 회사명을 입력하세요",
            placeholder="예: 금호석유화학, 네이버, 카카오, 쿠팡...",
            key="company_search"
        )
        
        if company_name:
            # 회사가 선택되면 3개 탭 표시 (사진과 동일한 구조)
            tab1, tab2, tab3 = st.tabs(["📋 커뮤니티", "ℹ️ 회사 정보", "🤖 AI 분석"])
            
            with tab1:
                self._render_community_tab(company_name)
            
            with tab2:
                self._render_company_info_tab(company_name)
            
            with tab3:
                self._render_ai_analysis_tab(company_name)
        else:
            # 회사가 선택되지 않았을 때 안내 메시지
            st.info("🏢 분석하고 싶은 회사명을 입력해주세요. AI가 해당 회사에 대한 종합적인 분석을 제공합니다.")
    
    def _render_community_tab(self, company_name: str):
        """1번 탭: 커뮤니티 글 모음"""
        st.markdown(f"### 📋 {company_name} 커뮤니티 글")
        
        # 샘플 커뮤니티 데이터
        community_posts = [
            {
                "title": f"{company_name} 울산 보건관리자 채용 문의",
                "content": "현직들 울산공장 보건관리자 경력계약직 채용있었는데 육휴대체 자리일까요? 처우는 어떨까요?",
                "likes": 3,
                "comments": 1,
                "views": 56,
                "time": "3일전"
            },
            {
                "title": f"{company_name} 부산",
                "content": "안녕하세요 금호석유화학 부산 ts 포지션 지원받았습니다 대략 연봉 및 분위기 알려주시면감사하겠습니다",
                "likes": 1,
                "comments": 1,
                "views": 341,
                "time": "1주전"
            }
        ]
        
        for post in community_posts:
            with st.expander(f"💬 {post['title']}"):
                st.write(post['content'])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"❤️ {post['likes']}")
                with col2:
                    st.write(f"💬 {post['comments']}")
                with col3:
                    st.write(f"👀 {post['views']}")
                with col4:
                    st.write(f"⏰ {post['time']}")
    
    def _render_company_info_tab(self, company_name: str):
        """2번 탭: 회사 정보 및 리뷰"""
        st.markdown(f"### ℹ️ {company_name} 회사 정보")
        
        # 기본 회사 정보
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 기본 정보")
            st.write("**업종**: 화학")
            st.write("**설립연도**: 1976년")
            st.write("**직원수**: 약 3,000명")
            st.write("**본사위치**: 서울특별시")
            
        with col2:
            st.markdown("#### ⭐ 평점 현황")
            st.metric("전체 평점", "3.2/5.0")
            st.metric("급여 만족도", "3.1/5.0")
            st.metric("복지 만족도", "3.0/5.0")
            st.metric("워라밸", "3.5/5.0")
        
        # 최근 리뷰
        st.markdown("#### 📝 최근 리뷰")
        reviews = [
            {"rating": 4, "title": "안정적인 대기업", "content": "화학업계의 안정적인 대기업이라 복리후생이 좋습니다."},
            {"rating": 3, "title": "전통적인 조직문화", "content": "보수적인 조직문화가 있지만 안정성은 높습니다."},
            {"rating": 3, "title": "워라밸 괜찮음", "content": "야근은 거의 없고 워라밸은 괜찮은 편입니다."}
        ]
        
        for review in reviews:
            with st.container():
                st.write(f"⭐ {review['rating']}/5 - **{review['title']}**")
                st.write(review['content'])
                st.divider()
    
    def _render_ai_analysis_tab(self, company_name: str):
        """3번 탭: AI 분석 기능 (핵심 기능)"""
        st.markdown(f"### 🤖 {company_name} AI 분석")
        
        # AI 분석 옵션 선택
        analysis_options = st.multiselect(
            "분석 항목을 선택하세요:",
            ["문화 분석", "연봉 분석", "성장성 분석", "커리어 분석", "리스크 분석"],
            default=["문화 분석", "연봉 분석"]
        )
        
        # AI 분석 실행 버튼
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("🔍 AI로 분석하기", type="primary", use_container_width=True):
                if analysis_options:
                    with st.spinner(f"🤖 AI가 {company_name}에 대해 분석중입니다..."):
                        # 여기서 RAG + MCP 연동하여 실제 분석 수행
                        analysis_result = asyncio.run(self._perform_ai_analysis(company_name, analysis_options))
                        if analysis_result:
                            st.session_state.ai_analysis_result = analysis_result
                            st.success("분석 완료!")
                            st.rerun()
                else:
                    st.warning("분석 항목을 선택해주세요.")
        
        with col2:
            st.info("🔍 AI가 커뮤니티 글, 리뷰, 뉴스 등을 종합 분석하여 인사이트를 제공합니다.")
        
        # 분석 결과 표시
        if st.session_state.get("ai_analysis_result"):
            self._display_ai_analysis_result(st.session_state.ai_analysis_result)
    
    async def _perform_ai_analysis(self, company_name: str, analysis_options: List[str]) -> Dict:
        """AI 분석 실행 (RAG + MCP 연동)"""
        try:
            # 1. RAG로 관련 데이터 검색
            if self.knowledge_base:
                # 회사 관련 문서 검색
                search_results = await self.knowledge_base.search(
                    query=f"{company_name} 회사 리뷰 문화 연봉",
                    company_name=company_name,
                    k=10
                )
            else:
                search_results = []
            
            # 2. MCP를 통한 외부 데이터 수집
            mcp_data = await self._collect_mcp_data(company_name)
            
            # 3. AI 에이전트들을 통한 분석
            analysis_result = await self._run_analysis_agents(
                company_name, analysis_options, search_results, mcp_data
            )
            
            return analysis_result
            
        except Exception as e:
            st.error(f"AI 분석 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"AI 분석 오류: {str(e)}")
            # 오류 발생 시 샘플 데이터 반환
            return self._get_sample_analysis_result(company_name, analysis_options)
    
    async def _collect_mcp_data(self, company_name: str) -> Dict:
        """MCP를 통한 외부 데이터 수집"""
        try:
            mcp_data = {}
            
            if self.mcp_client:
                # 각 MCP 프로바이더를 통해 데이터 수집
                providers_data = {
                    'blind': f"{company_name} 리뷰 데이터",
                    'job_sites': f"{company_name} 채용 정보",
                    'salary': f"{company_name} 연봉 정보",
                    'news': f"{company_name} 뉴스"
                }
                mcp_data = providers_data
            
            return mcp_data
            
        except Exception as e:
            logger.error(f"MCP 데이터 수집 실패: {str(e)}")
            return {}
    
    async def _run_analysis_agents(self, company_name: str, analysis_options: List[str], 
                                 search_results: List, mcp_data: Dict) -> Dict:
        """분석 에이전트들 실행"""
        try:
            # 실제로는 각 에이전트를 실행하지만, 현재는 샘플 데이터 반환
            return self._get_sample_analysis_result(company_name, analysis_options)
            
        except Exception as e:
            logger.error(f"분석 에이전트 실행 실패: {str(e)}")
            return self._get_sample_analysis_result(company_name, analysis_options)
    
    def _get_sample_analysis_result(self, company_name: str, analysis_options: List[str]) -> Dict:
        """샘플 분석 결과 생성"""
        analyses = {}
        
        if "문화 분석" in analysis_options:
            analyses["culture"] = {
                "summary": f"{company_name}는 전통적이고 안정적인 조직문화를 가지고 있습니다. 보수적이지만 안정성을 중시하는 문화입니다.",
                "strengths": ["안정성", "복리후생", "워라밸", "대기업 시스템"],
                "weaknesses": ["보수적 문화", "혁신 부족", "변화 속도 느림"],
                "score": 3.2
            }
        
        if "연봉 분석" in analysis_options:
            analyses["compensation"] = {
                "summary": f"{company_name}의 연봉은 업계 평균 수준이며, 복리후생이 좋은 편입니다.",
                "average_salary": {"신입": 4200, "경력 3년": 5500, "경력 5년": 7000, "경력 10년": 9500},
                "benefits": ["4대보험", "퇴직금", "성과급", "주택자금대출", "건강검진"],
                "score": 3.1
            }
        
        if "성장성 분석" in analysis_options:
            analyses["growth"] = {
                "summary": f"{company_name}는 안정적이지만 성장성은 보통 수준입니다.",
                "market_position": "안정적",
                "future_outlook": "보통",
                "growth_factors": ["안정적 사업구조", "글로벌 시장 진출"],
                "risks": ["화학업계 변화", "환경 규제 강화"],
                "score": 3.0
            }
        
        if "커리어 분석" in analysis_options:
            analyses["career"] = {
                "summary": f"{company_name}는 안정적인 커리어 패스를 제공하지만 빠른 성장은 제한적입니다.",
                "fit_score": 75,
                "recommended_actions": [
                    "화학 관련 전문 지식 습득",
                    "프로세스 최적화 경험 쌓기",
                    "품질 관리 역량 강화"
                ],
                "career_prospects": "안정적이지만 보수적",
                "score": 3.3
            }
        
        return {
            "company_name": company_name,
            "analysis_timestamp": datetime.now(),
            "analyses": analyses
        }
    
    def _display_ai_analysis_result(self, result: Dict):
        """AI 분석 결과 시각화"""
        st.markdown("### 📊 AI 분석 결과")
        
        company_name = result.get("company_name", "알 수 없음")
        analyses = result.get("analyses", {})
        
        # 종합 점수 표시
        if analyses:
            scores = [analysis.get("score", 3.0) for analysis in analyses.values() if "score" in analysis]
            if scores:
                avg_score = sum(scores) / len(scores)
                st.metric("🎯 종합 점수", f"{avg_score:.1f}/5.0")
        
        # 각 분석 결과 표시
        for analysis_type, analysis_data in analyses.items():
            if analysis_type == "culture":
                self._display_culture_analysis(analysis_data)
            elif analysis_type == "compensation":
                self._display_compensation_analysis(analysis_data)
            elif analysis_type == "growth":
                self._display_growth_analysis_ai(analysis_data)
            elif analysis_type == "career":
                self._display_career_analysis_ai(analysis_data)
    
    def _display_culture_analysis(self, data: Dict):
        """문화 분석 결과 표시"""
        with st.expander("🏢 문화 분석", expanded=True):
            st.write(f"**요약**: {data.get('summary', '')}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ 강점**")
                for strength in data.get('strengths', []):
                    st.write(f"• {strength}")
            
            with col2:
                st.markdown("**⚠️ 약점**")
                for weakness in data.get('weaknesses', []):
                    st.write(f"• {weakness}")
            
            st.metric("문화 점수", f"{data.get('score', 3.0):.1f}/5.0")
    
    def _display_compensation_analysis(self, data: Dict):
        """연봉 분석 결과 표시"""
        with st.expander("💰 연봉 분석", expanded=True):
            st.write(f"**요약**: {data.get('summary', '')}")
            
            # 연봉 차트
            salary_data = data.get('average_salary', {})
            if salary_data:
                positions = list(salary_data.keys())
                salaries = list(salary_data.values())
                
                fig = px.bar(
                    x=positions,
                    y=salaries,
                    title="포지션별 평균 연봉 (만원)",
                    labels={"x": "포지션", "y": "연봉 (만원)"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 복리후생
            benefits = data.get('benefits', [])
            if benefits:
                st.markdown("**🎁 주요 복리후생**")
                for benefit in benefits:
                    st.write(f"• {benefit}")
            
            st.metric("연봉 만족도", f"{data.get('score', 3.0):.1f}/5.0")
    
    def _display_growth_analysis_ai(self, data: Dict):
        """성장성 분석 결과 표시 (AI 버전)"""
        with st.expander("📈 성장성 분석", expanded=True):
            st.write(f"**요약**: {data.get('summary', '')}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**시장 위치**: {data.get('market_position', '보통')}")
                st.markdown("**📈 성장 요인**")
                for factor in data.get('growth_factors', []):
                    st.write(f"• {factor}")
            with col2:
                st.write(f"**미래 전망**: {data.get('future_outlook', '보통')}")
                st.markdown("**⚠️ 리스크 요인**")
                for risk in data.get('risks', []):
                    st.write(f"• {risk}")
            
            st.metric("성장성 점수", f"{data.get('score', 3.0):.1f}/5.0")
    
    def _display_career_analysis_ai(self, data: Dict):
        """커리어 분석 결과 표시 (AI 버전)"""
        with st.expander("🎯 커리어 분석", expanded=True):
            st.write(f"**요약**: {data.get('summary', '')}")
            
            # 개인화된 적합도
            fit_score = data.get('fit_score', 70)
            st.progress(fit_score / 100)
            st.write(f"**개인 적합도**: {fit_score}%")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**커리어 전망**: {data.get('career_prospects', '보통')}")
            with col2:
                st.metric("커리어 점수", f"{data.get('score', 3.0):.1f}/5.0")
            
            # 추천 액션
            actions = data.get('recommended_actions', [])
            if actions:
                st.markdown("**🚀 추천 액션**")
                for action in actions:
                    st.write(f"• {action}")

    async def _perform_company_analysis(self, company_name: str, analysis_type: str) -> Optional[Dict]:
        """회사 분석 실행"""
        
        try:
            if not self.workflow or not self.knowledge_base:
                st.error("시스템이 초기화되지 않았습니다.")
                return None
            
            # 분석 요청 생성
            user_profile = SessionManager.get_user_profile()
            
            analysis_request = AnalysisRequest(
                user_profile=user_profile,
                company_name=company_name,
                analysis_type=analysis_type.replace(" 분석", "").lower(),
                preferences={
                    "include_salary_info": True,
                    "include_culture_analysis": True,
                    "include_growth_analysis": True,
                    "include_career_path": True
                }
            )
            
            # 워크플로우 실행
            result = await self.workflow.run_analysis(analysis_request)
            
            # 세션에 저장
            st.session_state.current_analysis = result
            
            return result
            
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"회사 분석 오류: {str(e)}")
            return None
    
    def _display_analysis_result(self, result: Dict):
        """분석 결과 표시"""
        
        st.markdown("### 📊 분석 결과")
        
        company_name = result.get("company_name", "알 수 없음")
        
        # 결과 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs(["🏢 종합", "💰 연봉", "📈 성장성", "🎯 커리어"])
        
        with tab1:
            self._display_comprehensive_analysis(result)
        
        with tab2:
            self._display_salary_analysis(result)
        
        with tab3:
            self._display_growth_analysis(result)
        
        with tab4:
            self._display_career_analysis(result)
    
    def _display_comprehensive_analysis(self, result: Dict):
        """종합 분석 결과 표시"""
        
        # TODO: 실제 분석 결과를 표시하는 로직 구현
        # 현재는 샘플 데이터로 시연
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("전체 평점", "4.2/5.0", "0.3")
        
        with col2:
            st.metric("추천도", "78%", "5%")
        
        with col3:
            st.metric("성장성", "높음", "상승")
        
        st.markdown("#### 📋 주요 특징")
        
        # 샘플 인사이트
        insights = [
            "🎯 **워라밸**: 직원들이 가장 만족하는 부분으로 자유로운 근무 환경을 제공",
            "💼 **업무 환경**: 최신 기술 스택 사용과 자율적인 프로젝트 진행 방식",
            "👥 **동료 관계**: 수평적 소통 문화와 적극적인 지식 공유 분위기",
            "📈 **성장 기회**: 다양한 프로젝트 참여 기회와 교육 지원 프로그램 운영"
        ]
        
        for insight in insights:
            st.markdown(insight)
    
    def _display_salary_analysis(self, result: Dict):
        """연봉 분석 결과 표시"""
        
        # 샘플 연봉 데이터 생성
        positions = ["신입", "주니어", "시니어", "리드", "매니저"]
        salaries = [45000000, 65000000, 85000000, 120000000, 150000000]
        
        # 연봉 차트
        fig = px.bar(
            x=positions,
            y=salaries,
            title="포지션별 평균 연봉",
            labels={"x": "포지션", "y": "연봉 (원)"},
            color=salaries,
            color_continuous_scale="viridis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 연봉 상세 정보
        st.markdown("#### 💰 연봉 상세 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 경쟁력 분석**
            - 업계 평균 대비: **상위 30%**
            - 복리후생 점수: **4.1/5.0**
            - 성과급 지급률: **85%**
            """)
        
        with col2:
            st.markdown("""
            **📊 추가 혜택**
            - 스톡옵션: **제공**
            - 교육비 지원: **연간 200만원**
            - 건강검진: **종합검진 제공**
            """)
    
    def _display_growth_analysis(self, result: Dict):
        """성장성 분석 결과 표시"""
        
        # 성장성 지표 차트
        categories = ["시장성", "기술력", "재무안정성", "조직문화", "혁신성"]
        values = [85, 92, 78, 88, 95]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='회사 성장성'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="성장성 분석 레이더 차트"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 📈 성장성 상세 분석")
        
        analysis_points = [
            "**시장 지위**: 해당 분야에서 선도적 위치 확보",
            "**기술 혁신**: 지속적인 R&D 투자와 신기술 도입",
            "**조직 확장**: 전년 대비 20% 인력 증가",
            "**재무 건전성**: 안정적인 매출 성장과 수익성 개선"
        ]
        
        for point in analysis_points:
            st.markdown(f"• {point}")
    
    def _display_career_analysis(self, result: Dict):
        """커리어 분석 결과 표시"""
        
        st.markdown("#### 🎯 개인화된 커리어 분석")
        
        user_profile = SessionManager.get_user_profile()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 적합도 분석**")
            
            # 적합도 점수 (샘플)
            fit_score = 82
            st.progress(fit_score / 100)
            st.write(f"전체 적합도: **{fit_score}%**")
            
            st.markdown("**📋 매칭 포인트**")
            match_points = [
                "✅ 기술 스택 일치도: 높음",
                "✅ 경력 수준 적합: 매우 적합",
                "⚠️ 관심 도메인: 부분 일치",
                "✅ 성장 방향성: 일치"
            ]
            
            for point in match_points:
                st.markdown(point)
        
        with col2:
            st.markdown("**🚀 추천 액션 플랜**")
            
            action_items = [
                "**단기 (1-3개월)**",
                "• 해당 회사 기술 블로그 구독",
                "• 관련 오픈소스 프로젝트 참여",
                "",
                "**중기 (3-6개월)**", 
                "• 포트폴리오 업데이트 및 보완",
                "• 네트워킹 이벤트 참가",
                "",
                "**장기 (6개월+)**",
                "• 면접 준비 및 지원",
                "• 스킬 인증 획득"
            ]
            
            for item in action_items:
                if item.startswith("**"):
                    st.markdown(item)
                else:
                    st.markdown(item)
    
    def _render_career_consultation_page(self):
        """커리어 상담 페이지 렌더링"""
        
        st.markdown("## 💼 AI 커리어 상담")
        
        # 채팅 인터페이스
        self._render_chat_interface()
    
    def _render_chat_interface(self):
        """채팅 인터페이스 렌더링"""
        
        # 채팅 히스토리 표시
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # 채팅 메시지 컨테이너
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # 사용자 입력
        user_input = st.chat_input("커리어 관련 질문을 입력하세요...")
        
        if user_input:
            # 사용자 메시지 추가
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input
            })
            
            # AI 응답 생성 (시뮬레이션)
            ai_response = self._generate_ai_response(user_input)
            
            # AI 응답 추가
            st.session_state.chat_messages.append({
                "role": "assistant", 
                "content": ai_response
            })
            
            st.rerun()
    
    def _generate_ai_response(self, user_input: str) -> str:
        """AI 응답 생성 (시뮬레이션)"""
        
        # 실제 환경에서는 LLM API를 호출하여 응답 생성
        # 여기서는 간단한 키워드 기반 응답 제공
        
        user_input_lower = user_input.lower()
        
        if "연봉" in user_input_lower or "급여" in user_input_lower:
            return """
**💰 연봉 관련 조언**

연봉 협상에서 중요한 포인트들을 알려드릴게요:

1. **시장 조사**: 같은 포지션의 업계 평균 연봉을 파악하세요
2. **성과 어필**: 구체적인 성과와 기여도를 수치로 제시하세요  
3. **타이밍**: 성과 평가 시기나 프로젝트 완료 후가 좋습니다
4. **총 보상**: 기본급 외에 복리후생, 스톡옵션도 고려하세요

더 구체적인 상황을 알려주시면 맞춤형 조언을 드릴 수 있어요!
            """
        
        elif "이직" in user_input_lower or "전직" in user_input_lower:
            return """
**🚀 이직 전략 가이드**

성공적인 이직을 위한 단계별 가이드입니다:

**준비 단계 (1-2개월)**
- 포트폴리오 업데이트
- 이력서 및 자기소개서 작성
- 목표 회사 리스트 작성

**실행 단계 (1-3개월)**  
- 적극적인 지원 및 네트워킹
- 면접 준비 및 실전 연습
- 레퍼런스 확보

**마무리 단계**
- 처우 협상
- 원만한 퇴사 진행

어떤 단계에서 도움이 필요하신지 알려주세요!
            """
        
        elif "면접" in user_input_lower:
            return """
**🎯 면접 준비 가이드**

면접 성공을 위한 핵심 팁들입니다:

**기술 면접 준비**
- 기본 CS 지식 복습
- 코딩 테스트 연습 (알고리즘, 자료구조)
- 프로젝트 경험 정리 및 심화 질문 대비

**인성 면접 준비**
- 회사 및 직무 연구
- STAR 기법으로 경험 정리
- 역질문 준비

**실전 팁**
- 모의 면접 연습
- 복장 및 태도 점검
- 면접 후 감사 메일 발송

특정 회사나 포지션에 대한 면접 정보가 필요하시면 알려주세요!
            """
        
        else:
            return """
안녕하세요! BlindInsight AI 커리어 상담사입니다. 🤖

다음과 같은 주제로 도움을 드릴 수 있습니다:

• **연봉 협상** 및 보상 패키지 분석
• **이직 전략** 및 커리어 로드맵 설계  
• **면접 준비** 및 회사별 면접 정보
• **스킬 개발** 방향 및 학습 계획
• **회사 문화** 및 조직 분석

구체적인 질문이나 상황을 알려주시면 더 정확한 조언을 드릴 수 있어요!
            """
    
    def _render_data_exploration_page(self):
        """데이터 탐색 페이지 렌더링"""
        
        st.markdown("## 📊 데이터 탐색")
        
        if not self.knowledge_base:
            st.warning("지식 베이스가 초기화되지 않았습니다.")
            return
        
        # 검색 인터페이스
        search_query = st.text_input("검색어를 입력하세요", placeholder="예: 워라밸, 연봉, 승진...")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            company_filter = st.text_input("회사명 필터", placeholder="특정 회사만 검색")
        
        with col2:
            category_filter = st.selectbox("카테고리", ["전체", "문화", "연봉", "커리어", "면접"])
        
        with col3:
            result_count = st.slider("결과 개수", 5, 50, 10)
        
        # 검색 실행
        if st.button("🔍 검색") and search_query:
            with st.spinner("검색 중..."):
                results = asyncio.run(self._perform_search(
                    search_query, company_filter, category_filter, result_count
                ))
                
                if results:
                    self._display_search_results(results)
    
    async def _perform_search(self, query: str, company: str, category: str, count: int):
        """검색 실행"""
        
        try:
            if not self.knowledge_base:
                return []
            
            # 카테고리 매핑
            category_map = {
                "전체": None,
                "문화": "culture",
                "연봉": "salary", 
                "커리어": "career",
                "면접": "interview"
            }
            
            results = await self.knowledge_base.search(
                query=query,
                company_name=company if company else None,
                category=category_map.get(category),
                k=count
            )
            
            return results
            
        except Exception as e:
            st.error(f"검색 중 오류: {str(e)}")
            return []
    
    def _display_search_results(self, results: List):
        """검색 결과 표시"""
        
        st.markdown(f"### 🔍 검색 결과 ({len(results)}개)")
        
        for i, result in enumerate(results):
            with st.expander(f"결과 {i+1} - 유사도: {result.score:.2f}"):
                
                # 메타데이터 표시
                metadata = result.metadata
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**회사**: {metadata.get('company_name', '알 수 없음')}")
                with col2:
                    st.write(f"**카테고리**: {metadata.get('category', '일반')}")
                with col3:
                    st.write(f"**날짜**: {metadata.get('created_at', '알 수 없음')}")
                
                # 내용 표시
                st.markdown("**내용:**")
                st.write(result.content)
    
    def _render_settings_page(self):
        """설정 페이지 렌더링"""
        
        st.markdown("## ⚙️ 설정")
        
        # 사용자 프로필 편집
        self._render_user_profile_editor()
        
        st.markdown("---")
        
        # 시스템 설정
        self._render_system_settings()
    
    def _render_user_profile_editor(self):
        """사용자 프로필 편집기 렌더링"""
        
        st.markdown("### 👤 사용자 프로필")
        
        current_profile = SessionManager.get_user_profile()
        
        with st.form("user_profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("이름", value=current_profile.name or "")
                experience_years = st.number_input(
                    "경력 연수", 
                    min_value=0, 
                    max_value=50, 
                    value=current_profile.experience_years
                )
                education = st.selectbox(
                    "학력",
                    ["고등학교", "전문대", "대학교", "대학원"],
                    index=2 if current_profile.education_level.value == "bachelor" else 0
                )
            
            with col2:
                current_position = st.text_input(
                    "현재 직책", 
                    value=current_profile.current_position or ""
                )
                current_company = st.text_input(
                    "현재 회사", 
                    value=current_profile.current_company or ""
                )
                location = st.text_input(
                    "지역", 
                    value=current_profile.location_preferences[0].value if current_profile.location_preferences else ""
                )
            
            # 관심 분야
            st.markdown("**관심 분야 (쉼표로 구분)**")
            interests = st.text_area(
                "관심 분야",
                value=current_profile.target_position or "",
                height=100
            )
            
            # 저장 버튼
            if st.form_submit_button("💾 프로필 저장"):
                # 프로필 업데이트 (실제 구현에서는 UserProfile 객체 생성)
                st.success("프로필이 저장되었습니다!")
    
    def _render_system_settings(self):
        """시스템 설정 렌더링"""
        
        st.markdown("### 🔧 시스템 설정")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**데이터 소스 설정**")
            
            # 데이터 소스 활성화/비활성화
            enable_blind = st.checkbox("Blind 데이터", value=True)
            enable_jobsites = st.checkbox("채용 사이트 데이터", value=True)
            enable_news = st.checkbox("뉴스 데이터", value=False)
            
            if st.button("🔄 데이터 새로고침"):
                with st.spinner("데이터를 새로고침하는 중..."):
                    # 실제로는 MCP를 통해 데이터 동기화
                    asyncio.run(asyncio.sleep(2))  # 시뮬레이션
                    st.success("데이터 새로고침 완료!")
        
        with col2:
            st.markdown("**분석 설정**")
            
            # 분석 관련 설정
            analysis_depth = st.selectbox(
                "분석 깊이",
                ["빠른 분석", "상세 분석", "전문가 분석"],
                index=1
            )
            
            include_sentiment = st.checkbox("감정 분석 포함", value=True)
            include_trends = st.checkbox("트렌드 분석 포함", value=True)
            
            if st.button("💾 설정 저장"):
                st.success("설정이 저장되었습니다!")
        
        # 시스템 통계
        st.markdown("---")
        st.markdown("### 📊 시스템 통계")
        
        if self.knowledge_base:
            stats = self.knowledge_base.get_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "저장된 문서", 
                    stats.get("knowledge_base", {}).get("total_documents", 0)
                )
            
            with col2:
                st.metric(
                    "검색 요청", 
                    stats.get("performance", {}).get("search_queries", 0)
                )
            
            with col3:
                cache_stats = stats.get("embedding_cache", {})
                hit_rate = cache_stats.get("hit_rate", 0) * 100
                st.metric("캐시 적중률", f"{hit_rate:.1f}%")
            
            with col4:
                st.metric("평균 응답 시간", "0.8초")


def run_app():
    """애플리케이션 실행 함수"""
    
    try:
        app = BlindInsightApp()
        app.run()
        
    except Exception as e:
        st.error(f"애플리케이션 실행 중 오류가 발생했습니다: {str(e)}")
        logger.error(f"앱 실행 오류: {str(e)}")


if __name__ == "__main__":
    run_app()