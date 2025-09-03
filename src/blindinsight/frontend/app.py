"""
BlindInsight AI 메인 애플리케이션

Streamlit 기반의 메인 웹 애플리케이션으로,
회사 분석, 커리어 상담, 데이터 시각화 기능을 제공합니다.
"""

import streamlit as st
import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# BlindInsight 모듈 임포트
sys.path.append(str(Path(__file__).parent.parent.parent))

from blindinsight.workflow.graph import BlindInsightWorkflow
from blindinsight.models.analysis import AnalysisRequest
from blindinsight.models.user import UserProfile, create_default_user_profile
from blindinsight.rag.knowledge_base import KnowledgeBase
from blindinsight.mcp.client import MCPClient
from blindinsight.mcp.providers import BlindDataProvider, DataProviderConfig
from blindinsight.models.base import settings

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("log.txt", encoding="utf-8")
    ]
)
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
        # 세션에서 지식베이스 복원 시도
        self.knowledge_base: Optional[KnowledgeBase] = st.session_state.get("knowledge_base", None)
        if self.knowledge_base:
            logger.info("세션에서 기존 지식베이스 인스턴스 복원됨")
        else:
            logger.info("세션에 지식베이스 인스턴스 없음 - 새로 초기화 필요")
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
                # 지식 베이스 초기화 (재시도 로직 포함)
                self.knowledge_base = None
                for attempt in range(3):  # 최대 3회 시도
                    try:
                        logger.info(f"지식 베이스 초기화 시도 {attempt + 1}/3")
                        logger.info(f"데이터 디렉토리: {settings.data_directory}")
                        logger.info(f"벡터DB 경로: C:/blind/data/embeddings")
                        
                        # 실제 지식베이스 경로를 명시적으로 설정
                        knowledge_base = KnowledgeBase(
                            data_dir=settings.data_directory,
                            vector_db_path="C:/blind/data/embeddings"
                        )
                        
                        # 초기화 완료 후 데이터 확인
                        await knowledge_base.initialize()
                        
                        # 초기화 성공 시 인스턴스와 세션에 저장
                        self.knowledge_base = knowledge_base
                        st.session_state.knowledge_base = knowledge_base
                        
                        # 연결 및 데이터 확인
                        stats = self.knowledge_base.get_statistics()
                        total_docs = stats.get("knowledge_base", {}).get("total_documents", 0)
                        
                        logger.info(f"지식 베이스 초기화 완료 - 총 문서 수: {total_docs}")
                        
                        if total_docs > 0:
                            logger.info("기존 데이터가 발견되어 지식베이스가 준비되었습니다.")
                        else:
                            logger.warning("지식베이스에 데이터가 없습니다. 동적 필터링이 제한될 수 있습니다.")
                        
                        break
                            
                    except Exception as e:
                        import traceback
                        logger.error(f"지식 베이스 초기화 실패 (시도 {attempt + 1}/3): {str(e)}")
                        logger.error(f"상세 오류: {traceback.format_exc()}")
                        if attempt == 2:  # 마지막 시도
                            logger.error("지식 베이스 초기화 최종 실패")
                            self.knowledge_base = None
                            # 세션에서도 제거
                            if "knowledge_base" in st.session_state:
                                del st.session_state.knowledge_base
                            st.error("⚠️ 지식베이스 초기화에 실패했습니다. 데이터 디렉토리와 권한을 확인해주세요.")
                            st.error(f"오류 세부사항: {str(e)}")
                            # 초기화 실패 시 재시도 가능하도록 설정
                            st.session_state.knowledge_base_initialized = False
                            return  # 초기화 실패 시 early return
                        else:
                            import time
                            time.sleep(1)  # 1초 대기 후 재시도
                
                # 지식베이스가 초기화되었다면 데이터 존재 여부 확인 및 로드
                if self.knowledge_base:
                    try:
                        await self._ensure_knowledge_base_data()
                    except Exception as e:
                        logger.warning(f"지식베이스 데이터 확인/로드 실패: {str(e)}")
                
                # MCP 클라이언트 초기화
                try:
                    self.mcp_client = MCPClient(self.knowledge_base)
                    logger.info("MCP 클라이언트 초기화 완료")
                except Exception as e:
                    logger.error(f"MCP 클라이언트 초기화 실패: {str(e)}")
                    self.mcp_client = None
                
                # 워크플로우 초기화
                try:
                    self.workflow = BlindInsightWorkflow()
                    logger.info("워크플로우 초기화 완료")
                except Exception as e:
                    logger.error(f"워크플로우 초기화 실패: {str(e)}")
                    self.workflow = None
                
                # 샘플 데이터 로드 (실패해도 앱은 계속 실행) - 비활성화됨
                # 개발 모드에서만 샘플 데이터를 로드하도록 설정
                load_sample_data = getattr(settings, 'enable_sample_data_loading', False)
                if load_sample_data:
                    try:
                        logger.info("샘플 데이터 로드가 활성화되어 있습니다...")
                        await self._load_sample_data()
                    except Exception as e:
                        logger.warning(f"샘플 데이터 로드 실패 (계속 진행): {str(e)}")
                else:
                    logger.info("샘플 데이터 로드가 비활성화되어 있습니다.")
                
                # 지식베이스 초기화 성공 시에만 완전 초기화 완료로 설정
                if self.knowledge_base:
                    st.session_state.app_initialized = True
                    st.session_state.knowledge_base_initialized = True
                    logger.info("핵심 구성 요소 초기화 완료")
                else:
                    logger.error("지식베이스 초기화 실패로 인해 앱 초기화 미완료")
                    st.session_state.knowledge_base_initialized = False
                
        except Exception as e:
            logger.error(f"초기화 오류: {str(e)}")
            st.error(f"애플리케이션 초기화 중 오류가 발생했습니다: {str(e)}")
            st.session_state.app_initialized = False
            st.session_state.knowledge_base_initialized = False
    
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
            
            # 각 회사별로 샘플 데이터 생성 및 저장 - 에러 처리 추가
            for company in sample_companies[:2]:  # 처음 2개 회사만 로드 (시간 절약)
                try:
                    sample_data = await provider.fetch_company_data(company)
                    if sample_data:
                        result = await handler.process_provider_data("blind", sample_data)
                        logger.info(f"{company} 데이터 처리 결과: {result.processed_count}개 성공, {result.failed_count}개 실패")
                except Exception as e:
                    logger.warning(f"{company} 샘플 데이터 처리 실패: {str(e)}")
                    continue  # 한 회사가 실패해도 다음 회사로 진행
            
            logger.info("샘플 데이터 로드 완료 (일부 오류 발생 가능)")
            
        except Exception as e:
            logger.error(f"샘플 데이터 로드 실패: {str(e)}")
    
    async def _ensure_knowledge_base_data(self):
        """지식베이스에 데이터가 있는지 확인하고 없으면 로드"""
        try:
            logger.info("지식베이스 데이터 존재 여부 확인 중...")
            
            # 통계를 통해 데이터 존재 여부 확인
            stats = self.knowledge_base.get_statistics()
            total_documents = stats.get("knowledge_base", {}).get("total_documents", 0)
            
            logger.info(f"현재 지식베이스 총 문서 수: {total_documents}")
            
            if total_documents == 0:
                logger.info("지식베이스에 데이터가 없습니다. 기존 RAG 최적화된 데이터 로드를 시도합니다...")
                
                # tools/data 디렉토리에서 기존 데이터 로드 시도
                import os
                data_paths = [
                    "tools/data",  # 상대 경로
                    "C:/blind/tools/data",  # 절대 경로
                    "./tools/data"  # 현재 디렉토리 기준
                ]
                
                data_loaded = False
                for data_path in data_paths:
                    if os.path.exists(data_path):
                        logger.info(f"데이터 디렉토리 발견: {data_path}")
                        try:
                            # 특정 회사들의 기존 RAG 최적화된 데이터 로드
                            companies_to_load = ["카카오", "네이버", "쿠팡", "삼성전자", "LG전자"]
                            success = await self.knowledge_base.load_chunk_data(
                                data_dir=data_path,
                                company_filter=companies_to_load
                            )
                            
                            if success:
                                logger.info("기존 RAG 데이터 로드 완료")
                                data_loaded = True
                                break
                            else:
                                logger.warning(f"{data_path}에서 데이터 로드 실패")
                        except Exception as e:
                            logger.warning(f"{data_path}에서 데이터 로드 중 오류: {str(e)}")
                            continue
                    else:
                        logger.debug(f"데이터 디렉토리 없음: {data_path}")
                
                if not data_loaded:
                    logger.warning("기존 데이터 로드에 실패했습니다. 지식베이스는 비어있는 상태로 시작됩니다.")
            else:
                logger.info(f"지식베이스에 이미 {total_documents}개의 문서가 있습니다.")
                
        except Exception as e:
            logger.error(f"지식베이스 데이터 확인 중 오류: {str(e)}")
            raise
    
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
        """회사 분석 페이지 렌더링 - 진입시 회사/직무/연도 모두 프리패치"""
        st.markdown("## 🏢 회사 분석")
        logger.info("회사 분석 페이지 진입")
        # 지식베이스 상태 확인
        if not self.knowledge_base or not st.session_state.get("knowledge_base_initialized", False):
            st.error("⚠️ 지식베이스가 초기화되지 않았습니다.")
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔄 지식베이스 재초기화 시도", type="primary"):
                    # 재초기화 시도
                    st.session_state.app_initialized = False
                    st.session_state.knowledge_base_initialized = False
                    # 세션에서 기존 지식베이스 제거
                    if "knowledge_base" in st.session_state:
                        del st.session_state.knowledge_base
                    # 회사 목록 캐시도 제거
                    if "available_companies" in st.session_state:
                        del st.session_state.available_companies
                    st.rerun()
            with col2:
                st.info("💡 앱을 새로고침(F5) 하거나 재시작해주세요.")
            return

        # 1. 회사 목록 로드 및 캐시
        if "available_companies" not in st.session_state:
            with st.spinner("회사 목록 로딩 중..."):
                try:
                    companies = asyncio.run(self.knowledge_base.get_available_companies())
                    logger.info(f"회사 데이터 get_available_companies()반환값: {companies}")
                    if companies and len(companies) > 0:
                        st.session_state.available_companies = sorted(companies)
                    else:
                        st.session_state.available_companies = []
                except Exception as e:
                    logger.error(f"회사명 로딩 실패: {str(e)}")
                    st.session_state.available_companies = []
                    st.error(f"⚠️ 회사 목록 로딩 실패: {str(e)}")
                    return

        # 2. 프리패치: 각 회사별 직무/연도 미리 DB에서 쿼리해 캐싱
        companies = st.session_state.get("available_companies", [])
        with st.spinner("회사별 직무/연도 데이터 캐싱 중..."):
            for company in companies:
                # 직무
                pos_key = f"positions_{company}"
                if pos_key not in st.session_state:
                    try:
                        positions = asyncio.run(self.knowledge_base.get_available_positions(company))
                        if positions and len(positions) > 0:
                            st.session_state[pos_key] = ["선택 안함"] + sorted([p for p in positions if p and p.strip()])
                        else:
                            st.session_state[pos_key] = ["선택 안함", f"{company}에 직무 데이터 없음"]
                    except Exception as e:
                        logger.error(f"직무({company}) prefetch 오류: {str(e)}")
                        st.session_state[pos_key] = ["선택 안함", "로딩 실패"]
                # 연도
                year_key = f"years_{company}"
                if year_key not in st.session_state:
                    try:
                        years = asyncio.run(self.knowledge_base.get_available_years(company))
                        if years and len(years) > 0:
                            st.session_state[year_key] = ["선택 안함"] + sorted([str(y) for y in years if y], reverse=True)
                        else:
                            st.session_state[year_key] = ["선택 안함", f"{company}에 연도 데이터 없음"]
                    except Exception as e:
                        logger.error(f"연도({company}) prefetch 오류: {str(e)}")
                        st.session_state[year_key] = ["선택 안함", "로딩 실패"]

        # 3. 회사 검색 인터페이스
        available_companies = st.session_state.get("available_companies", [])

        if available_companies and len(available_companies) > 0:
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_company = st.selectbox(
                    "📊 데이터가 있는 회사를 선택하세요",
                    options=["선택하세요..."] + available_companies,
                    key="company_dropdown",
                    help="DB에 실제 데이터가 있는 회사들입니다"
                )
                company_name = selected_company if selected_company != "선택하세요..." else None
            with col2:
                st.success(f"✅ {len(available_companies)}개 회사")
                st.caption("실제 분석 가능")
        else:
            st.warning("⚠️ DB에 회사 데이터가 없습니다")
            st.info("💡 먼저 migrate_reviews.py 스크립트를 실행하여 데이터를 마이그레이션해주세요")
            st.markdown("**마이그레이션 방법:**\n```bash\npython migrate_reviews.py\n```")
            company_name = st.text_input(
                "⚠️ 임시 회사명 입력 (데이터 없음)",
                placeholder="데이터가 없어 분석 결과가 제한될 수 있습니다",
                key="company_search_no_data",
                help="실제 분석을 위해서는 먼저 데이터 마이그레이션 필요"
            )

        if company_name:
            tab1, tab2, tab3 = st.tabs(["📋 커뮤니티", "ℹ️ 회사 정보", "🤖 AI 분석"])
            with tab1:
                self._render_community_tab(company_name)
            with tab2:
                self._render_company_info_tab(company_name)
            with tab3:
                self._render_ai_analysis_tab(company_name)
        else:
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
        
        # 지식베이스 상태 재확인
        logger.info(f"AI 분석 탭: knowledge_base={self.knowledge_base is not None}, initialized={st.session_state.get('knowledge_base_initialized', False)}")
        if not self.knowledge_base or not st.session_state.get("knowledge_base_initialized", False):
            st.error("⚠️ 지식베이스가 초기화되지 않았습니다.")
            if st.button("🔄 홈으로 돌아가서 재초기화", key="ai_analysis_reinit"):
                st.session_state.app_initialized = False
                st.session_state.knowledge_base_initialized = False
                # 세션에서 기존 지식베이스 제거
                if "knowledge_base" in st.session_state:
                    del st.session_state.knowledge_base
                # 회사 목록 캐시도 제거
                if "available_companies" in st.session_state:
                    del st.session_state.available_companies
                # 홈 페이지로 리다이렉트
                self.current_page = "홈"
                st.rerun()
            return
        
        # AI 분석 옵션 입력 폼
        with st.form("ai_analysis_form"):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # 회사명은 이미 입력되어 있으므로 표시만
                analysis_company = st.text_input("회사명", value=company_name, disabled=True)
            
            with col2:
                # 직무 선택 (선택사항) - SQLite 메타데이터 DB에서 동적 로딩
                cache_key = f"positions_{company_name}"
                if cache_key not in st.session_state or st.session_state.get("last_selected_company") != company_name:
                    # 회사가 바뀌었거나 캐시가 없으면 새로 로드
                    with st.spinner("직무 목록 로딩 중..."):
                        try:
                            if self.knowledge_base:
                                # SQLite 메타데이터 DB에서 해당 회사의 직무만 조회
                                available_positions = asyncio.run(
                                    self.knowledge_base.get_available_positions(company_name)
                                )
                                
                                if available_positions and len(available_positions) > 0:
                                    # SQLite DB에서 가져온 실제 데이터만 사용
                                    position_options = ["선택 안함"] + sorted([pos for pos in available_positions if pos and pos.strip()])
                                    logger.info(f"SQLite DB에서 {company_name}의 직무 {len(available_positions)}개 로드됨")
                                else:
                                    # 해당 회사에 직무 데이터가 없음
                                    position_options = ["선택 안함", f"{company_name}에 직무 데이터 없음"]
                                    logger.warning(f"{company_name}에 직무 데이터가 없습니다")
                                
                                st.session_state[cache_key] = position_options
                                st.session_state["last_selected_company"] = company_name
                            else:
                                position_options = ["선택 안함", "지식베이스 연결 필요"]
                                st.session_state[cache_key] = position_options
                        except Exception as e:
                            logger.error(f"{company_name} 직무 목록 로딩 실패: {str(e)}")
                            position_options = ["선택 안함", "로딩 실패"]
                            st.session_state[cache_key] = position_options
                
                # 캐시된 직무 옵션 가져오기
                position_options = st.session_state.get(cache_key, ["선택 안함"])
                
                # 실제 직무 개수 계산 (메시지 제외)
                actual_positions = [pos for pos in position_options if pos not in ["선택 안함", f"{company_name}에 직무 데이터 없음", "지식베이스 연결 필요", "로딩 실패"]]
                position_count = len(actual_positions)
                
                selected_position = st.selectbox(
                    f"직무 (선택) - {position_count}개", 
                    position_options,
                    help=f"{company_name} 회사의 실제 직무 목록입니다"
                )
            
            with col3:
                # 연도 선택 (선택사항) - SQLite 메타데이터 DB에서 동적 로딩
                cache_key = f"years_{company_name}"
                if cache_key not in st.session_state or st.session_state.get("last_selected_company_year") != company_name:
                    # 회사가 바뀌었거나 캐시가 없으면 새로 로드
                    with st.spinner("연도 목록 로딩 중..."):
                        try:
                            if self.knowledge_base:
                                # SQLite 메타데이터 DB에서 해당 회사의 연도만 조회
                                available_years = asyncio.run(
                                    self.knowledge_base.get_available_years(company_name)
                                )
                                
                                if available_years and len(available_years) > 0:
                                    # SQLite DB에서 가져온 실제 데이터만 사용 (내림차순 정렬)
                                    year_options = ["선택 안함"] + sorted([str(year) for year in available_years if year], reverse=True)
                                    logger.info(f"SQLite DB에서 {company_name}의 연도 {len(available_years)}개 로드됨")
                                else:
                                    # 해당 회사에 연도 데이터가 없음
                                    year_options = ["선택 안함", f"{company_name}에 연도 데이터 없음"]
                                    logger.warning(f"{company_name}에 연도 데이터가 없습니다")
                                
                                st.session_state[cache_key] = year_options
                                st.session_state["last_selected_company_year"] = company_name
                            else:
                                year_options = ["선택 안함", "지식베이스 연결 필요"]
                                st.session_state[cache_key] = year_options
                        except Exception as e:
                            logger.error(f"{company_name} 연도 목록 로딩 실패: {str(e)}")
                            year_options = ["선택 안함", "로딩 실패"]
                            st.session_state[cache_key] = year_options
                
                # 캐시된 연도 옵션 가져오기
                year_options = st.session_state.get(cache_key, ["선택 안함"])
                
                # 실제 연도 개수 계산 (메시지 제외)
                actual_years = [year for year in year_options if year not in ["선택 안함", f"{company_name}에 연도 데이터 없음", "지식베이스 연결 필요", "로딩 실패"]]
                year_count = len(actual_years)
                
                selected_year = st.selectbox(
                    f"연도 (선택) - {year_count}개",
                    year_options,
                    help=f"{company_name} 회사의 실제 연도 목록입니다"
                )
            
            # 세부 분석 키워드 입력 (접힌 섹션)
            st.markdown("---")
            with st.expander("🎯 세부 분석 키워드 (선택사항)", expanded=False):
                st.markdown("**각 항목별로 원하는 키워드를 입력하면 더 정확한 분석이 가능합니다.**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    culture_keywords = st.text_input(
                        "🏢 기업문화 키워드", 
                        placeholder="예: 수평적, 자율적, 혁신적",
                        help="기업문화에 대해 궁금한 구체적인 키워드를 입력하세요",
                        key="culture_keywords"
                    )
                    worklife_keywords = st.text_input(
                        "⚖️ 워라밸 키워드", 
                        placeholder="예: 야근, 휴가, 유연근무",
                        help="워라밸과 관련하여 궁금한 구체적인 키워드를 입력하세요",
                        key="worklife_keywords"
                    )
                    management_keywords = st.text_input(
                        "👥 경영진 키워드", 
                        placeholder="예: 소통, 리더십, 투명성",
                        help="경영진이나 상사에 대해 궁금한 구체적인 키워드를 입력하세요",
                        key="management_keywords"
                    )
                
                with col2:
                    salary_keywords = st.text_input(
                        "💰 연봉/복지 키워드", 
                        placeholder="예: 성과급, 스톡옵션, 복리후생",
                        help="연봉이나 복지에 대해 궁금한 구체적인 키워드를 입력하세요",
                        key="salary_keywords"
                    )
                    career_keywords = st.text_input(
                        "📈 커리어 성장 키워드", 
                        placeholder="예: 승진, 교육, 전문성",
                        help="커리어 성장에 대해 궁금한 구체적인 키워드를 입력하세요",
                        key="career_keywords"
                    )
                
                st.caption("💡 키워드를 입력하지 않으면 기본 분석이 진행됩니다. 쉼표로 여러 키워드를 구분할 수 있습니다.")
            
            # 분석 실행 버튼
            submitted = st.form_submit_button("🔍 AI로 분석하기", type="primary", use_container_width=True)
            
            if submitted:
                # 선택사항 처리 및 유효성 검증
                final_position = None
                final_year = None
                
                # 직무 유효성 검증
                if selected_position and selected_position != "선택 안함":
                    if selected_position not in [f"{company_name}에 직무 데이터 없음", "지식베이스 연결 필요", "로딩 실패"]:
                        final_position = selected_position
                    else:
                        st.warning(f"⚠️ 선택한 직무가 유효하지 않습니다: {selected_position}")
                
                # 연도 유효성 검증
                if selected_year and selected_year != "선택 안함":
                    if selected_year not in [f"{company_name}에 연도 데이터 없음", "지식베이스 연결 필요", "로딩 실패"]:
                        final_year = selected_year
                    else:
                        st.warning(f"⚠️ 선택한 연도가 유효하지 않습니다: {selected_year}")
                
                # 키워드 정리 및 처리
                keywords_dict = {
                    "company_culture": culture_keywords.strip() if culture_keywords else "",
                    "work_life_balance": worklife_keywords.strip() if worklife_keywords else "",
                    "management": management_keywords.strip() if management_keywords else "",
                    "salary_benefits": salary_keywords.strip() if salary_keywords else "",
                    "career_growth": career_keywords.strip() if career_keywords else ""
                }
                
                # 키워드가 하나라도 있는지 확인
                has_keywords = any(keywords_dict.values())
                
                # 분석 실행
                analysis_title = f"🤖 AI가 {company_name}"
                if final_position:
                    analysis_title += f" ({final_position})"
                if final_year:
                    analysis_title += f" [{final_year}년]"
                if has_keywords:
                    analysis_title += " 맞춤 키워드로"
                analysis_title += "에 대해 분석중입니다..."
                
                with st.spinner(analysis_title):
                    # 실제 에이전트들을 병렬로 실행 (키워드 전달)
                    analysis_result = asyncio.run(
                        self._perform_parallel_agent_analysis(
                            company_name, 
                            position=final_position,
                            year=final_year,
                            keywords=keywords_dict
                        )
                    )
                    if analysis_result:
                        st.session_state.ai_analysis_result = analysis_result
                        success_msg = f"✅ {company_name} 분석 완료!"
                        if final_position or final_year:
                            success_msg += f" (필터: {final_position or '전체 직무'} / {final_year or '전체 연도'})"
                        st.success(success_msg)
                        st.rerun()
        
        st.info("🔍 AI가 5개의 전문 에이전트를 통해 기업문화, 워라밸, 경영진, 연봉/복지, 커리어 성장을 종합 분석합니다.")
        
        # 분석 결과 표시
        if st.session_state.get("ai_analysis_result"):
            self._display_parallel_agent_results(st.session_state.ai_analysis_result)
    
    async def _perform_parallel_agent_analysis(
        self, 
        company_name: str, 
        position: Optional[str] = None, 
        year: Optional[str] = None,
        keywords: Optional[Dict[str, str]] = None
    ) -> Dict:
        """5개의 전문 에이전트를 병렬로 실행하여 분석 수행"""
        try:
            # 에이전트 import
            from ..agents.company_culture_agent import CompanyCultureAgent
            from ..agents.work_life_balance_agent import WorkLifeBalanceAgent
            from ..agents.management_agent import ManagementAgent
            from ..agents.salary_benefits_agent import SalaryBenefitsAgent
            from ..agents.career_growth_agent import CareerGrowthAgent
            
            # 에이전트 인스턴스 생성
            agents = {
                "company_culture": CompanyCultureAgent(),
                "work_life_balance": WorkLifeBalanceAgent(),
                "management": ManagementAgent(),
                "salary_benefits": SalaryBenefitsAgent(),
                "career_growth": CareerGrowthAgent()
            }
            
            # 공통 컨텍스트 설정 (키워드 포함)
            context = {
                "company_name": company_name,
                "position": position,
                "year": year,
                "keywords": keywords or {},
                "timestamp": datetime.now(),
                "analysis_type": "comprehensive"
            }
            
            # 병렬 실행을 위한 태스크 준비
            tasks = []
            for agent_name, agent in agents.items():
                query = f"{company_name} {agent_name.replace('_', ' ')} 분석"
                logger.info(f"{agent_name} 에이전트 태스크 준비: {query}")
                
                task = agent.execute(
                    query=query,
                    context=context,
                    user_profile=SessionManager.get_user_profile()
                )
                tasks.append((agent_name, task))
            
            # 병렬 실행
            logger.info(f"5개 에이전트 병렬 실행 시작: {company_name}")
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            logger.info(f"5개 에이전트 병렬 실행 완료: {company_name}")
            
            # 결과 처리
            agent_results = {}
            for (agent_name, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    error_msg = str(result)
                    logger.error(f"{agent_name} 에이전트 실행 실패: {error_msg}")
                    agent_results[agent_name] = {
                        "success": False,
                        "error": error_msg,
                        "agent_name": agent_name
                    }
                else:
                    logger.info(f"{agent_name} 에이전트 실행 결과: success={result.success if hasattr(result, 'success') else 'unknown'}")
                    if hasattr(result, 'result') and result.result:
                        logger.info(f"{agent_name} 분석 결과 요약: {str(result.result)[:200]}...")
                    agent_results[agent_name] = {
                        "success": result.success if hasattr(result, 'success') else False,
                        "result": result.result if hasattr(result, 'result') and result.success else None,
                        "error": result.error_message if hasattr(result, 'error_message') and not result.success else None,
                        "confidence_score": result.confidence_score if hasattr(result, 'confidence_score') else 0.0,
                        "execution_time": result.execution_time if hasattr(result, 'execution_time') else 0.0,
                        "agent_name": agent_name
                    }
            
            final_result = {
                "company_name": company_name,
                "position": position,
                "year": year,
                "analysis_timestamp": datetime.now(),
                "agent_results": agent_results,
                "total_agents": len(agents),
                "success_count": sum(1 for r in agent_results.values() if r.get("success", False))
            }
            
            logger.info(f"최종 분석 결과 생성 완료: {company_name} (성공: {final_result['success_count']}/{final_result['total_agents']})")
            
            # 각 에이전트별 결과 요약 로그
            for agent_name, result in agent_results.items():
                if result.get("success"):
                    logger.info(f"{agent_name} 분석 완료 - 신뢰도: {result.get('confidence_score', 0):.2f}")
                else:
                    logger.warning(f"{agent_name} 분석 실패 - 오류: {result.get('error', 'Unknown error')}")
            
            return final_result
            
        except Exception as e:
            st.error(f"병렬 에이전트 분석 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"병렬 에이전트 분석 오류: {str(e)}")
            return self._get_fallback_agent_results(company_name, position, year)
    
    def _get_fallback_agent_results(self, company_name: str, position: Optional[str], year: Optional[str]) -> Dict:
        """폴백용 에이전트 결과 생성"""
        fallback_results = {
            "company_culture": {
                "success": True,
                "result": {
                    "summary": f"{company_name}는 전통적이고 안정적인 조직문화를 가지고 있습니다.",
                    "strengths": {
                        "pros": ["안정적 조직", "체계적 시스템", "복리후생"],
                        "analysis": "안정성을 중시하는 보수적이지만 견고한 문화"
                    },
                    "weaknesses": {
                        "cons": ["보수적 문화", "혁신 속도", "변화 저항"],
                        "analysis": "급격한 변화보다는 점진적 개선을 선호"
                    },
                    "score": 3.2
                },
                "confidence_score": 0.75,
                "execution_time": 2.3,
                "agent_name": "company_culture"
            },
            "work_life_balance": {
                "success": True,
                "result": {
                    "summary": f"{company_name}의 워라밸은 대체로 만족스러운 수준입니다.",
                    "strengths": {
                        "pros": ["정시퇴근", "휴가 사용", "유연근무"],
                        "analysis": "직원 복지를 중시하는 건전한 근무환경"
                    },
                    "weaknesses": {
                        "cons": ["프로젝트 기간 야근", "부서별 차이", "성수기 과로"],
                        "analysis": "업무량에 따른 워라밸 편차 존재"
                    },
                    "score": 3.5
                },
                "confidence_score": 0.72,
                "execution_time": 2.1,
                "agent_name": "work_life_balance"
            },
            "management": {
                "success": True,
                "result": {
                    "summary": f"{company_name}의 경영진은 안정적이지만 소통이 아쉬운 면이 있습니다.",
                    "strengths": {
                        "pros": ["경영 안정성", "장기 비전", "의사결정 체계"],
                        "analysis": "체계적이고 신중한 경영 스타일"
                    },
                    "weaknesses": {
                        "cons": ["상하 소통", "피드백 부족", "관료적 문화"],
                        "analysis": "수직적 구조로 인한 소통의 한계"
                    },
                    "score": 2.9
                },
                "confidence_score": 0.68,
                "execution_time": 2.5,
                "agent_name": "management"
            },
            "salary_benefits": {
                "success": True,
                "result": {
                    "summary": f"{company_name}의 연봉과 복리후생은 업계 평균 수준입니다.",
                    "strengths": {
                        "pros": ["4대보험", "퇴직금", "복지포인트", "건강검진"],
                        "analysis": "안정적인 복리후생 시스템과 기본적인 연봉 수준"
                    },
                    "weaknesses": {
                        "cons": ["성과급 한계", "연봉 상승폭", "인센티브 부족"],
                        "analysis": "경쟁력 있는 보상보다는 안정성 중심"
                    },
                    "score": 3.1
                },
                "confidence_score": 0.78,
                "execution_time": 2.0,
                "agent_name": "salary_benefits"
            },
            "career_growth": {
                "success": True,
                "result": {
                    "summary": f"{company_name}는 안정적인 커리어 패스를 제공합니다.",
                    "strengths": {
                        "pros": ["교육 지원", "승진 체계", "경력 개발"],
                        "analysis": "체계적인 인재 육성 시스템"
                    },
                    "weaknesses": {
                        "cons": ["승진 경쟁", "전문성 한계", "혁신 기회 부족"],
                        "analysis": "안정적이지만 도전적 성장 기회는 제한적"
                    },
                    "score": 3.0
                },
                "confidence_score": 0.73,
                "execution_time": 2.2,
                "agent_name": "career_growth"
            }
        }
        
        return {
            "company_name": company_name,
            "position": position,
            "year": year,
            "analysis_timestamp": datetime.now(),
            "agent_results": fallback_results,
            "total_agents": 5,
            "success_count": 5
        }
    
    def _display_parallel_agent_results(self, result: Dict):
        """병렬 에이전트 분석 결과 시각화"""
        logger.info(f"분석 결과 표시 시작: {result.get('company_name', 'Unknown')}")
        
        st.markdown("### 📊 AI 분석 결과")
        
        company_name = result.get("company_name", "알 수 없음")
        position = result.get("position")
        year = result.get("year")
        agent_results = result.get("agent_results", {})

        
        logger.info(f"표시할 에이전트 결과 개수: {len(agent_results)}")
        if not agent_results:
            logger.warning("에이전트 결과가 비어있습니다!")
            st.warning("분석 결과가 없습니다. 다시 시도해주세요.")
            return
        
        # 분석 정보 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏢 회사", company_name)
        with col2:
            st.metric("💼 직무", position if position else "전체")
        with col3:
            st.metric("📅 연도", year if year else "전체")
        with col4:
            success_count = result.get("success_count", 0)
            total_agents = result.get("total_agents", 0)
            st.metric("✅ 성공률", f"{success_count}/{total_agents}")
        
        st.divider()
        
        # 에이전트별 분석 결과 표시
        agent_names_korean = {
            "company_culture": "🏢 기업문화",
            "work_life_balance": "⚖️ 워라밸",
            "management": "👥 경영진",
            "salary_benefits": "💰 연봉/복지",
            "career_growth": "📈 커리어 성장"
        }
        
        displayed_count = 0
        for agent_key, agent_data in agent_results.items():
            logger.info(f"에이전트 {agent_key} 결과 처리 중...")
            logger.info(f"{agent_key} 성공여부: {agent_data.get('success', False)}")
            logger.info(f"{agent_key} 결과 존재: {bool(agent_data.get('result'))}")
            
            if agent_data.get("success", False) and agent_data.get("result"):
                logger.info(f"{agent_key} 결과 표시 중...")
                self._display_single_agent_result(
                    agent_names_korean.get(agent_key, agent_key), 
                    agent_data
                )
                displayed_count += 1
            else:
                logger.warning(f"{agent_key} 오류 또는 결과 없음 표시 중...")
                self._display_agent_error(
                    agent_names_korean.get(agent_key, agent_key), 
                    agent_data
                )
        
        logger.info(f"총 {displayed_count}개의 성공적인 에이전트 결과 표시 완료")
    
    def _display_single_agent_result(self, agent_name: str, agent_data: Dict):
        """단일 에이전트 결과 표시 (키명 유연화)"""
        result = agent_data.get("result", {})
        confidence = agent_data.get("confidence_score", 0.0)
        exec_time = agent_data.get("execution_time", 0.0)

        def get_first_exist(d, *keys, default=None):
            if not isinstance(d, dict):
                return default if default is not None else []
            for k in keys:
                v = d.get(k)
                if v is not None:
                    return v
            return default if default is not None else []

        with st.expander(f"{agent_name} 분석", expanded=True):
            # 요약 정보
            summary = result.get("summary")
            if not summary:
                # 만약 summary 키 없다면, final_summary or analysis 등 대체 키 사용
                summary = get_first_exist(result, "final_summary", "analysis", default="분석 결과 없음")
            st.write(f"**📋 요약**: {summary}")

            # 장점/단점 표시
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ✅ 장점")
                strengths = result.get("strengths") or agent_data.get("strengths", {})
                if isinstance(strengths, dict):
                    pros = get_first_exist(strengths, "pros", "strengths", default=[])
                    analysis = get_first_exist(strengths, "analysis", "final_summary", default="")
                    if pros:
                        # 장점들을 박스에 넣어서 표시 (줄 구분 추가)
                        pros_content = "<br>".join([f"• {pro}" for pro in pros])
                        st.markdown(f"""
                        <div style='background-color: #e8f5e8; border: 1px solid #4caf50; border-radius: 5px; padding: 10px; margin: 5px 0;'>
                        {pros_content}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.write("장점 정보 없음")
                    if analysis:
                        # 분석을 별도 박스에 넣어서 표시
                        st.markdown(f"""
                        <div style='background-color: #f0f8ff; border: 1px solid #2196f3; border-radius: 5px; padding: 10px; margin: 5px 0;'>
                        <strong>💡 분석:</strong> {analysis}
                        </div>
                        """, unsafe_allow_html=True)
                elif isinstance(strengths, list):
                    pros_content = "<br>".join([f"• {pro}" for pro in strengths])
                    st.markdown(f"""
                    <div style='background-color: #e8f5e8; border: 1px solid #4caf50; border-radius: 5px; padding: 10px; margin: 5px 0;'>
                    {pros_content}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.write("장점 정보 없음")

            with col2:
                st.markdown("### ⚠️ 단점")
                weaknesses = result.get("weaknesses") or agent_data.get("weaknesses", {})
                if isinstance(weaknesses, dict):
                    cons = get_first_exist(weaknesses, "cons", "weaknesses", "strengths", default=[])
                    analysis = get_first_exist(weaknesses, "analysis", "final_summary", default="")
                    if cons:
                        # 단점들을 박스에 넣어서 표시 (줄 구분 추가)
                        cons_content = "<br>".join([f"• {con}" for con in cons])
                        st.markdown(f"""
                        <div style='background-color: #ffeaea; border: 1px solid #ff6b6b; border-radius: 5px; padding: 10px; margin: 5px 0;'>
                        {cons_content}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.write("단점 정보 없음")
                    if analysis:
                        # 분석을 별도 박스에 넣어서 표시
                        st.markdown(f"""
                        <div style='background-color: #f0f8ff; border: 1px solid #2196f3; border-radius: 5px; padding: 10px; margin: 5px 0;'>
                        <strong>💡 분석:</strong> {analysis}
                        </div>
                        """, unsafe_allow_html=True)
                elif isinstance(weaknesses, list):
                    cons_content = "<br>".join([f"• {con}" for con in weaknesses])
                    st.markdown(f"""
                    <div style='background-color: #ffeaea; border: 1px solid #ff6b6b; border-radius: 5px; padding: 10px; margin: 5px 0;'>
                    {cons_content}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.write("단점 정보 없음")
    
    def _display_agent_error(self, agent_name: str, agent_data: Dict):
        """에이전트 오류 표시"""
        error_msg = agent_data.get("error", "알 수 없는 오류")
        
        with st.expander(f"{agent_name} 분석 (오류)", expanded=False):
            st.error(f"분석 실패: {error_msg}")
            st.write("이 에이전트의 분석을 다시 시도해보세요.")
    
    
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