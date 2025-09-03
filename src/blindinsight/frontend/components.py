"""
프론트엔드 UI 구성 요소들

재사용 가능한 Streamlit 위젯과 커스텀 컴포넌트들을 제공합니다.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json


class CompanySearchWidget:
    """
    회사 검색 위젯
    
    자동완성과 필터링 기능을 제공하는 회사 검색 인터페이스입니다.
    """
    
    @staticmethod
    def render(
        key: str = "company_search",
        placeholder: str = "회사명을 입력하세요...",
        suggestions: List[str] = None
    ) -> Optional[str]:
        """
        회사 검색 위젯 렌더링
        
        Args:
            key: 위젯 키
            placeholder: 플레이스홀더 텍스트
            suggestions: 추천 회사 리스트
            
        Returns:
            선택된 회사명 또는 None
        """
        
        # 기본 추천 회사들
        if suggestions is None:
            suggestions = [
                "네이버", "카카오", "쿠팡", "토스", "당근마켓", "배달의민족",
                "라인", "우아한형제들", "야놀자", "마켓컬리", "직방", "원티드"
            ]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 검색 입력
            search_input = st.text_input(
                "회사명 검색",
                placeholder=placeholder,
                key=f"{key}_input"
            )
        
        with col2:
            # 인기 회사 선택
            if st.button("🔥 인기 회사", key=f"{key}_popular"):
                # 랜덤하게 인기 회사 선택
                import random
                selected = random.choice(suggestions[:6])
                st.session_state[f"{key}_input"] = selected
                st.rerun()
        
        # 추천 회사 버튼들
        if search_input:
            # 입력된 텍스트와 일치하는 추천 회사들 필터링
            filtered_suggestions = [
                company for company in suggestions 
                if search_input.lower() in company.lower()
            ]
        else:
            filtered_suggestions = suggestions[:6]  # 상위 6개만 표시
        
        if filtered_suggestions:
            st.markdown("**추천 회사:**")
            
            # 3열로 버튼 배치
            cols = st.columns(3)
            for i, company in enumerate(filtered_suggestions[:6]):
                with cols[i % 3]:
                    if st.button(company, key=f"{key}_suggestion_{i}"):
                        st.session_state[f"{key}_input"] = company
                        st.rerun()
        
        return search_input if search_input else None


class AnalysisResultDisplay:
    """
    분석 결과 표시 구성 요소
    
    회사 분석 결과를 시각적으로 표현하는 다양한 차트와 위젯들을 제공합니다.
    """
    
    @staticmethod
    def render_company_overview(
        company_name: str,
        overall_rating: float,
        recommendation_rate: float,
        growth_indicator: str,
        employee_count: int = None
    ):
        """회사 개요 카드 렌더링"""
        
        st.markdown(f"### 🏢 {company_name} 개요")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "전체 평점",
                f"{overall_rating:.1f}/5.0",
                delta=f"{overall_rating - 3.5:+.1f}"
            )
        
        with col2:
            st.metric(
                "추천율",
                f"{recommendation_rate:.0f}%",
                delta=f"{recommendation_rate - 65:+.0f}%"
            )
        
        with col3:
            st.metric(
                "성장성",
                growth_indicator,
                delta="상승" if growth_indicator == "높음" else None
            )
        
        with col4:
            if employee_count:
                st.metric("직원 수", f"{employee_count:,}명")
            else:
                st.metric("분석 데이터", "충분")
    
    @staticmethod
    def render_rating_radar_chart(ratings: Dict[str, float], title: str = "평가 분석"):
        """레이더 차트 렌더링"""
        
        categories = list(ratings.keys())
        values = list(ratings.values())
        
        # 레이더 차트 생성
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # 원형으로 만들기 위해 첫 번째 값 추가
            theta=categories + [categories[0]],
            fill='toself',
            name=title,
            line_color='rgb(46, 138, 230)',
            fillcolor='rgba(46, 138, 230, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 5]
                )
            ),
            showlegend=False,
            title=title,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_salary_distribution(salary_data: List[Dict]):
        """연봉 분포 차트 렌더링"""
        
        if not salary_data:
            st.warning("연봉 데이터가 없습니다.")
            return
        
        df = pd.DataFrame(salary_data)
        
        # 박스 플롯
        fig = px.box(
            df,
            x="position",
            y="total_compensation",
            title="포지션별 연봉 분포",
            labels={"position": "포지션", "total_compensation": "총 연봉 (원)"}
        )
        
        fig.update_traces(boxpoints="all", jitter=0.3, pointpos=-1.8)
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 평균 연봉 테이블
        st.markdown("#### 📊 포지션별 평균 연봉")
        
        avg_salary = df.groupby("position")["total_compensation"].agg([
            "mean", "median", "count"
        ]).round().astype(int)
        
        avg_salary.columns = ["평균", "중앙값", "샘플 수"]
        avg_salary["평균"] = avg_salary["평균"].apply(lambda x: f"{x:,}원")
        avg_salary["중앙값"] = avg_salary["중앙값"].apply(lambda x: f"{x:,}원")
        
        st.dataframe(avg_salary, use_container_width=True)
    
    @staticmethod
    def render_sentiment_timeline(sentiment_data: List[Dict]):
        """감정 분석 타임라인 렌더링"""
        
        if not sentiment_data:
            st.warning("감정 분석 데이터가 없습니다.")
            return
        
        df = pd.DataFrame(sentiment_data)
        df["date"] = pd.to_datetime(df["date"])
        
        # 감정 점수 시계열 차트
        fig = px.line(
            df,
            x="date",
            y="sentiment_score",
            title="시간별 감정 분석 트렌드",
            labels={"date": "날짜", "sentiment_score": "감정 점수"},
            color_discrete_sequence=["#2E8AE6"]
        )
        
        # 중립선 추가
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="중립"
        )
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_keyword_cloud(keywords: List[Tuple[str, int]]):
        """키워드 클라우드 렌더링 (간단한 바 차트)"""
        
        if not keywords:
            st.warning("키워드 데이터가 없습니다.")
            return
        
        # 상위 10개 키워드만 표시
        top_keywords = keywords[:10]
        
        df = pd.DataFrame(top_keywords, columns=["키워드", "빈도"])
        
        fig = px.bar(
            df,
            x="빈도",
            y="키워드",
            orientation="h",
            title="주요 키워드 분석",
            color="빈도",
            color_continuous_scale="viridis"
        )
        
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_pros_cons_analysis(pros: List[str], cons: List[str]):
        """장단점 분석 렌더링"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ✅ 장점")
            if pros:
                for pro in pros[:5]:  # 상위 5개만 표시
                    st.markdown(f"• {pro}")
            else:
                st.write("장점 데이터가 없습니다.")
        
        with col2:
            st.markdown("#### ❌ 단점")
            if cons:
                for con in cons[:5]:  # 상위 5개만 표시
                    st.markdown(f"• {con}")
            else:
                st.write("단점 데이터가 없습니다.")


class ChatInterface:
    """
    채팅 인터페이스 구성 요소
    
    AI와의 대화형 인터페이스를 제공합니다.
    """
    
    @staticmethod
    def render_message_container(messages: List[Dict], container_height: int = 400):
        """메시지 컨테이너 렌더링"""
        
        # 스크롤 가능한 컨테이너 스타일
        st.markdown(f"""
        <div style="height: {container_height}px; overflow-y: auto; border: 1px solid #ddd; 
                    border-radius: 5px; padding: 10px; background-color: #f9f9f9;">
        """, unsafe_allow_html=True)
        
        for message in messages:
            ChatInterface._render_single_message(
                message["role"], 
                message["content"],
                message.get("timestamp", datetime.now())
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    @staticmethod
    def _render_single_message(role: str, content: str, timestamp: datetime):
        """단일 메시지 렌더링"""
        
        time_str = timestamp.strftime("%H:%M")
        
        if role == "user":
            # 사용자 메시지 (오른쪽 정렬)
            st.markdown(f"""
            <div style="text-align: right; margin: 10px 0;">
                <div style="display: inline-block; background-color: #007bff; color: white; 
                           padding: 8px 12px; border-radius: 15px; max-width: 70%;">
                    {content}
                </div>
                <div style="font-size: 0.8em; color: #666; margin-top: 2px;">
                    {time_str}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # AI 메시지 (왼쪽 정렬)
            st.markdown(f"""
            <div style="text-align: left; margin: 10px 0;">
                <div style="display: inline-block; background-color: #e9ecef; color: #333; 
                           padding: 8px 12px; border-radius: 15px; max-width: 70%;">
                    🤖 {content}
                </div>
                <div style="font-size: 0.8em; color: #666; margin-top: 2px;">
                    {time_str}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_suggested_questions(questions: List[str], key_prefix: str = "suggested"):
        """추천 질문 버튼들 렌더링"""
        
        if not questions:
            return None
        
        st.markdown("**💡 추천 질문:**")
        
        cols = st.columns(2)
        selected_question = None
        
        for i, question in enumerate(questions):
            with cols[i % 2]:
                if st.button(f"💬 {question}", key=f"{key_prefix}_{i}"):
                    selected_question = question
        
        return selected_question
    
    @staticmethod
    def render_quick_actions():
        """빠른 액션 버튼들 렌더링"""
        
        st.markdown("**⚡ 빠른 액션:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 회사 비교", key="quick_compare"):
                return "compare_companies"
        
        with col2:
            if st.button("💰 연봉 분석", key="quick_salary"):
                return "salary_analysis"
        
        with col3:
            if st.button("🎯 커리어 조언", key="quick_career"):
                return "career_advice"
        
        return None


class DataVisualization:
    """
    데이터 시각화 구성 요소
    
    다양한 형태의 데이터를 시각적으로 표현하는 차트들을 제공합니다.
    """
    
    @staticmethod
    def render_comparison_chart(companies: List[str], metrics: Dict[str, List[float]]):
        """회사 비교 차트 렌더링"""
        
        if not companies or not metrics:
            st.warning("비교할 데이터가 없습니다.")
            return
        
        # 데이터프레임 생성
        df_data = {"회사": companies}
        df_data.update(metrics)
        df = pd.DataFrame(df_data)
        
        # 멀티 바 차트
        fig = px.bar(
            df.melt(id_vars=["회사"], var_name="지표", value_name="점수"),
            x="회사",
            y="점수",
            color="지표",
            barmode="group",
            title="회사별 지표 비교"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_trend_analysis(data: List[Dict], x_col: str, y_col: str, title: str):
        """트렌드 분석 차트 렌더링"""
        
        if not data:
            st.warning("트렌드 데이터가 없습니다.")
            return
        
        df = pd.DataFrame(data)
        
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=title,
            markers=True
        )
        
        # 추세선 추가
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col].rolling(window=3).mean(),
                mode="lines",
                name="추세선",
                line=dict(dash="dash", color="red")
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_correlation_heatmap(correlation_matrix: pd.DataFrame):
        """상관관계 히트맵 렌더링"""
        
        if correlation_matrix.empty:
            st.warning("상관관계 데이터가 없습니다.")
            return
        
        fig = px.imshow(
            correlation_matrix,
            title="지표 간 상관관계 분석",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_distribution_chart(data: List[float], title: str, x_label: str):
        """분포 차트 렌더링"""
        
        if not data:
            st.warning("분포 데이터가 없습니다.")
            return
        
        fig = px.histogram(
            x=data,
            nbins=20,
            title=title,
            labels={"x": x_label, "y": "빈도"}
        )
        
        # 평균선 추가
        mean_val = sum(data) / len(data)
        fig.add_vline(
            x=mean_val,
            line_dash="dash",
            line_color="red",
            annotation_text=f"평균: {mean_val:.1f}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def render_gauge_chart(value: float, title: str, max_value: float = 100):
        """게이지 차트 렌더링"""
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title},
            delta={"reference": max_value * 0.7},  # 기준값을 70%로 설정
            gauge={
                "axis": {"range": [None, max_value]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, max_value * 0.5], "color": "lightgray"},
                    {"range": [max_value * 0.5, max_value * 0.8], "color": "gray"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


class StatusIndicator:
    """
    상태 표시 구성 요소
    
    시스템 상태, 처리 상태 등을 시각적으로 표시합니다.
    """
    
    @staticmethod
    def render_loading_spinner(message: str = "처리 중...", key: str = "loading"):
        """로딩 스피너 렌더링"""
        
        with st.spinner(message):
            # 실제 처리 작업은 호출하는 쪽에서 수행
            pass
    
    @staticmethod
    def render_progress_bar(current: int, total: int, message: str = ""):
        """진행률 바 렌더링"""
        
        progress = current / max(total, 1)
        st.progress(progress)
        
        if message:
            st.write(f"{message} ({current}/{total})")
    
    @staticmethod
    def render_status_badges(statuses: Dict[str, str]):
        """상태 배지들 렌더링"""
        
        cols = st.columns(len(statuses))
        
        for i, (label, status) in enumerate(statuses.items()):
            with cols[i]:
                color = StatusIndicator._get_status_color(status)
                st.markdown(f"""
                <div style="text-align: center; padding: 5px; 
                           background-color: {color}; border-radius: 5px; color: white;">
                    <strong>{label}</strong><br>
                    {status}
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def _get_status_color(status: str) -> str:
        """상태에 따른 색상 반환"""
        colors = {
            "성공|완료|정상|active": "#28a745",
            "진행중|처리중|pending": "#ffc107", 
            "실패|오류|error|failed": "#dc3545"
        }
        
        status_lower = status.lower()
        for keywords, color in colors.items():
            if any(keyword in status_lower for keyword in keywords.split("|")):
                return color
        return "#6c757d"


class FormComponents:
    """
    폼 구성 요소들
    
    사용자 입력을 받는 다양한 폼 요소들을 제공합니다.
    """
    
    @staticmethod
    def render_user_profile_form(current_profile: Dict = None) -> Optional[Dict]:
        """사용자 프로필 폼 렌더링"""
        
        if current_profile is None:
            current_profile = {}
        
        with st.form("user_profile_form"):
            st.markdown("### 👤 사용자 정보")
            
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("이름", value=current_profile.get("name", ""))
                experience = st.number_input(
                    "경력 연수", 
                    min_value=0, 
                    max_value=50, 
                    value=current_profile.get("experience_years", 0)
                )
                education = st.selectbox(
                    "학력",
                    ["고등학교", "전문대", "대학교", "대학원"],
                    index=2
                )
            
            with col2:
                position = st.text_input(
                    "현재 직책", 
                    value=current_profile.get("current_position", "")
                )
                company = st.text_input(
                    "현재 회사", 
                    value=current_profile.get("current_company", "")
                )
                location = st.text_input(
                    "지역", 
                    value=current_profile.get("location", "서울")
                )
            
            skills = st.text_area(
                "보유 스킬 (쉼표로 구분)",
                value=current_profile.get("skills", ""),
                height=100
            )
            
            interests = st.text_area(
                "관심 분야 (쉼표로 구분)",
                value=current_profile.get("interests", ""),
                height=100
            )
            
            submitted = st.form_submit_button("💾 저장")
            
            if submitted:
                return {
                    "name": name,
                    "experience_years": experience,
                    "education": education,
                    "current_position": position,
                    "current_company": company,
                    "location": location,
                    "skills": [s.strip() for s in skills.split(",") if s.strip()],
                    "interests": [i.strip() for i in interests.split(",") if i.strip()]
                }
        
        return None
    
    @staticmethod
    def render_analysis_settings_form() -> Optional[Dict]:
        """분석 설정 폼 렌더링"""
        
        with st.form("analysis_settings_form"):
            st.markdown("### ⚙️ 분석 설정")
            
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_depth = st.selectbox(
                    "분석 깊이",
                    ["빠른 분석", "표준 분석", "상세 분석"],
                    index=1
                )
                
                include_sentiment = st.checkbox("감정 분석 포함", value=True)
                include_salary = st.checkbox("연봉 분석 포함", value=True)
            
            with col2:
                result_count = st.slider("최대 결과 수", 5, 50, 20)
                
                include_trends = st.checkbox("트렌드 분석 포함", value=True)
                include_comparison = st.checkbox("경쟁사 비교 포함", value=False)
            
            submitted = st.form_submit_button("🔧 설정 적용")
            
            if submitted:
                return {
                    "analysis_depth": analysis_depth,
                    "include_sentiment": include_sentiment,
                    "include_salary": include_salary,
                    "include_trends": include_trends,
                    "include_comparison": include_comparison,
                    "result_count": result_count
                }
        
        return None