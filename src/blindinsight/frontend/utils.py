"""
프론트엔드 유틸리티 함수들

Streamlit 애플리케이션에서 사용되는 공통 유틸리티와 헬퍼 함수들을 제공합니다.
"""

import streamlit as st
import pandas as pd
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """
    Streamlit 캐시 관리 클래스
    
    애플리케이션의 성능 최적화를 위한 캐시 관리 기능을 제공합니다.
    """
    
    @staticmethod
    @st.cache_data(ttl=3600)  # 1시간 캐시
    def cached_company_search(query: str, limit: int = 10) -> List[str]:
        """
        회사 검색 결과 캐싱
        
        Args:
            query: 검색 쿼리
            limit: 결과 개수 제한
            
        Returns:
            검색된 회사 리스트
        """
        # 실제 환경에서는 데이터베이스나 API 호출
        # 여기서는 시뮬레이션 데이터 반환
        
        all_companies = [
            "네이버", "카카오", "쿠팡", "토스", "당근마켓", "배달의민족",
            "라인", "우아한형제들", "야놀자", "마켓컬리", "직방", "원티드",
            "삼성전자", "LG전자", "SK하이닉스", "현대자동차", "기아", "포스코",
            "넥슨", "엔씨소프트", "넷마블", "스마일게이트", "펄어비스"
        ]
        
        if not query:
            return all_companies[:limit]
        
        # 쿼리와 일치하는 회사들 필터링
        filtered = [
            company for company in all_companies
            if query.lower() in company.lower()
        ]
        
        return filtered[:limit]
    
    @staticmethod
    @st.cache_data(ttl=1800)  # 30분 캐시
    def cached_analysis_result(company_name: str, analysis_type: str) -> Optional[Dict]:
        """
        분석 결과 캐싱
        
        Args:
            company_name: 회사명
            analysis_type: 분석 유형
            
        Returns:
            캐시된 분석 결과 또는 None
        """
        # 실제로는 데이터베이스에서 조회
        # 시뮬레이션용 샘플 데이터
        
        if company_name and analysis_type:
            return {
                "company_name": company_name,
                "analysis_type": analysis_type,
                "cached_at": datetime.now().isoformat(),
                "sample_data": True
            }
        
        return None
    
    @staticmethod
    def clear_cache():
        """모든 캐시 클리어"""
        st.cache_data.clear()
        logger.info("Streamlit 캐시 클리어 완료")
    
    @staticmethod
    def get_cache_stats() -> Dict[str, Any]:
        """캐시 통계 정보 반환"""
        # Streamlit의 캐시 통계는 직접 접근이 어려우므로 
        # 간단한 정보만 제공
        return {
            "cache_enabled": True,
            "ttl_company_search": 3600,
            "ttl_analysis_result": 1800,
            "last_cleared": st.session_state.get("cache_last_cleared")
        }


class UIUtils:
    """
    UI 관련 유틸리티 함수들
    
    Streamlit UI 개발에 도움이 되는 다양한 헬퍼 함수들을 제공합니다.
    """
    
    @staticmethod
    def apply_custom_css():
        """커스텀 CSS 스타일 적용"""
        
        custom_css = """
        <style>
        /* 메인 헤더 스타일 */
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
        }
        
        .main-header p {
            color: #f0f0f0;
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        }
        
        /* 메트릭 카드 스타일 */
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e1e5e9;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        /* 채팅 메시지 스타일 */
        .chat-message {
            padding: 0.5rem 1rem;
            border-radius: 15px;
            margin: 0.5rem 0;
            max-width: 70%;
        }
        
        .chat-user {
            background-color: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .chat-assistant {
            background-color: #e9ecef;
            color: #333;
        }
        
        /* 상태 배지 스타일 */
        .status-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.875rem;
            font-weight: bold;
            text-align: center;
        }
        
        .status-success {
            background-color: #28a745;
            color: white;
        }
        
        .status-warning {
            background-color: #ffc107;
            color: #212529;
        }
        
        .status-error {
            background-color: #dc3545;
            color: white;
        }
        
        /* 사이드바 스타일 */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* 버튼 호버 효과 */
        .stButton > button:hover {
            background-color: #0056b3;
            border-color: #0056b3;
            color: white;
        }
        
        /* 데이터프레임 스타일 */
        .dataframe {
            font-size: 0.875rem;
        }
        
        /* 진행률 바 스타일 */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        </style>
        """
        
        st.markdown(custom_css, unsafe_allow_html=True)
    
    @staticmethod
    def create_metric_card(title: str, value: str, delta: str = None, color: str = "blue") -> str:
        """
        메트릭 카드 HTML 생성
        
        Args:
            title: 메트릭 제목
            value: 메트릭 값
            delta: 변화량 (선택사항)
            color: 카드 색상
            
        Returns:
            HTML 문자열
        """
        
        delta_html = ""
        if delta:
            delta_color = "green" if delta.startswith("+") else "red" if delta.startswith("-") else "gray"
            delta_html = f'<div style="color: {delta_color}; font-size: 0.8rem;">{delta}</div>'
        
        return f"""
        <div class="metric-card" style="border-left: 4px solid {color};">
            <div style="font-size: 0.8rem; color: #666; margin-bottom: 0.5rem;">{title}</div>
            <div style="font-size: 1.5rem; font-weight: bold; color: #333;">{value}</div>
            {delta_html}
        </div>
        """
    
    @staticmethod
    def create_status_badge(text: str, status: str) -> str:
        """
        상태 배지 HTML 생성
        
        Args:
            text: 배지 텍스트
            status: 상태 (success, warning, error)
            
        Returns:
            HTML 문자열
        """
        
        status_class = f"status-{status}"
        
        return f"""
        <span class="status-badge {status_class}">
            {text}
        </span>
        """
    
    @staticmethod
    def format_number(value: Union[int, float], format_type: str = "default") -> str:
        """
        숫자 포맷팅
        
        Args:
            value: 포맷할 숫자
            format_type: 포맷 유형 (default, currency, percentage)
            
        Returns:
            포맷된 문자열
        """
        
        if format_type == "currency":
            if value >= 100000000:  # 1억 이상
                return f"{value/100000000:.1f}억원"
            elif value >= 10000:  # 1만 이상
                return f"{value/10000:.0f}만원"
            else:
                return f"{value:,.0f}원"
        
        elif format_type == "percentage":
            return f"{value:.1f}%"
        
        else:  # default
            return f"{value:,.0f}" if isinstance(value, (int, float)) else str(value)
    
    @staticmethod
    def create_info_box(title: str, content: str, icon: str = "ℹ️") -> str:
        """
        정보 박스 HTML 생성
        
        Args:
            title: 박스 제목
            content: 박스 내용
            icon: 아이콘
            
        Returns:
            HTML 문자열
        """
        
        return f"""
        <div style="border: 1px solid #d1ecf1; border-radius: 5px; padding: 1rem; 
                    background-color: #d1ecf1; margin: 1rem 0;">
            <div style="font-weight: bold; margin-bottom: 0.5rem;">
                {icon} {title}
            </div>
            <div>{content}</div>
        </div>
        """
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """
        텍스트 자르기
        
        Args:
            text: 원본 텍스트
            max_length: 최대 길이
            suffix: 접미사
            
        Returns:
            자른 텍스트
        """
        
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def safe_get_nested_value(data: Dict, keys: List[str], default: Any = None) -> Any:
        """
        중첩된 딕셔너리에서 안전하게 값 가져오기
        
        Args:
            data: 딕셔너리 데이터
            keys: 키 경로 리스트
            default: 기본값
            
        Returns:
            추출된 값 또는 기본값
        """
        
        try:
            current = data
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError, IndexError):
            return default
    
    @staticmethod
    def validate_form_data(data: Dict, required_fields: List[str]) -> Tuple[bool, List[str]]:
        """
        폼 데이터 유효성 검증
        
        Args:
            data: 폼 데이터
            required_fields: 필수 필드 리스트
            
        Returns:
            (유효성 여부, 오류 메시지 리스트)
        """
        
        errors = []
        
        for field in required_fields:
            if field not in data or not data[field]:
                errors.append(f"{field}는 필수 입력 항목입니다.")
        
        # 이메일 검증 (이메일 필드가 있는 경우)
        if "email" in data and data["email"]:
            if "@" not in data["email"] or "." not in data["email"]:
                errors.append("올바른 이메일 형식을 입력해주세요.")
        
        # 전화번호 검증 (전화번호 필드가 있는 경우)
        if "phone" in data and data["phone"]:
            import re
            if not re.match(r"^\d{2,3}-\d{3,4}-\d{4}$", data["phone"]):
                errors.append("올바른 전화번호 형식을 입력해주세요. (예: 010-1234-5678)")
        
        return len(errors) == 0, errors


class DataProcessor:
    """
    데이터 처리 유틸리티 클래스
    
    프론트엔드에서 데이터를 가공하고 변환하는 기능을 제공합니다.
    """
    
    @staticmethod
    def prepare_chart_data(raw_data: List[Dict], x_field: str, y_field: str) -> pd.DataFrame:
        """
        차트용 데이터 준비
        
        Args:
            raw_data: 원시 데이터 리스트
            x_field: X축 필드명
            y_field: Y축 필드명
            
        Returns:
            준비된 데이터프레임
        """
        
        if not raw_data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(raw_data)
            
            # 필요한 컬럼만 선택
            if x_field in df.columns and y_field in df.columns:
                chart_df = df[[x_field, y_field]].copy()
                
                # 결측값 제거
                chart_df = chart_df.dropna()
                
                # 데이터 타입 최적화
                if pd.api.types.is_numeric_dtype(chart_df[y_field]):
                    chart_df[y_field] = pd.to_numeric(chart_df[y_field], errors='coerce')
                
                return chart_df
            else:
                logger.warning(f"필드 {x_field} 또는 {y_field}를 찾을 수 없습니다.")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"차트 데이터 준비 중 오류: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def aggregate_data(data: List[Dict], group_field: str, value_field: str, agg_func: str = "mean") -> Dict:
        """
        데이터 집계
        
        Args:
            data: 원시 데이터
            group_field: 그룹핑 필드
            value_field: 값 필드
            agg_func: 집계 함수 (mean, sum, count, median)
            
        Returns:
            집계된 데이터
        """
        
        if not data:
            return {}
        
        try:
            df = pd.DataFrame(data)
            
            if group_field not in df.columns or value_field not in df.columns:
                return {}
            
            # 집계 실행
            if agg_func == "mean":
                result = df.groupby(group_field)[value_field].mean()
            elif agg_func == "sum":
                result = df.groupby(group_field)[value_field].sum()
            elif agg_func == "count":
                result = df.groupby(group_field)[value_field].count()
            elif agg_func == "median":
                result = df.groupby(group_field)[value_field].median()
            else:
                result = df.groupby(group_field)[value_field].mean()
            
            return result.to_dict()
            
        except Exception as e:
            logger.error(f"데이터 집계 중 오류: {str(e)}")
            return {}
    
    @staticmethod
    def filter_data(data: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """
        데이터 필터링
        
        Args:
            data: 원시 데이터
            filters: 필터 조건 딕셔너리
            
        Returns:
            필터링된 데이터
        """
        
        if not data or not filters:
            return data
        
        filtered_data = []
        
        for item in data:
            include_item = True
            
            for field, condition in filters.items():
                if field not in item:
                    continue
                
                item_value = item[field]
                
                if isinstance(condition, dict):
                    # 범위 조건 (예: {"min": 5, "max": 10})
                    if "min" in condition and item_value < condition["min"]:
                        include_item = False
                        break
                    if "max" in condition and item_value > condition["max"]:
                        include_item = False
                        break
                
                elif isinstance(condition, list):
                    # 포함 조건
                    if item_value not in condition:
                        include_item = False
                        break
                
                else:
                    # 정확한 일치
                    if item_value != condition:
                        include_item = False
                        break
            
            if include_item:
                filtered_data.append(item)
        
        return filtered_data
    
    @staticmethod
    def sort_data(data: List[Dict], sort_field: str, ascending: bool = True) -> List[Dict]:
        """
        데이터 정렬
        
        Args:
            data: 원시 데이터
            sort_field: 정렬 필드
            ascending: 오름차순 여부
            
        Returns:
            정렬된 데이터
        """
        
        if not data:
            return data
        
        try:
            return sorted(
                data,
                key=lambda x: x.get(sort_field, 0),
                reverse=not ascending
            )
        except Exception as e:
            logger.error(f"데이터 정렬 중 오류: {str(e)}")
            return data
    
    @staticmethod
    def calculate_statistics(data: List[float]) -> Dict[str, float]:
        """
        기본 통계 계산
        
        Args:
            data: 숫자 데이터 리스트
            
        Returns:
            통계 정보 딕셔너리
        """
        
        if not data:
            return {}
        
        try:
            import statistics
            
            return {
                "count": len(data),
                "mean": statistics.mean(data),
                "median": statistics.median(data),
                "std": statistics.stdev(data) if len(data) > 1 else 0,
                "min": min(data),
                "max": max(data),
                "range": max(data) - min(data)
            }
            
        except Exception as e:
            logger.error(f"통계 계산 중 오류: {str(e)}")
            return {}


class FileUtils:
    """
    파일 관련 유틸리티 함수들
    
    파일 업로드, 다운로드, 변환 등의 기능을 제공합니다.
    """
    
    @staticmethod
    def save_uploaded_file(uploaded_file, save_directory: str = "uploads") -> Optional[str]:
        """
        업로드된 파일 저장
        
        Args:
            uploaded_file: Streamlit 업로드 파일 객체
            save_directory: 저장 디렉토리
            
        Returns:
            저장된 파일 경로 또는 None
        """
        
        if uploaded_file is None:
            return None
        
        try:
            # 저장 디렉토리 생성
            save_path = Path(save_directory)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 파일 저장
            file_path = save_path / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            logger.info(f"파일 저장 완료: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"파일 저장 실패: {str(e)}")
            return None
    
    @staticmethod
    def create_download_link(data: Union[str, Dict, pd.DataFrame], filename: str, format_type: str = "json") -> str:
        """
        다운로드 링크 생성
        
        Args:
            data: 다운로드할 데이터
            filename: 파일명
            format_type: 파일 형식 (json, csv, txt)
            
        Returns:
            다운로드 HTML 링크
        """
        
        try:
            if format_type == "json":
                if isinstance(data, pd.DataFrame):
                    content = data.to_json(orient="records", ensure_ascii=False)
                else:
                    content = json.dumps(data, ensure_ascii=False, indent=2)
                mime_type = "application/json"
            
            elif format_type == "csv":
                if isinstance(data, pd.DataFrame):
                    content = data.to_csv(index=False)
                else:
                    # 딕셔너리나 리스트를 DataFrame으로 변환
                    df = pd.DataFrame(data)
                    content = df.to_csv(index=False)
                mime_type = "text/csv"
            
            else:  # txt
                content = str(data)
                mime_type = "text/plain"
            
            # Base64 인코딩
            import base64
            b64_content = base64.b64encode(content.encode()).decode()
            
            return f"""
            <a href="data:{mime_type};base64,{b64_content}" 
               download="{filename}" 
               style="text-decoration: none; background-color: #007bff; color: white; 
                      padding: 8px 16px; border-radius: 4px;">
                📥 {filename} 다운로드
            </a>
            """
            
        except Exception as e:
            logger.error(f"다운로드 링크 생성 실패: {str(e)}")
            return "다운로드 링크 생성에 실패했습니다."
    
    @staticmethod
    def validate_file_type(uploaded_file, allowed_types: List[str]) -> bool:
        """
        파일 타입 검증
        
        Args:
            uploaded_file: 업로드된 파일
            allowed_types: 허용된 파일 타입 리스트
            
        Returns:
            유효성 여부
        """
        
        if uploaded_file is None:
            return False
        
        file_extension = Path(uploaded_file.name).suffix.lower()
        return file_extension in allowed_types