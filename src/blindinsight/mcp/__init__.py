"""
MCP (Model Context Protocol) 통합 레이어

BlindInsight AI의 외부 데이터 소스 연결 및 통합을 관리합니다.
실시간 회사 정보, 채용 데이터, 시장 동향을 가져오는 기능을 제공합니다.
"""

from .client import *
from .providers import *
from .handlers import *

__all__ = [
    # MCP 클라이언트
    "MCPClient",
    "ConnectionManager",
    "DataSyncManager",
    
    # 데이터 제공자
    "BlindDataProvider",
    "JobSiteProvider", 
    "SalaryDataProvider",
    "CompanyNewsProvider",
    
    # 핸들러
    "DataHandler",
    "CacheManager",
    "RateLimiter"
]