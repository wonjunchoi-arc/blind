"""
MCP 클라이언트 및 연결 관리 시스템

외부 API와의 안전하고 효율적인 통신을 관리하며,
연결 상태 모니터링, 재시도 로직, 오류 처리를 제공합니다.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import json
from urllib.parse import urljoin

from ..models.base import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """연결 상태 열거형"""
    DISCONNECTED = "disconnected"    # 연결 끊김
    CONNECTING = "connecting"        # 연결 시도 중
    CONNECTED = "connected"          # 연결됨
    ERROR = "error"                 # 오류 상태
    RATE_LIMITED = "rate_limited"   # 속도 제한


@dataclass
class APIEndpoint:
    """API 엔드포인트 정보를 담는 데이터 클래스"""
    
    name: str                        # 엔드포인트 이름
    url: str                         # API URL
    method: str = "GET"              # HTTP 메소드
    headers: Dict[str, str] = field(default_factory=dict)  # 헤더 정보
    rate_limit: int = 100            # 분당 요청 제한
    timeout: int = 30                # 타임아웃 (초)
    retry_attempts: int = 3          # 재시도 횟수
    cache_ttl: int = 300            # 캐시 유지 시간 (초)
    
    def __post_init__(self):
        """초기화 후 검증"""
        if not self.url:
            raise ValueError("API URL이 필요합니다")
        if self.rate_limit <= 0:
            raise ValueError("속도 제한은 0보다 커야 합니다")


class RateLimiter:
    """
    API 요청 속도 제한 관리 클래스
    
    토큰 버킷 알고리즘을 사용하여 요청 속도를 제어합니다.
    """
    
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        """
        속도 제한기 초기화
        
        Args:
            max_requests: 시간 창 내 최대 요청 수
            time_window: 시간 창 크기 (초)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []  # 요청 시간 기록
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        요청 토큰 획득
        
        Returns:
            토큰 획득 성공 여부
        """
        async with self._lock:
            now = time.time()
            
            # 시간 창을 벗어난 요청 기록 제거
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            # 요청 한도 확인
            if len(self.requests) >= self.max_requests:
                # 다음 요청 가능 시간 계산
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request)
                
                if wait_time > 0:
                    logger.warning(f"속도 제한 도달, {wait_time:.1f}초 대기")
                    await asyncio.sleep(wait_time)
                    return await self.acquire()  # 재귀 호출
            
            # 현재 요청 시간 기록
            self.requests.append(now)
            return True
    
    def get_remaining_requests(self) -> int:
        """남은 요청 가능 횟수 반환"""
        now = time.time()
        recent_requests = [req_time for req_time in self.requests 
                          if now - req_time < self.time_window]
        return max(0, self.max_requests - len(recent_requests))


class CacheManager:
    """
    API 응답 캐시 관리 클래스
    
    메모리 기반 캐시로 API 응답을 저장하고 관리합니다.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        캐시 관리자 초기화
        
        Args:
            max_size: 최대 캐시 항목 수
        """
        self.max_size = max_size
        self.cache: Dict[str, Dict] = {}  # 캐시 저장소
        self._access_times: Dict[str, float] = {}  # 접근 시간 기록
    
    def _generate_key(self, url: str, params: Dict = None) -> str:
        """
        캐시 키 생성
        
        Args:
            url: API URL
            params: 요청 매개변수
            
        Returns:
            캐시 키
        """
        import hashlib
        key_data = f"{url}_{params or {}}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, url: str, params: Dict = None, ttl: int = 300) -> Optional[Dict]:
        """
        캐시에서 데이터 조회
        
        Args:
            url: API URL
            params: 요청 매개변수
            ttl: 캐시 유지 시간 (초)
            
        Returns:
            캐시된 데이터 또는 None
        """
        cache_key = self._generate_key(url, params)
        
        if cache_key in self.cache:
            cache_data = self.cache[cache_key]
            cached_at = cache_data.get("cached_at", 0)
            
            # 캐시 만료 확인
            if time.time() - cached_at < ttl:
                self._access_times[cache_key] = time.time()
                logger.debug(f"캐시 히트: {cache_key}")
                return cache_data.get("data")
            else:
                # 만료된 캐시 제거
                self._remove_cache_item(cache_key)
        
        return None
    
    def set(self, url: str, data: Dict, params: Dict = None):
        """
        캐시에 데이터 저장
        
        Args:
            url: API URL
            data: 저장할 데이터
            params: 요청 매개변수
        """
        cache_key = self._generate_key(url, params)
        
        # 캐시 크기 제한 확인
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[cache_key] = {
            "data": data,
            "cached_at": time.time()
        }
        self._access_times[cache_key] = time.time()
        
        logger.debug(f"캐시 저장: {cache_key}")
    
    def _remove_cache_item(self, cache_key: str):
        """캐시 항목 제거"""
        if cache_key in self.cache:
            del self.cache[cache_key]
        if cache_key in self._access_times:
            del self._access_times[cache_key]
    
    def _evict_oldest(self):
        """가장 오래된 캐시 항목 제거"""
        if self._access_times:
            oldest_key = min(self._access_times.keys(), 
                           key=lambda k: self._access_times[k])
            self._remove_cache_item(oldest_key)
            logger.debug(f"오래된 캐시 제거: {oldest_key}")
    
    def clear(self):
        """모든 캐시 제거"""
        self.cache.clear()
        self._access_times.clear()
        logger.info("캐시 전체 제거")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 반환"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "memory_usage_mb": len(str(self.cache)) / 1024 / 1024
        }


class ConnectionManager:
    """
    연결 관리 클래스
    
    외부 API와의 연결을 관리하고 상태를 모니터링합니다.
    """
    
    def __init__(self):
        """연결 관리자 초기화"""
        self.connections: Dict[str, aiohttp.ClientSession] = {}
        self.connection_status: Dict[str, ConnectionStatus] = {}
        self.last_health_check: Dict[str, datetime] = {}
        self.health_check_interval = 300  # 5분마다 헬스 체크
    
    async def get_session(self, service_name: str) -> aiohttp.ClientSession:
        """
        서비스별 HTTP 세션 조회
        
        Args:
            service_name: 서비스 이름
            
        Returns:
            HTTP 클라이언트 세션
        """
        if service_name not in self.connections:
            # 새 세션 생성
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
            
            session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "User-Agent": f"BlindInsight-AI/{settings.version}",
                    "Accept": "application/json"
                }
            )
            
            self.connections[service_name] = session
            self.connection_status[service_name] = ConnectionStatus.CONNECTED
            
            logger.info(f"{service_name} 연결 생성")
        
        return self.connections[service_name]
    
    async def health_check(self, service_name: str, health_url: str) -> bool:
        """
        서비스 헬스 체크
        
        Args:
            service_name: 서비스 이름
            health_url: 헬스 체크 URL
            
        Returns:
            헬스 체크 성공 여부
        """
        try:
            session = await self.get_session(service_name)
            
            async with session.get(health_url) as response:
                if response.status == 200:
                    self.connection_status[service_name] = ConnectionStatus.CONNECTED
                    self.last_health_check[service_name] = datetime.now()
                    return True
                else:
                    self.connection_status[service_name] = ConnectionStatus.ERROR
                    return False
                    
        except Exception as e:
            logger.error(f"{service_name} 헬스 체크 실패: {str(e)}")
            self.connection_status[service_name] = ConnectionStatus.ERROR
            return False
    
    def get_connection_status(self, service_name: str) -> ConnectionStatus:
        """연결 상태 조회"""
        return self.connection_status.get(service_name, ConnectionStatus.DISCONNECTED)
    
    async def close_all(self):
        """모든 연결 종료"""
        for service_name, session in self.connections.items():
            try:
                await session.close()
                logger.info(f"{service_name} 연결 종료")
            except Exception as e:
                logger.error(f"{service_name} 연결 종료 실패: {str(e)}")
        
        self.connections.clear()
        self.connection_status.clear()


class DataSyncManager:
    """
    데이터 동기화 관리 클래스
    
    외부 데이터와 내부 지식 베이스 간의 동기화를 관리합니다.
    """
    
    def __init__(self, knowledge_base):
        """
        데이터 동기화 관리자 초기화
        
        Args:
            knowledge_base: 지식 베이스 인스턴스
        """
        self.knowledge_base = knowledge_base
        self.sync_tasks: Dict[str, asyncio.Task] = {}
        self.sync_status: Dict[str, Dict] = {}
        self.sync_schedules: Dict[str, Dict] = {}
    
    def schedule_sync(
        self, 
        provider_name: str, 
        sync_function: Callable,
        interval_hours: int = 24
    ):
        """
        정기적인 데이터 동기화 스케줄링
        
        Args:
            provider_name: 데이터 제공자 이름
            sync_function: 동기화 함수
            interval_hours: 동기화 간격 (시간)
        """
        self.sync_schedules[provider_name] = {
            "function": sync_function,
            "interval": interval_hours * 3600,  # 초 단위로 변환
            "last_sync": None,
            "next_sync": datetime.now()
        }
        
        logger.info(f"{provider_name} 동기화 스케줄 등록: {interval_hours}시간 간격")
    
    async def run_sync(self, provider_name: str) -> bool:
        """
        즉시 동기화 실행
        
        Args:
            provider_name: 데이터 제공자 이름
            
        Returns:
            동기화 성공 여부
        """
        if provider_name not in self.sync_schedules:
            logger.error(f"등록되지 않은 동기화 제공자: {provider_name}")
            return False
        
        try:
            sync_info = self.sync_schedules[provider_name]
            sync_function = sync_info["function"]
            
            # 동기화 상태 업데이트
            self.sync_status[provider_name] = {
                "status": "running",
                "started_at": datetime.now(),
                "progress": 0
            }
            
            logger.info(f"{provider_name} 데이터 동기화 시작")
            
            # 동기화 실행
            result = await sync_function()
            
            # 동기화 완료
            self.sync_status[provider_name].update({
                "status": "completed",
                "completed_at": datetime.now(),
                "progress": 100,
                "result": result
            })
            
            # 다음 동기화 시간 업데이트
            sync_info["last_sync"] = datetime.now()
            sync_info["next_sync"] = datetime.now() + timedelta(
                seconds=sync_info["interval"]
            )
            
            logger.info(f"{provider_name} 데이터 동기화 완료")
            return True
            
        except Exception as e:
            # 동기화 실패
            self.sync_status[provider_name].update({
                "status": "failed",
                "failed_at": datetime.now(),
                "error": str(e)
            })
            
            logger.error(f"{provider_name} 데이터 동기화 실패: {str(e)}")
            return False
    
    async def start_scheduler(self):
        """백그라운드 스케줄러 시작"""
        async def scheduler_loop():
            while True:
                try:
                    now = datetime.now()
                    
                    for provider_name, sync_info in self.sync_schedules.items():
                        # 동기화 시간 확인
                        if now >= sync_info["next_sync"]:
                            # 이미 실행 중이 아닌 경우만 동기화 실행
                            current_status = self.sync_status.get(provider_name, {}).get("status")
                            if current_status != "running":
                                # 비동기 태스크로 동기화 실행
                                task = asyncio.create_task(self.run_sync(provider_name))
                                self.sync_tasks[provider_name] = task
                    
                    # 1분마다 체크
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"스케줄러 오류: {str(e)}")
                    await asyncio.sleep(60)
        
        # 스케줄러 태스크 시작
        self.scheduler_task = asyncio.create_task(scheduler_loop())
        logger.info("데이터 동기화 스케줄러 시작")
    
    def get_sync_status(self, provider_name: str = None) -> Dict:
        """동기화 상태 조회"""
        if provider_name:
            return self.sync_status.get(provider_name, {"status": "not_scheduled"})
        else:
            return self.sync_status
    
    async def stop_scheduler(self):
        """스케줄러 중지"""
        if hasattr(self, 'scheduler_task'):
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # 진행 중인 동기화 태스크 정리
        for task in self.sync_tasks.values():
            if not task.done():
                task.cancel()
        
        logger.info("데이터 동기화 스케줄러 중지")


class MCPClient:
    """
    MCP 메인 클라이언트 클래스
    
    외부 데이터 소스와의 통합 관리를 담당하는 중앙 클라이언트입니다.
    """
    
    def __init__(self, knowledge_base=None):
        """
        MCP 클라이언트 초기화
        
        Args:
            knowledge_base: 지식 베이스 인스턴스
        """
        self.knowledge_base = knowledge_base
        
        # 관리자 인스턴스들
        self.connection_manager = ConnectionManager()
        self.cache_manager = CacheManager()
        self.rate_limiter = RateLimiter()
        self.sync_manager = DataSyncManager(knowledge_base) if knowledge_base else None
        
        # API 엔드포인트 저장소
        self.endpoints: Dict[str, APIEndpoint] = {}
        
        # 데이터 제공자들
        self.providers: Dict[str, Any] = {}
        
        logger.info("MCP 클라이언트 초기화 완료")
    
    def register_endpoint(self, endpoint: APIEndpoint):
        """
        API 엔드포인트 등록
        
        Args:
            endpoint: API 엔드포인트 정보
        """
        self.endpoints[endpoint.name] = endpoint
        logger.info(f"API 엔드포인트 등록: {endpoint.name}")
    
    def register_provider(self, name: str, provider):
        """
        데이터 제공자 등록
        
        Args:
            name: 제공자 이름
            provider: 제공자 인스턴스
        """
        self.providers[name] = provider
        logger.info(f"데이터 제공자 등록: {name}")
    
    async def make_request(
        self,
        endpoint_name: str,
        params: Dict = None,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> Optional[Dict]:
        """
        API 요청 실행
        
        Args:
            endpoint_name: 엔드포인트 이름
            params: 요청 매개변수
            use_cache: 캐시 사용 여부
            force_refresh: 강제 새로고침 여부
            
        Returns:
            응답 데이터 또는 None
        """
        if endpoint_name not in self.endpoints:
            logger.error(f"등록되지 않은 엔드포인트: {endpoint_name}")
            return None
        
        endpoint = self.endpoints[endpoint_name]
        
        # 캐시 확인 (강제 새로고침이 아닌 경우)
        if use_cache and not force_refresh:
            cached_data = self.cache_manager.get(
                endpoint.url, params, endpoint.cache_ttl
            )
            if cached_data:
                return cached_data
        
        # 속도 제한 확인
        await self.rate_limiter.acquire()
        
        try:
            # HTTP 세션 가져오기
            session = await self.connection_manager.get_session(endpoint_name)
            
            # 요청 실행
            request_kwargs = {
                "timeout": aiohttp.ClientTimeout(total=endpoint.timeout)
            }
            
            if endpoint.method.upper() == "GET":
                request_kwargs["params"] = params
            else:
                request_kwargs["json"] = params
            
            # 커스텀 헤더 추가
            if endpoint.headers:
                request_kwargs["headers"] = endpoint.headers
            
            async with session.request(
                endpoint.method, 
                endpoint.url, 
                **request_kwargs
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    # 캐시에 저장
                    if use_cache:
                        self.cache_manager.set(endpoint.url, data, params)
                    
                    return data
                
                elif response.status == 429:  # 속도 제한
                    logger.warning(f"API 속도 제한 도달: {endpoint_name}")
                    return None
                
                else:
                    logger.error(f"API 요청 실패: {endpoint_name}, 상태코드: {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"API 요청 타임아웃: {endpoint_name}")
            return None
        except Exception as e:
            logger.error(f"API 요청 오류: {endpoint_name}, {str(e)}")
            return None
    
    async def health_check_all(self) -> Dict[str, bool]:
        """모든 등록된 서비스의 헬스 체크"""
        results = {}
        
        for endpoint_name in self.endpoints:
            # 간단한 헬스 체크 (실제 요청으로 확인)
            result = await self.make_request(endpoint_name, use_cache=False)
            results[endpoint_name] = result is not None
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """MCP 시스템 상태 정보 반환"""
        return {
            "endpoints": {
                name: {
                    "url": endpoint.url,
                    "method": endpoint.method,
                    "rate_limit": endpoint.rate_limit,
                    "cache_ttl": endpoint.cache_ttl
                }
                for name, endpoint in self.endpoints.items()
            },
            "providers": list(self.providers.keys()),
            "cache_stats": self.cache_manager.get_stats(),
            "rate_limiter": {
                "remaining_requests": self.rate_limiter.get_remaining_requests(),
                "max_requests": self.rate_limiter.max_requests,
                "time_window": self.rate_limiter.time_window
            },
            "sync_status": self.sync_manager.get_sync_status() if self.sync_manager else {}
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            # 동기화 스케줄러 중지
            if self.sync_manager:
                await self.sync_manager.stop_scheduler()
            
            # 연결 종료
            await self.connection_manager.close_all()
            
            # 캐시 정리
            self.cache_manager.clear()
            
            logger.info("MCP 클라이언트 정리 완료")
            
        except Exception as e:
            logger.error(f"MCP 클라이언트 정리 중 오류: {str(e)}")