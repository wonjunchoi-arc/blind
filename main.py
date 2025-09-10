#!/usr/bin/env python3
"""
BlindInsight AI - 메인 실행 파일

회사 분석 및 커리어 상담을 위한 AI 기반 플랫폼의 메인 진입점입니다.
Streamlit 웹 애플리케이션을 실행합니다.

사용법:
    python main.py
    또는
    streamlit run main.py
"""

import sys
import os
from pathlib import Path
import logging

from dotenv import load_dotenv
load_dotenv()

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('blindinsight.log')
    ]
)

logger = logging.getLogger(__name__)


def check_requirements():
    """필수 패키지 설치 확인"""
    
    required_packages = [
        "streamlit",
        "langchain",
        "langgraph", 
        "chromadb",
        "openai",
        "plotly",
        "pandas",
        "numpy",
        "pydantic"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"다음 패키지들이 설치되지 않았습니다: {', '.join(missing_packages)}")
        logger.info("다음 명령어로 필요한 패키지들을 설치하세요:")
        logger.info("pip install -r requirements.txt")
        return False
    
    return True


def setup_environment():
    """환경 설정"""
    
    # 환경 변수 확인
    required_env_vars = ["OPENAI_API_KEY"]
    
    for var in required_env_vars:
        if not os.getenv(var):
            logger.warning(f"환경변수 {var}이 설정되지 않았습니다.")
            logger.info("OpenAI API 키를 설정해주세요:")
            logger.info(f"export {var}=your_api_key_here")
    
    # 데이터 디렉토리 생성
    data_dirs = [
        "data",
        "data/vector_db", 
        "data/documents",
        "data/cache",
        "logs"
    ]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("환경 설정 완료")


def main():
    """메인 실행 함수"""
    
    logger.info("BlindInsight AI 시작")
    
    # 요구사항 확인
    if not check_requirements():
        sys.exit(1)
    
    # 환경 설정
    setup_environment()
    
    try:
        # Streamlit 앱 실행
        from blindinsight.frontend.app import run_app
        
        logger.info("Streamlit 웹 애플리케이션 시작")
        logger.info("브라우저에서 http://localhost:8501 으로 접속하세요")
        
        # 앱 실행
        run_app()
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 종료되었습니다")
    
    except Exception as e:
        logger.error(f"애플리케이션 실행 중 오류 발생: {str(e)}")
        raise
    
    finally:
        logger.info("BlindInsight AI 종료")


if __name__ == "__main__":
    main()