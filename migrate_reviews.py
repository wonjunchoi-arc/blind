#!/usr/bin/env python3
"""
청크 데이터 마이그레이션 스크립트

tools/data의 chunk_first_vectordb JSON 파일들을 ChromaDB 벡터 데이터베이스로 마이그레이션합니다.
OpenAI text-embedding-3-small 모델을 사용하여 임베딩을 생성합니다.
"""
#migrate_reviews.py

import asyncio
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from blindinsight.rag.json_processor import ChunkDataLoader
from blindinsight.models.base import settings


async def main():
    """메인 마이그레이션 함수"""
    print("=" * 60)
    print("[처리중] BlindInsight Review Data Migration")
    print("=" * 60)
    
    # 환경 변수 확인
    if not settings.openai_api_key:
        print("[오류] OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        return False
    
    # 데이터 디렉토리 확인
    data_dir = project_root / "data" / "hybrid_vectordb"
    if not data_dir.exists():
        print(f"[오류] 데이터 디렉토리를 찾을 수 없습니다: {data_dir}")
        return False
    
    print(f"[폴더] 데이터 디렉토리: {data_dir}")
    print(f"[모델] 임베딩 모델: text-embedding-3-small (1536차원)")
    print(f"[저장] 벡터 DB: {settings.vector_db_path}")
    print()
    
    # ChunkDataLoader 초기화
    try:
        loader = ChunkDataLoader(str(data_dir))
        print("[성공] ChunkDataLoader 초기화 완료")
    except Exception as e:
        print(f"[오류] ChunkDataLoader 초기화 실패: {e}")
        return False
    
    # 사용자 확인
    response = input("마이그레이션을 시작하시겠습니까? (y/n): ").lower()
    if response != 'y':
        print("마이그레이션이 취소되었습니다.")
        return False
    
    print("\n[시작] 마이그레이션 시작...")
    start_time = time.time()
    
    try:
        # 모든 청크 데이터 로드 및 벡터화
        success = await loader.load_all_chunks()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if success:
            # 통계 정보 출력
            stats = loader.get_stats()
            print("\n" + "=" * 60)
            print("[완료] 마이그레이션 완료!")
            print("=" * 60)
            print(f"[통계] 처리된 파일: {stats['files_processed']}개")
            print(f"[문서] 생성된 문서: {stats['documents_created']}개")
            print(f"[회사] 처리된 회사: {len(stats['companies_processed'])}개")
            print(f"[시간] 총 처리 시간: {processing_time:.2f}초")
            
            if stats['companies_processed']:
                print(f"\n처리된 회사 목록:")
                for i, company in enumerate(sorted(stats['companies_processed']), 1):
                    print(f"  {i:2d}. {company}")
            
            # 데이터 무결성 검증
            print("\n[검증] 데이터 무결성 검증 중...")
            verification = await loader.verify_data_integrity()
            
            print(f"[문서] 총 저장된 문서: {verification['total_documents']}개")
            print("\n컬렉션별 문서 수:")
            for collection, stats in verification['collections_status'].items():
                count = stats.get('document_count', 0)
                print(f"  • {collection}: {count}개")
            
            if verification['errors']:
                print(f"\n[경고] 검증 중 발견된 오류:")
                for error in verification['errors']:
                    print(f"  - {error}")
            
        else:
            print("\n[실패] 마이그레이션 실패")
            return False
    
    except KeyboardInterrupt:
        print("\n\n[중단] 사용자에 의해 중단되었습니다.")
        return False
    except Exception as e:
        print(f"\n[오류] 마이그레이션 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[완료] 마이그레이션이 성공적으로 완료되었습니다!")
    return True


async def migrate_single_company(company_name: str):
    """특정 회사만 마이그레이션"""
    print(f"[회사] {company_name} 마이그레이션 시작...")
    
    data_dir = project_root / "data"
    loader = ChunkDataLoader(str(data_dir))
    
    success = await loader.load_single_company(company_name)
    
    if success:
        stats = loader.get_stats()
        print(f"[성공] {company_name} 마이그레이션 완료: {stats['documents_created']}개 문서")
    else:
        print(f"[실패] {company_name} 마이그레이션 실패")
    
    return success


def show_help():
    """도움말 출력"""
    print("""
사용법:
  python migrate_reviews.py              - 모든 회사 데이터 마이그레이션
  python migrate_reviews.py [회사명]     - 특정 회사만 마이그레이션
  python migrate_reviews.py --help       - 도움말 출력

예시:
  python migrate_reviews.py 네이버       - 네이버 청크 데이터만 마이그레이션
  python migrate_reviews.py 카카오       - 카카오 청크 데이터만 마이그레이션

환경 변수:
  OPENAI_API_KEY                         - OpenAI API 키 (필수)

출력:
  ChromaDB 벡터 데이터베이스에 다음 6개 컬렉션으로 저장:
  - company_culture      - 회사 문화 관련 청크
  - work_life_balance    - 워라밸 관련 청크
  - management           - 경영진/관리 관련 청크
  - salary_benefits      - 연봉/복지 관련 청크
  - career_growth        - 커리어 성장 관련 청크
  - general              - 일반적인 정보 청크
    """)


if __name__ == "__main__":
    # 명령행 인수 처리
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h", "help"]:
            show_help()
            sys.exit(0)
        
        # 특정 회사 마이그레이션
        company_name = sys.argv[1]
        print(f"특정 회사 마이그레이션 모드: {company_name}")
        asyncio.run(migrate_single_company(company_name))
    else:
        # 전체 마이그레이션
        asyncio.run(main())