import pandas as pd
import logging
from pathlib import Path
from fastmcp.server import FastMCP, Context
from typing import List, Dict, Any, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('merchant_search.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 전역 데이터 저장
DF: Optional[pd.DataFrame] = None

# MCP 서버 초기화
mcp = FastMCP(
    "MerchantSearchServer",
    instructions="""
    신한카드 가맹점을 검색하는 서비스입니다.
    
    사용자가 가맹점명을 입력하면 search_merchant 함수를 사용하여 해당 가맹점의 상세 정보를 검색합니다.
    가맹점명은 부분 일치로 검색되며, 대소문자를 구분하지 않습니다.
    
    검색 결과에는 다음 정보가 포함됩니다:
    - 가맹점명, 업종, 주소, 개설일자
    - 이용건수구간, 이용금액구간
    - 현지인 이용 비중, 영업시간
    - 상세 정보.
    """
)

# 데이터 로드 함수
def _load_df():
    global DF
    DF = pd.read_csv("./data/final_df.csv")
    return DF

# 서버 시작 시 데이터 로드
_load_df()

@mcp.tool()
def search_merchant(merchant_name: str) -> Dict[str, Any]:
    """
    가맹점명을 입력받아 해당 가맹점 정보를 검색합니다. 가맹점명은 데이터의 특성 상, *를 포함할 수도 있습니다. *을 포함하면, *도 함께 가맹점명으로 인식하세요.
    
    매개변수:
      - merchant_name: 검색할 가맹점명 (예: 유유커피, 유유**, 동대*)
    
    반환값:
      - 가맹점 정보가 담긴 딕셔너리
    """
    logger.info(f"search_merchant 함수 실행 시작 - 입력된 가맹점명: '{merchant_name}'")
    
    assert DF is not None, "DataFrame이 초기화되지 않았습니다."

    # 가맹점명 마스킹처리
    original_name = merchant_name
    if len(merchant_name) == 2:
        merchant_name = merchant_name[0] + "*"
    elif len(merchant_name) > 2:
        merchant_name = merchant_name[:2] + "*" * (len(merchant_name) - 2)
    
    logger.info(f"가맹점명 마스킹 처리 완료 - 원본: '{original_name}' -> 마스킹된 이름: '{merchant_name}'")
    
    # 가맹점명으로 검색 (exact match)
    result = DF[DF['가맹점명'].astype(str) == merchant_name]
    
    if len(result) == 0:
        logger.warning(f"검색 결과 없음 - 가맹점명: '{merchant_name}'")
        return {
            "found": False,
            "message": f"'{merchant_name}'에 해당하는 가맹점을 찾을 수 없습니다.",
            "count": 0,
            "merchants": []
        }
    
    # 기본 정보 (가맹점명, id, 주소)
    base_merchants = result[['가맹점명', '가맹점ID', '주소']].to_dict(orient='records')
    logger.info(f"검색 성공 - 가맹점명: '{merchant_name}', 찾은 가맹점 수: {len(base_merchants)}")

    result_data = {
        "found": True,
        "message": f"'{merchant_name}'에 해당하는 가맹점 {len(base_merchants)}개를 찾았습니다.",
        "count": len(base_merchants),
        "merchants": base_merchants
    }
    
    logger.info(f"search_merchant 함수 실행 완료 - 반환 데이터: found={result_data['found']}, count={result_data['count']}")
    return result_data


@mcp.tool()
def get_merchant_detail(merchant_id: str) -> Dict[str, Any]:
    """
    Merchant ID(문자열)로만 상세정보를 검색하는 MCP Tool

    매개변수:
      - merchant_id: 가맹점 Merchant ID (문자열, 예: "000F03E44A")

    반환값:
      {
        "found": bool,             # 검색 성공 여부
        "count": int,              # 검색된 행 수 (0 또는 1)
        "detail": dict | None,     # 단일 결과 row dict
        "message": str             # 안내 메시지
      }
    """
    logger.info(f"get_merchant_detail 시작 - Merchant ID={merchant_id!r}")
    assert DF is not None, "DataFrame이 초기화되지 않았습니다."

    # 가맹점 ID 기준 검색 (정확 매칭)
    sel = DF[DF["가맹점ID"].astype(str) == merchant_id]

    if len(sel) == 0:
        logger.warning(f"{merchant_id!r} 결과 없음")
        return {
            "found": False,
            "count": 0,
            "detail": None,
            "message": f"{merchant_id} 에 해당하는 가맹점 없음"
        }

    # Merchant ID는 유일하다고 가정 → 첫 번째 row만 반환
    detail = sel.iloc[0].to_dict()
    logger.info(f"get_merchant_detail 성공 - {merchant_id!r}")

    return {
        "found": True,
        "count": 1,
        "detail": detail,
        "message": f"{merchant_id} 의 가맹점 상세정보를 찾았습니다."
    }

def _get_metric_columns(df: pd.DataFrame) -> List[str]:
    """비교 지표로 쓸 컬럼만 자동 추출"""
    exclude_cols = {
        "가맹점ID", "기준년월", "주소", "가맹점명",
        "브랜드코드", "지역", "업종", "상권",
        "개설일", "폐업일"
    }
    return [col for col in df.columns if col not in exclude_cols]

@mcp.tool()
def get_compare_industry(merchant_id: str) -> Dict[str, Any]:
    """
    동일 업종 기준으로 비교 지표를 반환하는 MCP Tool
    """
    assert DF is not None, "DataFrame이 초기화되지 않았습니다."
    key = "가맹점ID"

    logger.info(f"[get_compare_industry] 시작 - merchant_id={merchant_id!r}")

    # 대상 추출
    sel = DF[DF[key].astype(str) == str(merchant_id)]
    if len(sel) == 0:
        logger.warning(f"[get_compare_industry] {merchant_id!r} 데이터 없음")
        return {"found": False, "message": f"{merchant_id} 데이터 없음"}

    target = sel.iloc[0].to_dict()
    industry = target.get("업종")
    logger.info(f"[get_compare_industry] 대상 가맹점명={target.get('가맹점명')}, 업종={industry}")

    if not industry:
        logger.warning(f"[get_compare_industry] {merchant_id!r} 업종 정보 없음")
        return {"found": False, "message": "업종 정보 없음"}

    # 비교 집단: 같은 업종 전체
    peers = DF[DF["업종"] == industry]
    logger.info(f"[get_compare_industry] 업종 '{industry}' 비교 집단 크기={len(peers)}")

    if len(peers) == 0:
        logger.warning(f"[get_compare_industry] {industry} 업종 데이터 없음")
        return {"found": False, "message": f"{industry} 업종 데이터 없음"}

    # 자동 추출된 지표
    metrics = _get_metric_columns(DF)
    logger.info(f"[get_compare_industry] 추출된 지표 컬럼 수={len(metrics)}, 예시={metrics[:5]}")

    # 업계 평균 계산
    avg_data = {}
    for m in metrics:
        try:
            vals = pd.to_numeric(peers[m], errors="coerce")
            if vals.notna().any():
                avg = round(vals.mean(), 2)
                avg_data[m] = avg
                logger.debug(f"[get_compare_industry] {m} 평균={avg}")
            else:
                mode_val = peers[m].mode().iloc[0] if not peers[m].mode().empty else None
                avg_data[m] = mode_val
                logger.debug(f"[get_compare_industry] {m} 최빈값={mode_val}")
        except Exception as e:
            logger.error(f"[get_compare_industry] {m} 처리 중 오류: {e}")
            avg_data[m] = None

    logger.info(f"[get_compare_industry] 완료 - merchant_id={merchant_id!r}, metrics={len(metrics)}개")

    return {
        "found": True,
        "merchant_id": merchant_id,
        "industry": industry,
        "metrics": metrics,
        "target": {m: target.get(m) for m in metrics},
        "industry_peers": {
            "count": int(len(peers)),
            "avg": avg_data
        }
    }

if __name__ == "__main__":
    mcp.run()