# api/index.py
import os
import re
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="RealEstate Aggregator (Building; Address→PNU supported)", version="1.3.0")

# ── 환경변수 ────────────────────────────────────────────────────────────────
# 공공데이터포털(data.go.kr) 서비스키 (권장: Encoding 키, 기존에 동작하는 값 그대로 사용)
PUBLICDATA_KEY = os.environ.get("PUBLICDATA_KEY")
# Vercel에서 설정한 내부 보호용 키(있으면 호출 시 X-API-Key 필요)
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")
# Business JUSO(도로명주소) 확인키(confmKey)
JUSO_KEY = os.environ.get("JUSO_KEY")

# 건축물대장 HUB 서비스 베이스
BLD_BASE = "https://apis.data.go.kr/1613000/BldRgstHubService"

# ── 공통 유틸 ───────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def require_api_key(x_api_key: Optional[str]):
    """내부 보호용 API 키 검증 (환경변수 설정 시에만 작동)"""
    if SERVICE_API_KEY and x_api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def split_pnu(pnu: str) -> Dict[str, str]:
    """PNU(19자리) → 법정동코드/대지구분/본번/부번 분해"""
    if not re.fullmatch(r"\d{19}", pnu):
        raise HTTPException(400, "pnu must be 19 digits")
    return {
        "bjd_code": pnu[:10],   # 법정동코드(10)
        "plat_gb": pnu[10],     # 0:대지, 1:산
        "bun": pnu[11:15],      # 본번(4)
        "ji": pnu[15:19],       # 부번(4)
    }

def compose_pnu(bjd_code: str, plat_gb: str, bun: str, ji: str) -> str:
    """법정동코드(10) + 대지구분(1:0/1) + 본번(4) + 부번(4) → 19자리 PNU"""
    if not re.fullmatch(r"\d{10}", bjd_code):
        raise HTTPException(400, "bjd_code must be 10 digits")
    if plat_gb not in ("0", "1"):
        raise HTTPException(400, "plat_gb must be '0' or '1'")
    bun = f"{int(bun):04d}" if bun else "0000"
    ji  = f"{int(ji):04d}"  if ji  else "0000"
    return f"{bjd_code}{plat_gb}{bun}{ji}"

async def call_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """공공 API 호출 (재시도/헤더/JSON 강제)"""
    headers = {"Accept": "application/json", "User-Agent": "vercel-fastapi/1.0"}
    last_err = None
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        for i in range(5):  # 최대 5회 재시도
            try:
                r = await client.get(url, params=params)
                if r.status_code == 200:
                    try:
                        return r.json()
                    except Exception:
                        last_err = f"Non-JSON: {r.text[:300]}"
                else:
                    last_err = f"{r.status_code} {r.text[:200]}"
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                last_err = f"timeout {e}"
            await asyncio.sleep(0.6 * (i + 1))
    raise HTTPException(502, f"Upstream timeout/error: {last_err}")

# ── 외부 연동 ───────────────────────────────────────────────────────────────
async def juso_address_to_pnu(addr: str) -> str:
    """
    Business JUSO 주소검색 API로 주소 표준화 후 PNU 생성.
    - admCd(10) → 법정동코드
    - mtYn('0' 대지 / '1' 산) → plat_gb
    - lnbrMnnm / lnbrSlno → 지번 본번/부번
    """
    if not JUSO_KEY:
        raise HTTPException(500, "JUSO_KEY is not set")
    url = "https://business.juso.go.kr/addrlink/addrLinkApi.do"
    params = {
        "confmKey": JUSO_KEY,
        "currentPage": 1,
        "countPerPage": 1,
        "keyword": addr,
        "resultType": "json",
        # 필요 시 옵션:
        # "firstSort": "location",  # 지번 우선정렬
        # "hstryYn": "N",
        # "addInfoYn": "N",
    }
    data = await call_json(url, params)
    results = (data or {}).get("results") or {}
    common = results.get("common") or {}
    err = common.get("errorCode")
    if err and err != "0":
        msg = common.get("errorMessage", "JUSO error")
        raise HTTPException(400, f"JUSO error {err}: {msg}")

    juso_list = results.get("juso") or []
    if not juso_list:
        raise HTTPException(404, "No address match from JUSO")

    j = juso_list[0]
    bjd_code = j.get("admCd")            # 10자리
    plat_gb  = j.get("mtYn") or "0"      # '0' 대지, '1' 산
    bun      = j.get("lnbrMnnm") or "0"  # 본번
    ji       = j.get("lnbrSlno") or "0"  # 부번

    if not (bjd_code and len(bjd_code) == 10 and plat_gb in ("0", "1")):
        raise HTTPException(400, "Invalid data from JUSO")

    return compose_pnu(bjd_code, plat_gb, bun, ji)

async def get_building_title_by_pnu_upstream(pnu: str, pageNo: int, numOfRows: int) -> Dict[str, Any]:
    """건축HUB 표제부(getBrTitleInfo) 조회 (PNU 기반)"""
    if not PUBLICDATA_KEY:
        raise HTTPException(500, "PUBLICDATA_KEY is not set")

    parts = split_pnu(pnu)
    sigunguCd = parts["bjd_code"][:5]
    bjdongCd = parts["bjd_code"][5:]

    url = f"{BLD_BASE}/getBrTitleInfo"
    params = {
        "serviceKey": PUBLICDATA_KEY,
        "_type": "json",
        "sigunguCd": sigunguCd,
        "bjdongCd": bjdongCd,
        "platGbCd": parts["plat_gb"],
        "bun": parts["bun"],
        "ji": parts["ji"],
        "numOfRows": numOfRows,
        "pageNo": pageNo,
    }
    return await call_json(url, params)

# ── 스키마 ──────────────────────────────────────────────────────────────────
class BuildingBundle(BaseModel):
    pnu: str
    building: Dict[str, Any]
    lastUpdatedAt: str

# ── 라우트 ──────────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    return {"ok": True, "time": now_iso()}

@app.get("/")
async def root():
    return {"service": "RealEstate Aggregator (Building; Address→PNU)", "version": "1.3.0"}

@app.get("/realestate/building/by-pnu/{pnu}", response_model=BuildingBundle)
async def get_building_title_info(
    pnu: str,
    x_api_key: Optional[str] = Header(None),
    pageNo: int = Query(1, ge=1),
    numOfRows: int = Query(10, ge=1, le=100),
):
    """건축물대장 표제부(getBrTitleInfo) — PNU 직접 조회"""
    require_api_key(x_api_key)

    resp = await get_building_title_by_pnu_upstream(pnu, pageNo, numOfRows)

    # 첫 레코드만 요약 추출(원본 필요하면 그대로 반환)
    item = {}
    try:
        items = resp.get("response", {}).get("body", {}).get("items", {})
        it = items.get("item")
        if isinstance(it, list) and it:
            item = it[0]
        elif isinstance(it, dict):
            item = it
    except Exception:
        pass

    return {"pnu": pnu, "building": item or resp, "lastUpdatedAt": now_iso()}

@app.get("/realestate/building/by-address", response_model=BuildingBundle)
async def get_building_title_by_address(
    addr: str,
    x_api_key: Optional[str] = Header(None),
    pageNo: int = Query(1, ge=1),
    numOfRows: int = Query(10, ge=1, le=100),
):
    """건축물대장 표제부(getBrTitleInfo) — 주소 입력 → JUSO 변환 → PNU 조회"""
    require_api_key(x_api_key)

    pnu = await juso_address_to_pnu(addr)
    resp = await get_building_title_by_pnu_upstream(pnu, pageNo, numOfRows)

    item = {}
    try:
        items = resp.get("response", {}).get("body", {}).get("items", {})
        it = items.get("item")
        if isinstance(it, list) and it:
            item = it[0]
        elif isinstance(it, dict):
            item = it
    except Exception:
        pass

    return {"pnu": pnu, "building": item or resp, "lastUpdatedAt": now_iso()}
