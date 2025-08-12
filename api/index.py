# api/index.py
import os, re, asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="RealEstate Aggregator (Building only)", version="1.2.0")

# ▶ 공공데이터포털 인코딩(Encoding) 키 사용!  (%가 포함된 URL-인코딩된 값)
PUBLICDATA_KEY = os.environ.get("PUBLICDATA_KEY")
# ▶ 우리 API 보호용 (있으면 호출 시 X-API-Key 필요)
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")

BASE = "https://apis.data.go.kr/1613000/BldRgstHubService"  # 건축HUB

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def require_api_key(x_api_key: Optional[str]):
    if SERVICE_API_KEY and x_api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def split_pnu(pnu: str) -> Dict[str, str]:
    if not re.fullmatch(r"\d{19}", pnu):
        raise HTTPException(400, "pnu must be 19 digits")
    return {
        "bjd_code": pnu[:10],      # 법정동 10
        "plat_gb": pnu[10],        # 0:대지 1:산
        "bun": pnu[11:15],         # 본번
        "ji": pnu[15:19],          # 부번
    }

async def call_public(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not PUBLICDATA_KEY:
        raise HTTPException(500, "PUBLICDATA_KEY is not set")
    q = {**params, "serviceKey": PUBLICDATA_KEY}
    headers = {"Accept": "application/json", "User-Agent": "vercel-fastapi/1.0"}
    last_err = None
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        for i in range(5):  # 재시도
            try:
                r = await client.get(url, params=q)
                if r.status_code == 200:
                    try:
                        return r.json()
                    except Exception:
                        raise HTTPException(502, "Non-JSON response from upstream")
                else:
                    last_err = f"{r.status_code} {r.text[:200]}"
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                last_err = f"timeout {e}"
            await asyncio.sleep(0.6*(i+1))
    raise HTTPException(502, f"Upstream timeout/error: {last_err}")

class BuildingBundle(BaseModel):
    pnu: str
    building: Dict[str, Any]
    lastUpdatedAt: str

@app.get("/healthz")
async def healthz():
    return {"ok": True, "time": now_iso()}

@app.get("/realestate/building/by-pnu/{pnu}", response_model=BuildingBundle)
async def get_building_title_info(
    pnu: str,
    x_api_key: Optional[str] = Header(None),
    pageNo: int = Query(1, ge=1),
    numOfRows: int = Query(10, ge=1, le=100),
):
    """
    건축HUB 건축물대장 '표제부' 조회(getBrTitleInfo)만 수행
    필요한 파라미터: sigunguCd, bjdongCd, platGbCd, bun, ji, _type=json
    """
    require_api_key(x_api_key)

    parts = split_pnu(pnu)
    sigunguCd = parts["bjd_code"][:5]
    bjdongCd = parts["bjd_code"][5:]

    url = f"{BASE}/getBrTitleInfo"
    params = {
        "_type": "json",
        "sigunguCd": sigunguCd,
        "bjdongCd": bjdongCd,
        "platGbCd": parts["plat_gb"],
        "bun": parts["bun"],
        "ji": parts["ji"],
        "numOfRows": numOfRows,
        "pageNo": pageNo,
    }

    resp = await call_public(url, params)

    # 첫 레코드만 요약 추출(필드명은 문서 응답 기준)
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

    return {"pnu": pnu, "building": item, "lastUpdatedAt": now_iso()}
