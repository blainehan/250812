import os, re, asyncio
from typing import Optional, Dict, Any
import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime, timezone

app = FastAPI(title="RealEstate Aggregator (Building only)", version="1.1.0")

PUBLICDATA_KEY = os.environ.get("PUBLICDATA_KEY")      # data.go.kr (URL-encoded)
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")    # optional protection

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def require_api_key(x_api_key: Optional[str]):
    if not SERVICE_API_KEY:
        return
    if not x_api_key or x_api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def split_pnu(pnu: str) -> Dict[str, str]:
    if not re.fullmatch(r"\d{19}", pnu):
        raise HTTPException(400, "pnu must be 19 digits")
    return {
        "bjd_code": pnu[:10],   # 법정동코드 10자리
        "plat_gb": pnu[10],     # 0:대지, 1:산
        "bun": pnu[11:15],      # 본번(4)
        "ji": pnu[15:19],       # 부번(4)
    }

async def call_public(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not PUBLICDATA_KEY:
        raise HTTPException(500, "PUBLICDATA_KEY is not set")
    q = {**params, "serviceKey": PUBLICDATA_KEY}
    async with httpx.AsyncClient(timeout=20) as client:
        for i in range(3):
            r = await client.get(url, params=q)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    raise HTTPException(502, "Non-JSON response from upstream")
            await asyncio.sleep(0.3 * (i + 1))
    raise HTTPException(502, "Upstream timeout/error")

# === 응답 스키마: building only ===
class BuildingBundle(BaseModel):
    pnu: str
    building: Dict[str, Any]
    lastUpdatedAt: str

@app.get("/healthz")
async def healthz():
    return {"ok": True, "time": now_iso()}

@app.get("/")
async def root():
    return {"service": "RealEstate Aggregator (Building only)", "version": "1.1.0"}

@app.get("/realestate/building/by-pnu/{pnu}", response_model=BuildingBundle)
async def building_by_pnu(
    pnu: str,
    x_api_key: Optional[str] = Header(None),
):
    """
    건축물대장 표제부(대표) 조회:
    - 서비스: ArchBldgInfoService_v2
    - 파라미터: sigunguCd(5) / bjdongCd(5) / platGbCd(0/1) / bun / ji / _type=json
    """
    require_api_key(x_api_key)
    parts = split_pnu(pnu)
    sigunguCd = parts["bjd_code"][:5]
    bjdongCd = parts["bjd_code"][5:]

    # 표제부(대표) — 엔드포인트는 서비스 문서에 맞춰 필요시 변경
    url = "https://apis.data.go.kr/1613000/ArchBldgInfoService_v2/getArchPmsmInfo"
    resp = await call_public(url, {
        "_type": "json",
        "sigunguCd": sigunguCd,
        "bjdongCd": bjdongCd,
        "platGbCd": parts["plat_gb"],
        "bun": parts["bun"],
        "ji": parts["ji"],
        "numOfRows": 10,
        "pageNo": 1
    })

    # 핵심 필드만 요약 추출(키 이름은 실제 응답 스키마에 맞게 보정)
    item = None
    try:
        items = resp.get("response", {}).get("body", {}).get("items", {})
        it = items.get("item")
        if isinstance(it, list) and it:
            item = it[0]
        elif isinstance(it, dict):
            item = it
    except Exception:
        item = None

    bld = {}
    if isinstance(item, dict):
        # 대표적으로 자주 쓰는 필드들(서비스 실제 필드명에 맞추어 필요시 수정)
        bld = {
            "bldNm": item.get("bldNm"),                     # 건물명
            "mainPurpsCdNm": item.get("mainPurpsCdNm"),     # 주용도
            "strctCdNm": item.get("strctCdNm"),             # 구조
            "useAprDay": item.get("useAprDay"),             # 사용승인일(yyyymmdd)
            "grndFlrCnt": item.get("grndFlrCnt"),           # 지상층수
            "ugrndFlrCnt": item.get("ugrndFlrCnt"),         # 지하층수
            "totArea": item.get("totArea"),                 # 연면적
            "heatingType": item.get("heatMthdCdNm"),        # 난방방식(있다면)
            "roofType": item.get("roofCdNm"),               # 지붕
            "addr": item.get("platPlc") or item.get("newPlatPlc")  # 소재지
        }

    return {"pnu": pnu, "building": bld or resp, "lastUpdatedAt": now_iso()}
