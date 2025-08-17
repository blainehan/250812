from __future__ import annotations

import os, re
from datetime import datetime, timezone
from typing import Optional, Tuple, List

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel
from urllib.parse import unquote_plus

APP_VERSION = "5.0.0"  # Kakao Local API only

app = FastAPI(
    title="RealEstate Toolkit (Kakao-only)",
    description="카카오 로컬 API로 주소→PNU, 공공데이터포털로 PNU→건축물대장(표제부)",
    version=APP_VERSION,
)

# ── ENV ─────────────────────────────────────────────────────────────────────────
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")
PUBLICDATA_KEY  = os.getenv("PUBLICDATA_KEY", "")     # 공공데이터포털 일반 인증키(Decoding)
KAKAO_REST_KEY  = os.getenv("KAKAO_REST_KEY", "")     # 카카오 로컬 REST API 키

def require_api_key(x_api_key: Optional[str]):
    if not SERVICE_API_KEY:
        return
    if not x_api_key or x_api_key.strip() != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# ── Utils ──────────────────────────────────────────────────────────────────────
def _fix_input_text(raw: str) -> str:
    """브라우저/툴에서 이중 인코딩/깨짐을 최대한 복구."""
    if not raw:
        return ""
    text = unquote_plus(raw)
    if "%" in text:
        try:
            text2 = unquote_plus(text)
            if text2 != text:
                text = text2
        except Exception:
            pass
    if "\ufffd" in text:
        try:
            text = text.encode("latin-1", "ignore").decode("cp949")
        except Exception:
            pass
    return text.strip()

# ── Schemas ────────────────────────────────────────────────────────────────────
class ConvertReq(BaseModel):
    text: str

class ConvertResp(BaseModel):
    ok: bool
    input: str
    normalized: Optional[str] = None
    full: Optional[str] = None
    admCd10: Optional[str] = None
    bun: Optional[str] = None
    ji: Optional[str] = None
    mtYn: Optional[str] = None
    pnu: Optional[str] = None
    candidates: Optional[List[str]] = None

class BuildingBundle(BaseModel):
    pnu: str
    building: Optional[dict] = None
    lastUpdatedAt: str

# ── Kakao Local API ────────────────────────────────────────────────────────────
KAKAO_ADDR_URL = "https://dapi.kakao.com/v2/local/search/address.json"

async def kakao_search_address(query: str) -> dict:
    if not KAKAO_REST_KEY:
        raise HTTPException(status_code=500, detail="KAKAO_REST_KEY not set")

    headers = {"Authorization": f"KakaoAK {KAKAO_REST_KEY}"}
    params = {"query": query, "analyze_type": "similar"}

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.get(KAKAO_ADDR_URL, headers=headers, params=params)
            r.raise_for_status()
            data = r.json()
            return data
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Kakao error: {e.response.text}") from e
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Kakao upstream error: {e}") from e

def _pick_best_document(docs: list) -> Optional[dict]:
    if not docs:
        return None
    return docs[0]

def _doc_to_pnu_parts(doc: dict) -> Tuple[str, str, str, str, str]:
    addr = doc.get("address") or {}
    road = doc.get("road_address") or {}

    if addr and addr.get("b_code"):
        target = addr
        full_name = addr.get("address_name")
    elif road and road.get("b_code"):
        target = road
        full_name = road.get("address_name")
    else:
        raise HTTPException(status_code=404, detail="Kakao document has no usable b_code")

    b_code = target.get("b_code", "")
    if not re.fullmatch(r"\d{10}", b_code or ""):
        raise HTTPException(status_code=400, detail=f"Invalid b_code from Kakao: {b_code}")

    mountain_yn = target.get("mountain_yn", "N")
    mtYn = "1" if (str(mountain_yn).upper() == "Y") else "0"

    bun_raw = target.get("main_address_no")
    ji_raw  = target.get("sub_address_no")
    try:
        bun = f"{int(bun_raw):04d}" if bun_raw not in (None, "",) else "0000"
    except Exception:
        bun = "0000"
    try:
        ji = f"{int(ji_raw):04d}" if ji_raw not in (None, "",) else "0000"
    except Exception:
        ji = "0000"

    return b_code, mtYn, bun, ji, (full_name or "")

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"service": "RealEstate (Kakao-only)", "version": APP_VERSION}

@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "time": utc_now_iso(),
        "kakao": bool(KAKAO_REST_KEY),
        "publicdata": bool(PUBLICDATA_KEY),
    }

# 추가 엔드포인트들
@app.get("/version")
async def version():
    return {"version": APP_VERSION, "who": "fastapi-index"}

@app.get("/_healthz")
async def _healthz():
    return {
        "ok": True,
        "time": utc_now_iso(),
        "kakao": bool(KAKAO_REST_KEY),
        "publicdata": bool(PUBLICDATA_KEY),
    }

@app.post("/pnu/convert", response_model=ConvertResp)
async def pnu_convert_post(body: ConvertReq, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    fixed = _fix_input_text(body.text)
    data = await kakao_search_address(fixed)

    docs = data.get("documents") or []
    if not docs:
        return ConvertResp(ok=False, input=fixed, candidates=[])

    best = _pick_best_document(docs)
    adm10, mtYn, bun, ji, full = _doc_to_pnu_parts(best)
    pnu = f"{adm10}{mtYn}{bun}{ji}"

    return ConvertResp(
        ok=True, input=fixed, normalized=full, full=full,
        admCd10=adm10, bun=bun, ji=ji, mtYn=mtYn, pnu=pnu, candidates=[]
    )

@app.get("/pnu/convert", response_model=ConvertResp)
async def pnu_convert_get(
    text: str = Query(..., description="예: '서울특별시 서초구 양재동 2-14'"),
    x_api_key: Optional[str] = Header(None),
):
    require_api_key(x_api_key)
    fixed = _fix_input_text(text)
    data = await kakao_search_address(fixed)

    docs = data.get("documents") or []
    if not docs:
        return ConvertResp(ok=False, input=fixed, candidates=[])

    best = _pick_best_document(docs)
    adm10, mtYn, bun, ji, full = _doc_to_pnu_parts(best)
    pnu = f"{adm10}{mtYn}{bun}{ji}"

    return ConvertResp(
        ok=True, input=fixed, normalized=full, full=full,
        admCd10=adm10, bun=bun, ji=ji, mtYn=mtYn, pnu=pnu, candidates=[]
    )

def _split_pnu(pnu: str) -> Tuple[str, str, str, str]:
    if not re.fullmatch(r"\d{19}", pnu):
        raise HTTPException(status_code=400, detail="pnu must be 19 digits")
    adm10 = pnu[:10]
    mtYn  = pnu[10]
    bun   = pnu[11:15]
    ji    = pnu[15:19]
    return adm10, mtYn, bun, ji

@app.get("/realestate/building/by-pnu/{pnu}", response_model=BuildingBundle)
async def building_by_pnu(pnu: str, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)

    if not PUBLICDATA_KEY:
        raise HTTPException(status_code=500, detail="PUBLICDATA_KEY not set")

    adm10, mtYn, bun, ji = _split_pnu(pnu)
    sigunguCd = adm10[:5]
    bjdongCd  = adm10[5:]
    platGbCd  = "1" if mtYn == "1" else "0"

    params = {
        "_type": "json",
        "sigunguCd": sigunguCd,
        "bjdongCd": bjdongCd,
        "platGbCd": platGbCd,
        "bun": bun,
        "ji": ji,
        "numOfRows": 10,
        "pageNo": 1,
        "serviceKey": PUBLICDATA_KEY,
    }
    url = "https://apis.data.go.kr/1613000/BldRgstService_v2/getBrTitleInfo"

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Upstream timeout/error: {e}") from e
        except Exception:
            raise HTTPException(status_code=502, detail="Non-JSON response from upstream")

    building = None
    try:
        body  = (data.get("response") or {}).get("body") or {}
        items = (body.get("items") or {})
        item  = items.get("item")
        if isinstance(item, list):
            building = item[0] if item else None
        elif isinstance(item, dict):
            building = item
    except Exception:
        building = None

    return BuildingBundle(pnu=pnu, building=building, lastUpdatedAt=utc_now_iso())
