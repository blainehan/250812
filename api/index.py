# api/index.py
import os, re, csv
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="RealEstate (Building: fullname+bunji→PNU)", version="1.5.0")

# 환경변수
PUBLICDATA_KEY = os.environ.get("PUBLICDATA_KEY")      # data.go.kr 건축HUB 키(현재 동작값 그대로)
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")    # 내부 보호키(X-API-Key)

BLD_BASE = "https://apis.data.go.kr/1613000/BldRgstHubService"

# ── PNU 앞10자리 로더 (2열: 코드 + 풀네임) ────────────────────────────────
PNU10_PATHS = [
    Path(__file__).parent.parent / "data" / "pnu10.tsv",
    Path(__file__).parent.parent / "data" / "pnu10.csv",
]
_pnu10_map: Dict[str, str] = {}

def _normalize_fullname(s: str) -> str:
    s = (s or "").strip().replace("\u3000", " ")
    s = " ".join(s.split())
    repl = {
        "서울시": "서울특별시", "서울": "서울특별시",
        "부산시": "부산광역시", "부산": "부산광역시",
        "대구시": "대구광역시", "대구": "대구광역시",
        "인천시": "인천광역시", "인천": "인천광역시",
        "광주시": "광주광역시", "광주": "광주광역시",
        "대전시": "대전광역시", "대전": "대전광역시",
        "울산시": "울산광역시", "울산": "울산광역시",
        "세종": "세종특별자치시",
    }
    for k, v in repl.items():
        if s.startswith(k + " "): s = s.replace(k, v, 1)
    return s

def load_pnu10():
    found = None
    for p in PNU10_PATHS:
        if p.exists():
            found = p; break
    if not found:
        raise RuntimeError(f"pnu10.tsv/csv not found in {PNU10_PATHS[0].parent}")
    sep = "\t" if found.suffix.lower() == ".tsv" else ","
    with open(found, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(sep)
            if len(parts) < 2: continue
            adm = parts[0].strip()
            full = _normalize_fullname(parts[1])
            if re.fullmatch(r"\d{10}", adm):
                _pnu10_map[full] = adm

load_pnu10()

# ── 유틸 ───────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def require_api_key(x_api_key: Optional[str]):
    if SERVICE_API_KEY and x_api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def compose_pnu(adm10: str, mtYn: str, bun: str, ji: str) -> str:
    if not re.fullmatch(r"\d{10}", adm10):
        raise HTTPException(400, "admCd10 must be 10 digits")
    if mtYn not in ("0","1"):
        raise HTTPException(400, "mtYn must be '0' or '1'")
    bun = f"{int(re.sub(r'\\D','', bun or '0')):04d}"
    ji  = f"{int(re.sub(r'\\D','', ji  or '0')):04d}"
    return f"{adm10}{mtYn}{bun}{ji}"

async def call_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": "vercel-fastapi/1.0"}
    async with httpx.AsyncClient(timeout=30, headers=headers) as client:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            raise HTTPException(502, f"Upstream error: {r.status_code} {r.text[:200]}")
        try:
            return r.json()
        except Exception:
            raise HTTPException(502, "Non-JSON response from upstream")

async def hub_title_by_pnu(pnu: str, pageNo: int, numOfRows: int):
    if not PUBLICDATA_KEY:
        raise HTTPException(500, "PUBLICDATA_KEY is not set")
    params = {
        "serviceKey": PUBLICDATA_KEY, "_type": "json",
        "sigunguCd": pnu[:5], "bjdongCd": pnu[5:10],
        "platGbCd": pnu[10], "bun": pnu[11:15], "ji": pnu[15:19],
        "numOfRows": numOfRows, "pageNo": pageNo
    }
    url = f"{BLD_BASE}/getBrTitleInfo"
    return await call_json(url, params)

# ── 응답 스키마 ─────────────────────────────────────────────────────────────
class BuildingBundle(BaseModel):
    pnu: str
    pnuParts: Dict[str, Any]
    building: Dict[str, Any]
    lastUpdatedAt: str

# ── 라우트 ──────────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz(): return {"ok": True, "time": now_iso()}

@app.get("/")
async def root():
    return {"service": "RealEstate (fullname+bunji→PNU)", "version": "1.5.0"}

@app.get("/realestate/building/by-fullname", response_model=BuildingBundle)
async def by_fullname(
    full: str = Query(..., description="예: '서울특별시 종로구 효자동'"),
    bunji: str = Query(..., description="예: '24-5' (부번 없으면 '24')"),
    mtYn: str = Query("0", pattern="^[01]$", description="0=대지, 1=산"),
    pageNo: int = Query(1, ge=1),
    numOfRows: int = Query(10, ge=1, le=100),
    x_api_key: Optional[str] = Header(None),
):
    require_api_key(x_api_key)

    key = _normalize_fullname(full)
    adm10 = _pnu10_map.get(key)
    if not adm10:
        # 비슷한 후보 힌트(최대 5)
        cand = [k for k in _pnu10_map.keys() if key in k][:5]
        raise HTTPException(404, detail={"message": f"No admCd10 for '{key}'", "candidates": cand})

    # bunji 파싱
    if "-" in bunji: a, b = bunji.split("-", 1)
    else: a, b = bunji, "0"
    pnu = compose_pnu(adm10, mtYn, a, b)

    # upstream 조회
    resp = await hub_title_by_pnu(pnu, pageNo, numOfRows)

    # 첫 아이템만 요약
    item = {}
    try:
        items = resp.get("response", {}).get("body", {}).get("items", {})
        it = items.get("item")
        if isinstance(it, list) and it: item = it[0]
        elif isinstance(it, dict): item = it
    except Exception: pass

    return {
        "pnu": pnu,
        "pnuParts": {"admCd10": adm10, "mtYn": mtYn, "bun": pnu[11:15], "ji": pnu[15:19]},
        "building": item or resp,
        "lastUpdatedAt": now_iso()
    }

@app.get("/realestate/building/by-pnu/{pnu}", response_model=BuildingBundle)
async def by_pnu(
    pnu: str,
    x_api_key: Optional[str] = Header(None),
    pageNo: int = Query(1, ge=1),
    numOfRows: int = Query(10, ge=1, le=100),
):
    require_api_key(x_api_key)
    if not re.fullmatch(r"\d{19}", pnu): raise HTTPException(400, "pnu must be 19 digits")
    resp = await hub_title_by_pnu(pnu, pageNo, numOfRows)
    item = {}
    try:
        items = resp.get("response", {}).get("body", {}).get("items", {})
        it = items.get("item")
        if isinstance(it, list) and it: item = it[0]
        elif isinstance(it, dict): item = it
    except Exception: pass
    return {"pnu": pnu, "pnuParts": {"admCd10": pnu[:10], "mtYn": pnu[10], "bun": pnu[11:15], "ji": pnu[15:19]},
            "building": item or resp, "lastUpdatedAt": now_iso()}
