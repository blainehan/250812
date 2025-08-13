# api/index.py
import os, re
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="RealEstate (fullname+bunji→PNU, no JUSO)", version="2.0.0")

# ── 환경변수 ────────────────────────────────────────────────────────────────
PUBLICDATA_KEY = os.environ.get("PUBLICDATA_KEY")      # data.go.kr 인코딩키(지금 쓰던 것)
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")    # 내부 보호키(X-API-Key)

BLD_BASE = "https://apis.data.go.kr/1613000/BldRgstHubService"

# ── 동명→앞10자리 매핑 로더 (CSV/TSV 자동 인식) ────────────────────────────
PNU10_PATHS = [
    Path(__file__).parent.parent / "data" / "pnu10.csv",
    Path(__file__).parent.parent / "data" / "pnu10.tsv",
]
_pnu10_map: Dict[str, str] = {}

def _normalize_fullname(s: str) -> str:
    s = (s or "").replace("\u3000", " ").strip()
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
        if s.startswith(k + " "):
            s = v + s[len(k):]
            break
    return s

def _load_pnu10():
    found = None
    for p in PNU10_PATHS:
        if p.exists():
            found = p; break
    if not found:
        raise RuntimeError(f"pnu10.csv/tsv not found in {PNU10_PATHS[0].parent}")

    sep = "," if found.suffix.lower() == ".csv" else "\t"
    # utf-8-sig 로 읽어 BOM 자동 제거
    with open(found, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            if sep in line:
                left, right = line.split(sep, 1)
            else:
                # 연속 공백으로 분리된 경우도 허용
                parts = line.split()
                if len(parts) >= 2:
                    left, right = parts[0], " ".join(parts[1:])
                else:
                    continue
            code10 = left.strip()
            full = _normalize_fullname(right)
            if re.fullmatch(r"\d{10}", code10) and full:
                _pnu10_map[full] = code10

_load_pnu10()

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
    bun = re.sub(r"\D", "", bun or "0")
    ji  = re.sub(r"\D", "", ji  or "0")
    bun = f"{int(bun):04d}"
    ji  = f"{int(ji):04d}"
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
    return {"service": "RealEstate (fullname+bunji→PNU, no JUSO)", "version": "2.0.0"}

@app.get("/realestate/building/by-fullname", response_model=BuildingBundle)
async def by_fullname(
    full: str = Query(..., description="예: '서울특별시 종로구 청운동'"),
    bunji: str = Query(..., description="예: '24-5' (부번 없으면 '24')"),
    mtYn: str = Query("0", pattern="^[01]$", description="0=대지, 1=산"),
    pageNo: int = Query(1, ge=1),
    numOfRows: int = Query(10, ge=1, le=100),
    x_api_key: Optional[str] = Header(None),
):
    """동 풀네임 + 번지 + 산여부 → PNU 조립 → 표제부 조회"""
    require_api_key(x_api_key)

    key = _normalize_fullname(full)
    adm10 = _pnu10_map.get(key)
    if not adm10:
        # 후보 힌트(최대 5)
        cand = [k for k in _pnu10_map if key in k or k in key][:5]
        raise HTTPException(404, detail={"message": f"No admCd10 for '{key}'", "candidates": cand})

    pnu = compose_pnu(adm10, mtYn, * (bunji.split("-", 1) if "-" in bunji else (bunji, "0")) )
    resp = await hub_title_by_pnu(pnu, pageNo, numOfRows)

    # 첫 레코드만 요약
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
    """PNU(19자리)로 직접 조회"""
    require_api_key(x_api_key)
    if not re.fullmatch(r"\d{19}", pnu):
        raise HTTPException(400, "pnu must be 19 digits")
    resp = await hub_title_by_pnu(pnu, pageNo, numOfRows)
    item = {}
    try:
        items = resp.get("response", {}).get("body", {}).get("items", {})
        it = items.get("item")
        if isinstance(it, list) and it: item = it[0]
        elif isinstance(it, dict): item = it
    except Exception: pass
    return {
        "pnu": pnu,
        "pnuParts": {"admCd10": pnu[:10], "mtYn": pnu[10], "bun": pnu[11:15], "ji": pnu[15:19]},
        "building": item or resp,
        "lastUpdatedAt": now_iso()
    }
