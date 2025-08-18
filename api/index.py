
from __future__ import annotations

import os, re
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

import httpx
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel
from urllib.parse import unquote_plus

APP_VERSION = "7.0.0"  # CSV-only (no Kakao)

SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")
PUBLICDATA_KEY  = os.getenv("PUBLICDATA_KEY", "")
PNU10_CSV_PATH  = os.getenv("PNU10_CSV_PATH", "pnu10.csv")

def require_api_key(x_api_key: Optional[str]):
    if not SERVICE_API_KEY:
        return
    if not x_api_key or x_api_key.strip() != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _fix_input_text(raw: str) -> str:
    if not raw:
        return ""
    from urllib.parse import unquote_plus
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
    source: Optional[str] = None
    candidates: Optional[List[str]] = None

class BuildingBundle(BaseModel):
    pnu: str
    building: Optional[dict] = None
    lastUpdatedAt: str

_DASHES = "－–—"

def _norm_spaces(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s).strip()

def normalize_address(s: str) -> str:
    s = s.replace("서울시", "서울특별시").replace("서울 특별시", "서울특별시")
    s = s.replace("광역 시", "광역시")
    for ch in _DASHES:
        s = s.replace(ch, "-")
    return _norm_spaces(s)

def _aliases_for_name(name: str):
    import re
    aliases = {name}
    if " " in name:
        aliases.add(name.replace(" ", ""))
    m = re.match(r"(.+동)\s*(\d+)가$", name)
    if m:
        aliases.add(m.group(1) + m.group(2) + "가")
    return list(aliases)

def _token_boundary_match(addr: str, candidate: str) -> bool:
    import re
    esc = re.escape(candidate)
    pat = rf"(?<![가-힣A-Za-z0-9]){esc}(?![가-힣A-Za-z])"
    return re.search(pat, addr) is not None

import re as _re
_BUN_JI_RE = _re.compile(
    r"""
    (?:^|[\s,()])
    (?:산\s*)?
    (?P<bun>\d{1,6})
    (?:\s*-\s*(?P<ji>\d{1,6}))?
    (?!\d)
    """, _re.VERBOSE
)

def parse_bunjib(addr: str):
    mt = 1 if _re.search(r"\b산\s*\d", addr) else 0
    matches = list(_BUN_JI_RE.finditer(addr))
    if not matches:
        return mt, None, None
    m = matches[-1]
    bun = int(m.group("bun"))
    ji = int(m.group("ji")) if m.group("ji") else 0
    return mt, bun, ji

class CSVConverter:
    def __init__(self, csv_path: str = "pnu10.csv", encoding="utf-8"):
        self.ok = False
        self.alias_list = []
        try:
            df = pd.read_csv(csv_path, sep=",", encoding=encoding, low_memory=False)
            df["법정동"] = df["법정동"].astype(str).str.strip()
            df["pnu10"] = df["pnu"].astype(str).str.zfill(10)
            items = []
            for name, code in zip(df["법정동"], df["pnu10"]):
                for al in _aliases_for_name(name):
                    items.append((al, code, name))
            items.sort(key=lambda x: len(x[0]), reverse=True)
            self.alias_list = items
            self.ok = True
        except Exception:
            self.ok = False

    @staticmethod
    def build_pnu(code10: str, mt: int, bun: int, ji: int) -> str:
        return f"{code10}{mt}{bun:04d}{ji:04d}"

    def find_best_dong(self, addr: str):
        for alias, code10, canonical in self.alias_list:
            if alias in addr and _token_boundary_match(addr, alias):
                return canonical, code10
        return None, None

    def convert(self, address: str) -> Dict[str, Any]:
        raw = address
        addr = normalize_address(address)
        mt, bun, ji = parse_bunjib(addr)
        name, code10 = self.find_best_dong(addr)
        result = {
            "ok": False,
            "input": raw,
            "normalized": addr,
            "full": None,
            "admCd10": code10,
            "bun": f"{bun:04d}" if isinstance(bun, int) else None,
            "ji": f"{ji:04d}" if isinstance(ji, int) else None,
            "mtYn": str(mt) if isinstance(mt, int) else None,
            "pnu": None,
            "source": "csv",
            "candidates": None,
        }
        if code10 is None:
            return result
        if bun is None:
            return result
        pnu = self.build_pnu(code10, mt, bun, ji if ji is not None else 0)
        result.update({"ok": True, "pnu": pnu, "full": name})
        return result

_csv = CSVConverter(PNU10_CSV_PATH)

from fastapi import FastAPI, Header, HTTPException, Query

app = FastAPI(
    title="RealEstate Toolkit (CSV-only)",
    description="CSV(법정동 10자리) 기반 주소→PNU, 공공데이터포털로 PNU→건축물대장(표제부)",
    version=APP_VERSION,
)

@app.get("/")
async def root():
    return {"service": "RealEstate Toolkit (CSV-only)", "version": APP_VERSION}

@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "time": utc_now_iso(),
        "publicdata": bool(PUBLICDATA_KEY),
        "csv_loaded": _csv.ok,
        "csv_path": PNU10_CSV_PATH if _csv.ok else None,
    }

@app.get("/version")
async def version():
    return {"version": APP_VERSION, "who": "fastapi-index"}

@app.get("/_healthz")
async def _healthz():
    return {
        "ok": True,
        "time": utc_now_iso(),
        "publicdata": bool(PUBLICDATA_KEY),
        "csv_loaded": _csv.ok,
        "csv_path": PNU10_CSV_PATH if _csv.ok else None,
    }

@app.post("/pnu/convert", response_model=ConvertResp)
async def pnu_convert_post(body: ConvertReq, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    fixed = _fix_input_text(body.text)
    if not _csv.ok:
        raise HTTPException(status_code=500, detail="PNU10 CSV not loaded")
    res = _csv.convert(fixed)
    return ConvertResp(**res)

@app.get("/pnu/convert", response_model=ConvertResp)
async def pnu_convert_get(
    text: str = Query(..., description="예: '서울특별시 서초구 양재동 2-14'"),
    x_api_key: Optional[str] = Header(None),
):
    require_api_key(x_api_key)
    fixed = _fix_input_text(text)
    if not _csv.ok:
        raise HTTPException(status_code=500, detail="PNU10 CSV not loaded")
    res = _csv.convert(fixed)
    return ConvertResp(**res)

def _split_pnu(pnu: str) -> Tuple[str, str, str, str]:
    import re as _re
    if not _re.fullmatch(r"\d{19}", pnu):
        raise HTTPException(status_code=400, detail="pnu must be 19 digits")
    return pnu[:10], pnu[10], pnu[11:15], pnu[15:19]

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
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()

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
