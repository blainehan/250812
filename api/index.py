
from __future__ import annotations

import os, re, json
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

import httpx
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

APP_VERSION = "7.1.0"  # CSV-only + precise 법정동 disambiguation

SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")
PUBLICDATA_KEY  = os.getenv("PUBLICDATA_KEY", "")
BASE_DIR = os.path.dirname(__file__)
PNU10_CSV_PATH = os.getenv("PNU10_CSV_PATH", os.path.join(BASE_DIR, "pnu10.csv"))


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
    full: Optional[str] = None           # 매칭된 법정동 풀네임
    admCd10: Optional[str] = None        # 10자리 법정동 코드
    bun: Optional[str] = None
    ji: Optional[str] = None
    mtYn: Optional[str] = None
    pnu: Optional[str] = None            # 19자리 PNU (있을 때만)
    source: Optional[str] = None
    candidates: Optional[List[str]] = None

class BuildingBundle(BaseModel):
    pnu: str
    building: Optional[dict] = None
    lastUpdatedAt: str

_DASHES = "－–—"

def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def normalize_address(s: str) -> str:
    # 대표 치환
    s = s.replace("서울시", "서울특별시").replace("서울 특별시", "서울특별시")
    s = s.replace("광역 시", "광역시")
    for ch in _DASHES:
        s = s.replace(ch, "-")
    return _norm_spaces(s)

# ========= pnu_lookup-style precise matching =========

_SI_SYNONYMS = {
    "서울": "서울특별시", "서울시": "서울특별시",
    "부산": "부산광역시", "부산시": "부산광역시",
    "인천": "인천광역시", "인천시": "인천광역시",
    "대구": "대구광역시", "대구시": "대구광역시",
    "대전": "대전광역시", "대전시": "대전광역시",
    "광주": "광주광역시", "광주시": "광주광역시",
    "울산": "울산광역시", "울산시": "울산광역시",
    "세종": "세종특별자치시", "세종시": "세종특별자치시",
    "제주": "제주특별자치도", "제주시": "제주특별자치도",
    "경기": "경기도", "강원": "강원특별자치도",
    "강원도": "강원특별자치도",
    "충북": "충청북도", "충남": "충청남도",
    "전북": "전북특별자치도", "전라북도": "전북특별자치도",
    "전남": "전라남도",
    "경북": "경상북도", "경남": "경상남도",
}
def _canonical_si(token: str) -> str:
    t = token.strip()
    return _SI_SYNONYMS.get(t, t)

def _split_parts(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = _norm_spaces(name).split(" ")
    if not parts:
        return None, None, None
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 2:
        return parts[0], None, parts[1]
    # len >= 3
    return parts[0], parts[1], parts[-1]

_BUN_JI_RE = re.compile(
    r"""
    (?:^|[\s,()])
    (?:산\s*)?
    (?P<bun>\d{1,6})
    (?:\s*-\s*(?P<ji>\d{1,6}))?
    (?!\d)
    """, re.VERBOSE
)
def parse_bunjib(addr: str):
    mt = 1 if re.search(r"\b산\s*\d", addr) else 0
    matches = list(_BUN_JI_RE.finditer(addr))
    if not matches:
        return mt, None, None
    m = matches[-1]
    bun = int(m.group("bun"))
    ji = int(m.group("ji")) if m.group("ji") else 0
    return mt, bun, ji

def _strip_bunjib(addr: str) -> str:
    return _BUN_JI_RE.sub(" ", addr)

class PNUIndex:
    def __init__(self, csv_path: str, encoding: str = "utf-8"):
        self.ok = False
        self.rows: List[Dict[str, str]] = []
        self.by_full: Dict[str, str] = {}
        self.by_sigu_emd: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}
        self.by_emd: Dict[str, List[Tuple[str, str, str, str]]] = {}
        try:
            df = pd.read_csv(csv_path, sep=",", encoding=encoding, low_memory=False)
            if "법정동" not in df.columns or "pnu" not in df.columns:
                raise RuntimeError("CSV에는 '법정동', 'pnu' 컬럼이 필요합니다.")
            df["법정동"] = df["법정동"].astype(str).str.strip()
            df["pnu10"] = df["pnu"].astype(str).str.zfill(10)
            for name, code in zip(df["법정동"], df["pnu10"]):
                full = _norm_spaces(name)
                self.rows.append({"법정동": full, "pnu": code})
            # build indices
            for r in self.rows:
                full = r["법정동"]
                code = r["pnu"]
                si, sigu, emd = _split_parts(full)
                self.by_full[full] = code
                if emd:
                    if sigu:
                        self.by_sigu_emd.setdefault((sigu, emd), []).append((full, code, si))
                    self.by_emd.setdefault(emd, []).append((full, code, si, sigu))
            self.ok = True
        except Exception as e:
            print(f"[PNUIndex] load error: {e}")
            self.ok = False

    @staticmethod
    def build_pnu19(code10: str, mt: int, bun: int, ji: int) -> str:
        return f"{code10}{mt}{bun:04d}{ji:04d}"

    def _lookup_pnu10_from_name(self, name: str) -> Dict[str, Any]:
        """Core logic copied from CLI version (exact disambiguation)."""
        q = _norm_spaces(name)
        if not q:
            return {"ok": False, "error": "질의가 비어 있습니다.", "query": name}

        parts = q.split(" ")
        if parts:
            parts[0] = _canonical_si(parts[0])

        full = " ".join(parts)
        if full in self.by_full:
            return {"ok": True, "admCd10": self.by_full[full], "matched": full}

        if len(parts) >= 3:
            cand = " ".join([_canonical_si(parts[-3]), parts[-2], parts[-1]])
            if cand in self.by_full:
                return {"ok": True, "admCd10": self.by_full[cand], "matched": cand}

        if len(parts) == 2:
            a, b = parts
            # case: '서초구 양재동'
            if a.endswith(("구", "군", "시")):
                key = (a, b)
                if key in self.by_sigu_emd:
                    hits = self.by_sigu_emd[key]
                    if len(hits) == 1:
                        full, code, _si = hits[0]
                        return {"ok": True, "admCd10": code, "matched": full}
                    else:
                        return {
                            "ok": False,
                            "error": "여러 시/도에서 동일한 시군구·법정동 조합이 있습니다. 시도까지 포함해 주세요.",
                            "query": name,
                            "candidates": [h[0] for h in hits],
                        }
            # case: '서울특별시 양재동'
            key_si = _canonical_si(a)
            cands = []
            for full2, code2 in self.by_full.items():
                si, sigu, emd = _split_parts(full2)
                if si == key_si and emd == b:
                    cands.append((full2, code2))
            if len(cands) == 1:
                full2, code2 = cands[0]
                return {"ok": True, "admCd10": code2, "matched": full2}
            elif len(cands) > 1:
                return {
                    "ok": False,
                    "error": "여러 지역에서 일치합니다. 시군구를 포함해 주세요.",
                    "query": name,
                    "candidates": [c[0] for c in cands],
                }

        if len(parts) == 1:
            emd = parts[0]
            hits = self.by_emd.get(emd, [])
            if not hits:
                return {"ok": False, "error": "법정동을 찾지 못했습니다.", "query": name}
            if len(hits) == 1:
                full, code, _si, _sigu = hits[0]
                return {"ok": True, "admCd10": code, "matched": full}
            return {
                "ok": False,
                "error": "여러 지역에서 일치합니다. '서초구 양재동'처럼 시군구를 포함해 주세요.",
                "query": name,
                "candidates": [h[0] for h in hits],
            }

        # Fallback: try tail match like '서울 서초구 양재동'
        tail2 = " ".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        cands = [full2 for full2 in self.by_full.keys() if full2.endswith(tail2)]
        if len(cands) == 1:
            full2 = cands[0]
            return {"ok": True, "admCd10": self.by_full[full2], "matched": full2}
        elif len(cands) > 1:
            return {
                "ok": False,
                "error": "여러 지역에서 일치합니다. 시군구를 포함해 주세요.",
                "query": name,
                "candidates": cands,
            }

        return {"ok": False, "error": "법정동을 찾지 못했습니다.", "query": name}

    def lookup_from_address(self, address: str) -> Dict[str, Any]:
        """
        Accepts a full address line (may include bun/ji). Strips bun/ji and tries
        precise lookup. If that fails, tries to find a full legal-dong name contained in the address.
        """
        cleaned = normalize_address(address)
        name_part = _strip_bunjib(cleaned)
        name_part = _norm_spaces(name_part)

        # primary: precise logic
        res = self._lookup_pnu10_from_name(name_part)
        if res.get("ok") or res.get("candidates"):
            return res

        # secondary: substring search for any known full name inside the address
        hits = [full for full in self.by_full.keys() if full in cleaned]
        if len(hits) == 1:
            full = hits[0]
            return {"ok": True, "admCd10": self.by_full[full], "matched": full}
        elif len(hits) > 1:
            return {
                "ok": False,
                "error": "여러 지역에서 일치합니다. 시군구를 포함해 주세요.",
                "query": address,
                "candidates": hits,
            }
        return {"ok": False, "error": "법정동을 찾지 못했습니다.", "query": address}

# Instantiate index
_index = PNUIndex(PNU10_CSV_PATH)

# ========= FastAPI =========

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
        "pnu10_loaded": {
            "path": PNU10_CSV_PATH if _index.ok else None,
            "entries": len(_index.rows) if _index.ok else 0,
        },
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
        "pnu10_loaded": {
            "path": PNU10_CSV_PATH if _index.ok else None,
            "entries": len(_index.rows) if _index.ok else 0,
        },
    }

@app.post("/pnu/convert", response_model=ConvertResp)
async def pnu_convert_post(body: ConvertReq, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    if not _index.ok:
        raise HTTPException(status_code=500, detail="PNU10 CSV not loaded")
    fixed = _fix_input_text(body.text)
    return _convert_impl(fixed)

@app.get("/pnu/convert", response_model=ConvertResp)
async def pnu_convert_get(
    text: str = Query(..., description="예: '서울특별시 서초구 양재동 2-14'"),
    x_api_key: Optional[str] = Header(None),
):
    require_api_key(x_api_key)
    if not _index.ok:
        raise HTTPException(status_code=500, detail="PNU10 CSV not loaded")
    fixed = _fix_input_text(text)
    return _convert_impl(fixed)

def _convert_impl(address: str) -> ConvertResp:
    addr = normalize_address(address)
    mt, bun, ji = parse_bunjib(addr)

    # 1) find admCd10 with disambiguation
    res10 = _index.lookup_from_address(addr)
    base = {
        "ok": False,
        "input": address,
        "normalized": addr,
        "full": res10.get("matched"),
        "admCd10": res10.get("admCd10"),
        "bun": f"{bun:04d}" if isinstance(bun, int) else None,
        "ji": f"{ji:04d}" if isinstance(ji, int) else None,
        "mtYn": str(mt) if isinstance(mt, int) else None,
        "pnu": None,
        "source": "csv",
        "candidates": res10.get("candidates"),
    }

    # ambiguous or not found → return as-is
    if not res10.get("ok"):
        return ConvertResp(**base)

    # found admCd10; if no bun/ji → just return admCd10 (no 19-digit pnu)
    if bun is None:
        base.update({"ok": True})
        return ConvertResp(**base)

    # build 19-digit
    pnu19 = _index.build_pnu19(res10["admCd10"], mt, bun, ji if ji is not None else 0)
    base.update({"ok": True, "pnu": pnu19})
    return ConvertResp(**base)

def _split_pnu(pnu: str) -> Tuple[str, str, str, str]:
    if not re.fullmatch(r"\d{19}", pnu):
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
