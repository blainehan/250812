# api/index.py
import os, re, csv
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Body
from pydantic import BaseModel

app = FastAPI(title="RealEstate Toolkit", version="4.1.0")

# ── ENV ─────────────────────────────────────────────────────────────────────
PUBLICDATA_KEY = os.environ.get("PUBLICDATA_KEY")      # data.go.kr (Encoding Key)
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")    # Internal API key (X-API-Key header)

BLD_BASE = "https://apis.data.go.kr/1613000/BldRgstHubService"

# ── Helpers ─────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def require_api_key(x_api_key: Optional[str]):
    if SERVICE_API_KEY and x_api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

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

# ── PNU10 loader (동 풀네임 → 10자리코드) ───────────────────────────────────
_PNU10_FILES = [
    Path(__file__).parent.parent / "data" / "pnu10.csv",
    Path(__file__).parent.parent / "data" / "pnu10.tsv",
    Path(__file__).parent / "data" / "pnu10.csv",
    Path(__file__).parent / "data" / "pnu10.tsv",
    Path.cwd() / "data" / "pnu10.csv",
    Path.cwd() / "data" / "pnu10.tsv",
]
_PNU_MAP: Dict[str, str] = {}   # {"서울특별시 서초구 양재동": "1165010200", ...}
_PNU_META: Dict[str, Any] = {"path": None, "entries": 0, "delimiter": None, "reversed_cols": False}

def _norm_name(s: str) -> str:
    if not s: return ""
    s = s.replace("\u3000", " ").strip()
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

def _load_pnu10_once():
    global _PNU_MAP
    if _PNU_MAP:  # already loaded
        return
    found = None
    for p in _PNU10_FILES:
        if p.exists():
            found = p; break
    if not found:
        # 변환기만 제한되고 나머지 라우트는 동작 가능
        return
    with open(found, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096); f.seek(0)
        if "\t" in sample and sample.count("\t") >= sample.count(","):
            delim = "\t"
        elif "," in sample:
            delim = ","
        else:
            delim = None
        if delim:
            rows = list(csv.reader(f, delimiter=delim))
            _PNU_META["delimiter"] = "\\t" if delim == "\t" else ","
        else:
            rows = [line.strip().split() for line in f if line.strip()]
            _PNU_META["delimiter"] = "whitespace"

    reversed_cols = False
    for row in rows:
        if not row: continue
        if _PNU_META["delimiter"] == "whitespace" and len(row) >= 2:
            left, right = row[0], " ".join(row[1:])
        elif len(row) >= 2:
            left, right = row[0], row[1]
        else:
            continue
        left = left.strip().strip('"').strip("'")
        right = right.strip().strip('"').strip("'")
        if re.fullmatch(r"\d{10}", left):
            code10, full = left, _norm_name(right)
        elif re.fullmatch(r"\d{10}", right):
            code10, full = right, _norm_name(left); reversed_cols = True
        else:
            continue
        if full and re.fullmatch(r"\d{10}", code10):
            _PNU_MAP[full] = code10

    _PNU_META["path"] = str(found)
    _PNU_META["entries"] = len(_PNU_MAP)
    _PNU_META["reversed_cols"] = reversed_cols

@app.on_event("startup")
async def _startup():
    _load_pnu10_once()

# ── Text → PNU ──────────────────────────────────────────────────────────────
def _parse_text_to_parts(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """
    입력 예:
      '서울특별시 서초구 양재동 2-14'
      '서초구 양재동 산2-14'
      '양재동 2-14'
    반환: (name_part, bun, ji, mtYn)  ; name_part가 None이면 파싱 실패
    """
    s = (text or "").strip()
    s = s.replace("　", " ").replace(",", " ").replace("(", " ").replace(")", " ")
    s = " ".join(s.split())
    mtYn = "0"

    m = re.search(r"(산)\s*(\d+)(?:-(\d+))?$", s)
    if m:
        mtYn = "1"
        bun = m.group(2)
        ji = m.group(3) or "0"
        name_part = s[:m.start()].strip()
    else:
        m2 = re.search(r"(\d+)(?:-(\d+))?$", s)
        if not m2:
            return None, None, None, "0"
        bun = m2.group(1)
        ji = m2.group(2) or "0"
        name_part = s[:m2.start()].strip()
    if not name_part:
        return None, None, None, mtYn
    return name_part, bun, ji, mtYn

def _find_adm10(name_part: str) -> Tuple[Optional[str], str, List[str]]:
    """
    name_part 정규화 후 exact 매칭 → 없으면 부분일치 후보(최대 10)
    반환: (adm10 or None, normalized_key, candidates[])
    """
    key = _norm_name(name_part)
    if key in _PNU_MAP:
        return _PNU_MAP[key], key, []
    candidates = [k for k in _PNU_MAP.keys() if key and key in k]
    def score(k: str):
        sc = 0
        if k.endswith("동"): sc += 1
        if key and k.startswith(key): sc += 1
        if " " in k: sc += 1
        return (-sc, len(k))
    candidates = sorted(candidates, key=score)[:10]
    if len(candidates) == 1:
        c = candidates[0]
        return _PNU_MAP[c], c, []
    return None, key, candidates

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

@app.post("/pnu/convert", response_model=ConvertResp)
async def pnu_convert_post(
    body: ConvertReq,
    x_api_key: Optional[str] = Header(None),
):
    require_api_key(x_api_key)
    _load_pnu10_once()
    name_part, bun, ji, mtYn = _parse_text_to_parts(body.text)
    if not name_part:
        return ConvertResp(ok=False, input=body.text, candidates=[])
    adm10, norm, cand = _find_adm10(name_part)
    if not adm10:
        return ConvertResp(ok=False, input=body.text, normalized=norm, candidates=cand)
    pnu = f"{adm10}{mtYn}{int(bun):04d}{int(ji):04d}"
    return ConvertResp(
        ok=True, input=body.text, normalized=norm, full=norm, admCd10=adm10,
        bun=f"{int(bun):04d}", ji=f"{int(ji):04d}", mtYn=mtYn, pnu=pnu, candidates=[]
    )

# (테스트/브라우저용) GET 변환
@app.get("/pnu/convert", response_model=ConvertResp)
async def pnu_convert_get(
    text: str = Query(..., description="예: '양재동 2-14'"),
    x_api_key: Optional[str] = Header(None),
):
    require_api_key(x_api_key)
    _load_pnu10_once()
    name_part, bun, ji, mtYn = _parse_text_to_parts(text)
    if not name_part:
        return ConvertResp(ok=False, input=text, candidates=[])
    adm10, norm, cand = _find_adm10(name_part)
    if not adm10:
        return ConvertResp(ok=False, input=text, normalized=norm, candidates=cand)
    pnu = f"{adm10}{mtYn}{int(bun):04d}{int(ji):04d}"
    return ConvertResp(
        ok=True, input=text, normalized=norm, full=norm, admCd10=adm10,
        bun=f"{int(bun):04d}", ji=f"{int(ji):04d}", mtYn=mtYn, pnu=pnu, candidates=[]
    )

# ── Building by PNU ─────────────────────────────────────────────────────────
class BuildingBundle(BaseModel):
    pnu: str
    pnuParts: Dict[str, Any]
    building: Dict[str, Any]
    lastUpdatedAt: str

async def hub_title_by_pnu(pnu: str, pageNo: int, numOfRows: int):
    if not PUBLICDATA_KEY:
        raise HTTPException(500, "PUBLICDATA_KEY is not set")
    params = {
        "serviceKey": PUBLICDATA_KEY,
        "_type": "json",
        "sigunguCd": pnu[:5],
        "bjdongCd": pnu[5:10],
        "platGbCd": pnu[10],
        "bun": pnu[11:15],
        "ji": pnu[15:19],
        "numOfRows": numOfRows,
        "pageNo": pageNo,
    }
    url = f"{BLD_BASE}/getBrTitleInfo"
    return await call_json(url, params)

@app.get("/realestate/building/by-pnu/{pnu}", response_model=BuildingBundle)
async def by_pnu(
    pnu: str,
    x_api_key: Optional[str] = Header(None),
    pageNo: int = Query(1, ge=1),
    numOfRows: int = Query(10, ge=1, le=100),
):
    require_api_key(x_api_key)
    if not re.fullmatch(r"\d{19}", pnu):
        raise HTTPException(400, "pnu must be 19 digits")

    resp = await hub_title_by_pnu(pnu, pageNo, numOfRows)

    item: Dict[str, Any] = {}
    try:
        items = resp.get("response", {}).get("body", {}).get("items", {})
        it = items.get("item")
        if isinstance(it, list) and it: item = it[0]
        elif isinstance(it, dict): item = it
    except Exception:
        pass

    return {
        "pnu": pnu,
        "pnuParts": {"admCd10": pnu[:10], "mtYn": pnu[10], "bun": pnu[11:15], "ji": pnu[15:19]},
        "building": item or resp,
        "lastUpdatedAt": now_iso(),
    }

# ── Health / Root ───────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    return {"ok": True, "time": now_iso(), "pnu10_loaded": _PNU_META}

@app.get("/")
async def root():
    return {"service": "RealEstate Toolkit", "version": "4.1.0"}
