# api/index.py
import os, re, csv
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

from pydantic import BaseModel

class FullnameReq(BaseModel):
    full: str          # 예: "서울특별시 종로구 청운동"
    bunji: str         # 예: "24-5" (부번 없으면 "24")
    mtYn: str = "0"    # "0"=대지, "1"=산
    pageNo: int = 1
    numOfRows: int = 10

def _resolve_adm10_from_full(full_input: str) -> str:
    key = _normalize_fullname(full_input)
    return _pnu10_map.get(key, "")

app = FastAPI(title="RealEstate (fullname+bunji→PNU, no JUSO)", version="2.1.0")

# ────────────────────────────────────────────────────────────────────────────
# 환경변수
#  - PUBLICDATA_KEY : data.go.kr 건축HUB 서비스키 (Encoding 키 권장)
#  - SERVICE_API_KEY: 우리 API 보호용 키 (있으면 X-API-Key 필수)
# ────────────────────────────────────────────────────────────────────────────
PUBLICDATA_KEY = os.environ.get("PUBLICDATA_KEY")
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")

BLD_BASE = "https://apis.data.go.kr/1613000/BldRgstHubService"

# ────────────────────────────────────────────────────────────────────────────
# 동 풀네임 → 앞10자리(admCd10) 매핑 로더
#  - CSV/TSV/공백 구분 자동 인식
#  - 컬럼 순서 자동 인식: (코드,이름) 또는 (이름,코드)
#  - 여러 경로 후보 검색
# ────────────────────────────────────────────────────────────────────────────
PNU10_CANDIDATES = [
    Path(__file__).parent.parent / "data" / "pnu10.csv",
    Path(__file__).parent.parent / "data" / "pnu10.tsv",
    Path(__file__).parent / "data" / "pnu10.csv",
    Path(__file__).parent / "data" / "pnu10.tsv",
    Path.cwd() / "data" / "pnu10.csv",
    Path.cwd() / "data" / "pnu10.tsv",
]

_pnu10_map: Dict[str, str] = {}
_pnu10_meta: Dict[str, Any] = {"path": None, "entries": 0, "delimiter": None, "reversed_cols": False}

def _normalize_fullname(s: str) -> str:
    """입력 풀네임 정규화: 공백/축약 보정"""
    s = (s or "").replace("\u3000", " ").strip()
    s = " ".join(s.split())  # 중복 공백 제거
    # 흔한 축약 보정
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
    """pnu10 파일을 찾아 로드하여 _pnu10_map에 {풀네임: 코드10} 저장"""
    global _pnu10_map, _pnu10_meta
    found = None
    for p in PNU10_CANDIDATES:
        if p.exists():
            found = p
            break
    if not found:
        raise RuntimeError(f"pnu10.csv/tsv not found in any of: {[str(p) for p in PNU10_CANDIDATES]}")

    _pnu10_meta["path"] = str(found)

    # 파일 내용 일부 샘플로 구분자 추정
    with open(found, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        if "\t" in sample and (sample.count("\t") >= sample.count(",")):
            delimiter = "\t"
        elif "," in sample:
            delimiter = ","
        else:
            delimiter = None  # 공백 분리 모드

        rows = []
        if delimiter:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
            _pnu10_meta["delimiter"] = "\\t" if delimiter == "\t" else ","
        else:
            rows = [line.strip().split() for line in f if line.strip()]
            _pnu10_meta["delimiter"] = "whitespace"

    reversed_cols = False
    for row in rows:
        if not row:
            continue

        # 공백 모드면 이름에 공백이 있을 수 있으니 합치기
        if _pnu10_meta["delimiter"] == "whitespace" and len(row) >= 2:
            left = row[0]
            right = " ".join(row[1:])
        elif len(row) >= 2:
            left, right = row[0], row[1]
        else:
            continue

        left = (left or "").strip().strip('"').strip("'")
        right = (right or "").strip().strip('"').strip("'")

        # 패턴 1: left=코드(10자리), right=풀네임
        if re.fullmatch(r"\d{10}", left):
            code10 = left
            full = _normalize_fullname(right)
        # 패턴 2: left=풀네임, right=코드(10자리)
        elif re.fullmatch(r"\d{10}", right):
            code10 = right
            full = _normalize_fullname(left)
            reversed_cols = True
        else:
            # 다른 형식은 스킵
            continue

        if full and re.fullmatch(r"\d{10}", code10):
            _pnu10_map[full] = code10

    _pnu10_meta["entries"] = len(_pnu10_map)
    _pnu10_meta["reversed_cols"] = reversed_cols
    if _pnu10_meta["entries"] == 0:
        raise RuntimeError(f"Loaded 0 entries from {found}. Check delimiter/column order/encoding.")

# 서버 시작 시 로드
_load_pnu10()

# ────────────────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────────────────
def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def require_api_key(x_api_key: Optional[str]):
    if SERVICE_API_KEY and x_api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def compose_pnu(adm10: str, mtYn: str, bun_raw: str, ji_raw: str) -> str:
    if not re.fullmatch(r"\d{10}", adm10):
        raise HTTPException(400, "admCd10 must be 10 digits")
    if mtYn not in ("0", "1"):
        raise HTTPException(400, "mtYn must be '0' or '1'")
    bun = re.sub(r"\D", "", bun_raw or "0")
    ji  = re.sub(r"\D", "", ji_raw or "0")
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

# ────────────────────────────────────────────────────────────────────────────
# 응답 스키마
# ────────────────────────────────────────────────────────────────────────────
class BuildingBundle(BaseModel):
    pnu: str
    pnuParts: Dict[str, Any]
    building: Dict[str, Any]
    lastUpdatedAt: str

# ────────────────────────────────────────────────────────────────────────────
# 라우트
# ────────────────────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    return {"ok": True, "time": now_iso()}

@app.get("/")
async def root():
    return {"service": "RealEstate (fullname+bunji→PNU, no JUSO)", "version": "2.1.0"}

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
        # 입력이 풀네임과 완전히 같지 않으면 부분일치 후보 제공
        cand = [k for k in _pnu10_map if key in k or k in key][:5]
        raise HTTPException(404, detail={"message": f"No admCd10 for '{key}'", "candidates": cand})

    # bunji 파싱: "본-부" → (본,부), "본" → (본,"0")
    bun_raw, ji_raw = (bunji.split("-", 1) if "-" in bunji else (bunji, "0"))
    pnu = compose_pnu(adm10, mtYn, bun_raw, ji_raw)

    resp = await hub_title_by_pnu(pnu, pageNo, numOfRows)

    # 첫 레코드만 요약
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

    return {
        "pnu": pnu,
        "pnuParts": {"admCd10": adm10, "mtYn": mtYn, "bun": pnu[11:15], "ji": pnu[15:19]},
        "building": item or resp,
        "lastUpdatedAt": now_iso(),
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
        if isinstance(it, list) and it:
            item = it[0]
        elif isinstance(it, dict):
            item = it
    except Exception:
        pass
    return {
        "pnu": pnu,
        "pnuParts": {"admCd10": pnu[:10], "mtYn": pnu[10], "bun": pnu[11:15], "ji": pnu[15:19]},
        "building": item or resp,
        "lastUpdatedAt": now_iso(),
    }

# ────────────────────────────────────────────────────────────────────────────
# 디버그 엔드포인트 (파일 로딩/매칭 확인용)
# ────────────────────────────────────────────────────────────────────────────
@app.get("/debug/pnu10/stats")
async def debug_pnu10_stats():
    return {
        "path": _pnu10_meta.get("path"),
        "entries": _pnu10_meta.get("entries"),
        "delimiter": _pnu10_meta.get("delimiter"),
        "reversed_cols": _pnu10_meta.get("reversed_cols"),
        "sample_5": list(_pnu10_map.items())[:5],
    }

@app.get("/debug/pnu10/lookup")
async def debug_pnu10_lookup(full: str):
    key = _normalize_fullname(full)
    exact = _pnu10_map.get(key)
    partial = [k for k in _pnu10_map if key in k or k in key][:10]
    return {"input": full, "normalized": key, "exact": exact, "partial": partial}

@app.post("/realestate/building/by-fullname", response_model=BuildingBundle)
async def by_fullname_post(
    body: FullnameReq,
    x_api_key: Optional[str] = Header(None),
):
    """Actions/GPT 용: JSON 바디로 한글 그대로 받기 (URL 인코딩 문제 없음)"""
    require_api_key(x_api_key)

    adm10 = _resolve_adm10_from_full(body.full)
    if not adm10:
        key = _normalize_fullname(body.full)
        cand = [k for k in _pnu10_map if key in k or k in key][:5]
        raise HTTPException(404, detail={"message": f"No admCd10 for '{key}'", "candidates": cand})

    bun_raw, ji_raw = (body.bunji.split("-", 1) if "-" in body.bunji else (body.bunji, "0"))
    pnu = compose_pnu(adm10, body.mtYn, bun_raw, ji_raw)

    resp = await hub_title_by_pnu(pnu, body.pageNo, body.numOfRows)

    # 첫 레코드 하나 요약 추출
    item = {}
    try:
        items = resp.get("response", {}).get("body", {}).get("items", {})
        it = items.get("item")
        if isinstance(it, list) and it: item = it[0]
        elif isinstance(it, dict): item = it
    except Exception:
        pass

    return {
        "pnu": pnu,
        "pnuParts": {"admCd10": adm10, "mtYn": body.mtYn, "bun": pnu[11:15], "ji": pnu[15:19]},
        "building": item or resp,
        "lastUpdatedAt": now_iso(),
    }
