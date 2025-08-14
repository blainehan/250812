# api/index.py
from __future__ import annotations

import csv
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel
from urllib.parse import unquote_plus

APP_VERSION = "4.1.2"

app = FastAPI(
    title="RealEstate Toolkit",
    description="PNU 변환 + 건축물대장(표제부) 조회",
    version=APP_VERSION,
)

# -----------------------------
# 환경 변수
# -----------------------------
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")
PUBLICDATA_KEY = os.getenv("PUBLICDATA_KEY", "")  # 공공데이터포털 일반 인증키(Decoding)

# -----------------------------
# 공통 유틸 / 보안
# -----------------------------
def require_api_key(x_api_key: Optional[str]) -> None:
    if not SERVICE_API_KEY:
        # 로컬 테스트 편의를 위해 키 미설정 시 통과 (원하면 막으세요)
        return
    if not x_api_key or x_api_key.strip() != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# -----------------------------
# pnu10.csv 로딩
# -----------------------------
_PNU10_MAP: Optional[Dict[str, str]] = None  # full name -> 10자리

def _normalize_name(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u3000", " ").strip()
    s = " ".join(s.split())
    # 광역시/특별시 축약 보정
    repl = {
        "서울시": "서울특별시", "서울": "서울특별시",
        "부산시": "부산광역시", "부산": "부산광역시",
        "대구시": "대구광역시", "대구": "대구광역시",
        "인천시": "인천광역시", "인천": "인천광역시",
        "광주시": "광주광역시", "광주": "광주광역시",
        "대전시": "대전광역시", "대전": "대전광역시",
        "울산시": "울산광역시", "울산": "울산광역시",
        "세종시": "세종특별자치시", "세종": "세종특별자치시",
    }
    for k, v in repl.items():
        if s.startswith(k + " "):
            s = v + s[len(k):]
            break
    return s

def _load_pnu10_once() -> None:
    global _PNU10_MAP
    if _PNU10_MAP is not None:
        return

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "pnu10.csv")
    data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        # vercel 함수 경로 내로 복사했을 때 대비
        alt = os.path.join("/var/task/data", "pnu10.csv")
        if os.path.exists(alt):
            data_path = alt

    mapping: Dict[str, str] = {}
    with open(data_path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "\t" in line:
                code10, full = line.split("\t", 1)
            elif "," in line:
                code10, full = line.split(",", 1)
            else:
                parts = line.split()
                if len(parts) >= 2:
                    code10, full = parts[0], " ".join(parts[1:])
                else:
                    continue
            code10 = code10.strip()
            full = _normalize_name(full)
            if re.fullmatch(r"\d{10}", code10):
                mapping[full] = code10

    _PNU10_MAP = mapping

def _find_adm10(name_input: str) -> Tuple[str, str, List[str]]:
    """이름으로 10자리 행정코드 찾기. (정확/부분 후보)"""
    key = _normalize_name(name_input)
    if key in _PNU10_MAP:
        return _PNU10_MAP[key], key, []
    # 부분 일치 후보
    cands = [k for k in _PNU10_MAP.keys() if key and (key in k or k in key)]
    if not cands:
        return "", key, []
    chosen = cands[0]
    return _PNU10_MAP[chosen], chosen, cands[:10]

# -----------------------------
# 입력 복구 (GET 한글/모지바케)
# -----------------------------
def _fix_input_text(raw: str) -> str:
    """
    - URL 인코딩(+ 포함) 복구
    - 이중 인코딩 흔적 복구 시도
    - 모지바케(�) 감지 시 cp949 재해석 시도
    """
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

# -----------------------------
# 텍스트 → (이름, 본번, 부번, 산여부)
# -----------------------------
_BUNJI_RE = re.compile(r"(\d+)(?:\s*[-~]\s*(\d+))?")

def _parse_text_to_parts(text: str) -> Tuple[str, int, int, str]:
    """
    '서울 서초구 양재동 2-14', '양재동 산 2-14', '양재동 2' 등에서
    (이름, 본번, 부번, mtYn) 을 추출
    """
    s = _normalize_name(text)
    mtYn = "1" if (" 산" in s or s.startswith("산")) else "0"

    # 숫자 패턴 찾기
    m = _BUNJI_RE.search(s)
    bun, ji = 0, 0
    if m:
        bun = int(m.group(1) or 0)
        ji = int(m.group(2) or 0)
        name_part = s[: m.start()].strip()
    else:
        name_part = s.strip()

    # 괄호 등 제거
    name_part = re.sub(r"\s*\(.*?\)\s*$", "", name_part).strip()
    return name_part, bun, ji, mtYn

# -----------------------------
# 응답 모델
# -----------------------------
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

# -----------------------------
# 라우트: 루트/헬스체크
# -----------------------------
@app.get("/")
async def root():
    return {"service": "RealEstate Toolkit", "version": APP_VERSION}

@app.get("/healthz")
async def healthz():
    try:
        _load_pnu10_once()
        entries = len(_PNU10_MAP or {})
    except Exception:
        entries = 0
    return {
        "ok": True,
        "time": utc_now_iso(),
        "pnu10_loaded": {"entries": entries},
    }

# -----------------------------
# 라우트: PNU 변환 (POST)
# -----------------------------
@app.post("/pnu/convert", response_model=ConvertResp)
async def pnu_convert_post(
    body: ConvertReq,
    x_api_key: Optional[str] = Header(None),
):
    require_api_key(x_api_key)
    _load_pnu10_once()

    fixed = _fix_input_text(body.text)

    name_part, bun, ji, mtYn = _parse_text_to_parts(fixed)
    if not name_part:
        return ConvertResp(ok=False, input=fixed, candidates=[])

    adm10, norm, cand = _find_adm10(name_part)
    if not adm10:
        return ConvertResp(ok=False, input=fixed, normalized=norm, candidates=cand)

    pnu = f"{adm10}{mtYn}{int(bun):04d}{int(ji):04d}"
    return ConvertResp(
        ok=True,
        input=fixed,
        normalized=norm,
        full=norm,
        admCd10=adm10,
        bun=f"{int(bun):04d}",
        ji=f"{int(ji):04d}",
        mtYn=mtYn,
        pnu=pnu,
        candidates=[],
    )

# -----------------------------
# 라우트: PNU 변환 (GET, 브라우저/테스트용)
# -----------------------------
@app.get("/pnu/convert", response_model=ConvertResp)
async def pnu_convert_get(
    text: str = Query(..., description="예: '양재동 2-14'"),
    x_api_key: Optional[str] = Header(None),
):
    require_api_key(x_api_key)
    _load_pnu10_once()

    fixed = _fix_input_text(text)

    name_part, bun, ji, mtYn = _parse_text_to_parts(fixed)
    if not name_part:
        return ConvertResp(ok=False, input=fixed, candidates=[])

    adm10, norm, cand = _find_adm10(name_part)
    if not adm10:
        return ConvertResp(ok=False, input=fixed, normalized=norm, candidates=cand)

    pnu = f"{adm10}{mtYn}{int(bun):04d}{int(ji):04d}"
    return ConvertResp(
        ok=True,
        input=fixed,
        normalized=norm,
        full=norm,
        admCd10=adm10,
        bun=f"{int(bun):04d}",
        ji=f"{int(ji):04d}",
        mtYn=mtYn,
        pnu=pnu,
        candidates=[],
    )

# -----------------------------
# 라우트: 건축물대장 표제부 by PNU
# -----------------------------
class BuildingBundle(BaseModel):
    pnu: str
    building: Optional[dict] = None
    lastUpdatedAt: str

def _split_pnu(pnu: str) -> Tuple[str, str, str, str]:
    if not re.fullmatch(r"\d{19}", pnu):
        raise HTTPException(status_code=400, detail="pnu must be 19 digits")
    adm10 = pnu[:10]
    mtYn = pnu[10]
    bun = pnu[11:15]
    ji = pnu[15:19]
    return adm10, mtYn, bun, ji

@app.get("/realestate/building/by-pnu/{pnu}", response_model=BuildingBundle)
async def building_by_pnu(
    pnu: str,
    x_api_key: Optional[str] = Header(None),
):
    require_api_key(x_api_key)

    adm10, mtYn, bun, ji = _split_pnu(pnu)
    sigunguCd = adm10[:5]
    bjdongCd = adm10[5:]
    platGbCd = "1" if mtYn == "1" else "0"

    # 공공데이터포털: 건축물대장 표제부 (JSON)
    if not PUBLICDATA_KEY:
        raise HTTPException(status_code=500, detail="PUBLICDATA_KEY not set")

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
        except Exception as e:
            # 일부 지역/키 설정에 따라 XML이 나올 수 있음
            raise HTTPException(status_code=502, detail="Non-JSON response from upstream") from e

    # 응답 해석 (공공데이터 v2 공통 구조)
    # data['response']['body']['items']['item'] 형태가 일반적
    building: Optional[dict] = None
    try:
        resp = data.get("response", {})
        body = (resp or {}).get("body", {})
        items = (body or {}).get("items", {})
        item = (items or {}).get("item")
        # item이 리스트/단일 객체 모두 대응
        if isinstance(item, list):
            building = item[0] if item else None
        elif isinstance(item, dict):
            building = item
    except Exception:
        building = None

    return BuildingBundle(
        pnu=pnu,
        building=building,
        lastUpdatedAt=utc_now_iso(),
    )
