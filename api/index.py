from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel

# ================== 앱/환경 ==================
APP_VERSION = "9.2.0-mois-only-onekey"  # CSV 제거, MOIS 스펙 100% 반영, 공공데이터키 1개 사용
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY", "")

# 공공데이터포털 서비스키(Decoding Key 권장) - 행안부/건축물대장 공용
PUBLICDATA_KEY  = os.getenv("PUBLICDATA_KEY", "")
MOIS_SERVICE_KEY = PUBLICDATA_KEY  # 같은 키 사용

# MOIS(행안부) 법정표준코드 API
MOIS_BASE_URL  = os.getenv("MOIS_BASE_URL", "https://apis.data.go.kr/1741000/StanReginCd")
MOIS_LIST_PATH = os.getenv("MOIS_LIST_PATH", "getStanReginCdList")  # 법정동코드 조회

# ================== 공통 유틸 ==================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def require_api_key(x_api_key: Optional[str]):
    """
    SERVICE_API_KEY가 설정되어 있으면 X-API-Key를 검사.
    미설정이면 인증 생략(개발/공개 테스트용).
    """
    if not SERVICE_API_KEY:
        return
    if not x_api_key or x_api_key.strip() != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _fix_input_text(raw: str) -> str:
    """URL 인코딩/깨짐 방어."""
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
    # U+FFFD(�) 포함 시 간단 복구 시도
    if "\ufffd" in text:
        try:
            text = text.encode("latin-1", "ignore").decode("cp949")
        except Exception:
            pass
    return _norm_spaces(text)

# ================== 주소 정규화/번지 파싱 ==================
_DASHES = "－–—"  # 다양한 대시 기호

def normalize_address(s: str) -> str:
    s = (s or "")
    s = s.replace("서울시", "서울특별시").replace("서울 특별시", "서울특별시")
    s = s.replace("광역 시", "광역시")
    for ch in _DASHES:
        s = s.replace(ch, "-")
    return _norm_spaces(s)

_BUN_JI_RE = re.compile(
    r"""
    (?:^|[\s,()])          # 경계
    (?:산\s*)?             # 산 표기 선택
    (?P<bun>\d{1,6})       # 본번
    (?:\s*-\s*(?P<ji>\d{1,6}))?   # 부번
    (?!\d)
    """,
    re.VERBOSE,
)

def parse_bunjib(addr: str) -> Tuple[int, Optional[int], Optional[int]]:
    # '산'은 숫자 바로 앞에 올 때만 산지 처리(오탐 방지)
    mt = 1 if re.search(r"(?<![가-힣A-Za-z])산\s*\d", addr) else 0
    matches = list(_BUN_JI_RE.finditer(addr or ""))
    if not matches:
        return mt, None, None
    m = matches[-1]  # 마지막 표기 채택
    bun = int(m.group("bun"))
    ji = int(m.group("ji")) if m.group("ji") else 0
    return mt, bun, ji

def _strip_bunjib(addr: str) -> str:
    return _BUN_JI_RE.sub(" ", addr or "")

# ================== 행정구역 정규화 ==================
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
    "경기": "경기도",
    "강원": "강원특별자치도", "강원도": "강원특별자치도",
    "충북": "충청북도", "충남": "충청남도",
    "전북": "전북특별자치도", "전라북도": "전북특별자치도",
    "전남": "전라남도",
    "경북": "경상북도", "경남": "경상남도",
}
def _canonical_si(token: str) -> str:
    t = (token or "").strip()
    return _SI_SYNONYMS.get(t, t)

def _split_parts(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    parts = _norm_spaces(name).split(" ")
    if not parts:
        return None, None, None
    if len(parts) == 1:
        return parts[0], None, None           # emd만
    if len(parts) == 2:
        return parts[0], None, parts[1]       # si/emd
    return parts[0], parts[1], parts[-1]      # si/sigu/emd

# ================== MOIS(행안부) API 클라이언트 — 문서 스펙 100% 일치 ==================
class MOISClient:
    """
    행정표준코드(법정동코드) API 조회 (StanReginCd/getStanReginCdList).
    - 요청 파라미터: ServiceKey, type=json, flag=Y, pageNo, numOfRows, locatadd_nm
    - 응답 필드: region_cd(10자리 법정동코드), locatadd_nm(지역주소명), locat_order(서열)
    스펙 근거: 행안부 Open API 활용가이드:contentReference[oaicite:1]{index=1}
    """
    def __init__(self, base_url: str, list_path: str, service_key: str):
        self.base_url = base_url.rstrip("/")
        self.list_path = list_path.strip("/")
        self.key = service_key

    def _endpoint(self) -> str:
        # 예: https://apis.data.go.kr/1741000/StanReginCd/getStanReginCdList
        return f"{self.base_url}/{self.list_path}"

    @staticmethod
    def _extract_items(payload: dict) -> List[dict]:
        """
        response -> body -> items -> item (list or single) 구조를 안전 파싱:contentReference[oaicite:2]{index=2}
        """
        try:
            resp = payload.get("response") or {}
            body = resp.get("body") or {}
            items = (body.get("items") or {}).get("item")
            if isinstance(items, list):
                return items
            if isinstance(items, dict):
                return [items]
            return []
        except Exception:
            return []

    @staticmethod
    def _result_header(payload: dict) -> Dict[str, Any]:
        """
        결과코드/메시지 획득 (일부 구현의 header/RESULT 혼재를 방어):contentReference[oaicite:3]{index=3}
        """
        resp = payload.get("response") or {}
        header = resp.get("header") or {}
        if not header and "RESULT" in (resp.get("head") or {}):
            header = (resp.get("head") or {}).get("RESULT", {})
        return {
            "resultCode": header.get("resultCode"),
            "resultMsg": header.get("resultMsg"),
        }

    @staticmethod
    def _adm10_from_item(it: dict) -> Optional[str]:
        """
        region_cd: 10자리 법정동코드 (문서 스펙):contentReference[oaicite:4]{index=4}
        """
        adm = str(it.get("region_cd") or "").strip()
        return adm if (len(adm) == 10 and adm.isdigit()) else None

    @staticmethod
    def _is_legal_dong_level(it: dict) -> bool:
        """
        locat_order(서열)로 3단계(시/도-시군구-법정동)를 우선 채택:contentReference[oaicite:5]{index=5}.
        """
        return str(it.get("locat_order") or "").strip() == "3"

    @staticmethod
    def _name_of(it: dict) -> str:
        """
        지역주소명: locatadd_nm (문서 스펙):contentReference[oaicite:6]{index=6}
        """
        return _norm_spaces(str(it.get("locatadd_nm") or it.get("locallow_nm") or ""))

    async def _query(self, params: Dict[str, Any]) -> List[dict]:
        """
        요청 파라미터를 문서 명세와 동일하게 사용:
        - ServiceKey(필수, URL-Encode 권장), type(json/xml), flag(Y), pageNo, numOfRows, locatadd_nm:contentReference[oaicite:7]{index=7}
        """
        if not self.key:
            raise HTTPException(status_code=500, detail="PUBLICDATA_KEY not set")

        q = {
            "ServiceKey": self.key,   # 문서 표기 그대로(대소문자)
            "type": "json",           # JSON 응답
            "flag": "Y",              # 문서 예시의 신규API 플래그
            "pageNo": params.pop("pageNo", 1),
            "numOfRows": params.pop("numOfRows", 50),
        }
        q.update(params)  # locatadd_nm 등

        url = self._endpoint()
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=q)
            r.raise_for_status()
            data = r.json()

        # 결과코드 확인(정상: 00 또는 INFO-0)
        hdr = self._result_header(data)
        rc = (hdr.get("resultCode") or "").upper()
        if rc and rc not in ("00", "INFO-0"):
            raise HTTPException(status_code=502, detail=f"MOIS API error: {hdr}")

        return self._extract_items(data)

    async def find_adm10(self, name: str) -> Dict[str, Any]:
        """
        입력 명칭(name)에서 10자리 법정동코드(region_cd)를 유추.
        - 완전일치(locatadd_nm) → (시군구,법정동)/(시,법정동) 조합 → tail 2토큰 → 단일 토큰은 안내
        """
        name = _norm_spaces(name)
        if not name:
            return {"ok": False, "error": "질의가 비어 있습니다.", "query": name}

        # 맨 앞 토큰 시·도 정규화
        parts = name.split(" ")
        if parts:
            parts[0] = _canonical_si(parts[0])
        name_norm = " ".join(parts)

        # 1) 완전 일치
        items = await self._query({"locatadd_nm": name_norm})
        if items:
            exact = [it for it in items if self._name_of(it) == name_norm]
            # 서열 3단계 우선
            exact_lvl3 = [it for it in exact if self._is_legal_dong_level(it)]
            pool = exact_lvl3 or exact or items
            pool = [it for it in pool if self._adm10_from_item(it)]
            uniq = {(self._name_of(it), self._adm10_from_item(it)) for it in pool}
            if len(uniq) == 1:
                full, adm = next(iter(uniq))
                return {"ok": True, "admCd10": adm, "matched": full}
            elif len(uniq) > 1:
                return {
                    "ok": False,
                    "error": "여러 지역에서 일치합니다. 시군구를 포함해 주세요.",
                    "query": name,
                    "candidates": [f for f, _ in sorted(uniq)],
                }

        # 2) 시군구+법정동 / 시+법정동 조합
        si, sigu, emd = _split_parts(name_norm)
        combos: List[str] = []
        if sigu and emd:
            combos.append(f"{_canonical_si(si) if si else ''} {sigu} {emd}".strip())
        if si and emd and not sigu:
            combos.append(f"{_canonical_si(si)} {emd}")

        for cand in combos:
            items = await self._query({"locatadd_nm": cand})
            if not items:
                continue
            lvl3 = [it for it in items if self._is_legal_dong_level(it)]
            pool = lvl3 or items
            pool = [it for it in pool if self._adm10_from_item(it)]
            uniq = {(self._name_of(it), self._adm10_from_item(it)) for it in pool}
            if len(uniq) == 1:
                full, adm = next(iter(uniq))
                return {"ok": True, "admCd10": adm, "matched": full}
            elif len(uniq) > 1:
                return {
                    "ok": False,
                    "error": "여러 지역에서 일치합니다. 시도/시군구를 포함해 주세요.",
                    "query": name,
                    "candidates": [f for f, _ in sorted(uniq)],
                }

        # 3) 꼬리 2토큰 (예: "... 서초구 양재동")
        tokens = name_norm.split(" ")
        if len(tokens) >= 2:
            tail2 = " ".join(tokens[-2:])
            items = await self._query({"locatadd_nm": tail2})
            if items:
                lvl3 = [it for it in items if self._is_legal_dong_level(it)]
                pool = lvl3 or items
                pool = [it for it in pool if self._adm10_from_item(it)]
                uniq = {(self._name_of(it), self._adm10_from_item(it)) for it in pool}
                if len(uniq) == 1:
                    full, adm = next(iter(uniq))
                    return {"ok": True, "admCd10": adm, "matched": full}
                elif len(uniq) > 1:
                    return {
                        "ok": False,
                        "error": "여러 지역에서 일치합니다. 시군구를 포함해 주세요.",
                        "query": name,
                        "candidates": [f for f, _ in sorted(uniq)],
                    }

        # 4) 단일 토큰(emd-only)은 모호 가능 → 안내
        if len(tokens) == 1:
            return {
                "ok": False,
                "error": "여러 지역에서 일치할 수 있습니다. '시군구 법정동' 형식으로 입력해 주세요.",
                "query": name,
            }

        return {"ok": False, "error": "법정동을 찾지 못했습니다.", "query": name}

# 전역 MOIS 클라이언트
_mois = MOISClient(MOIS_BASE_URL, MOIS_LIST_PATH, MOIS_SERVICE_KEY)

# ================== 스키마 ==================
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
    pnu: Optional[str] = None            # 19자리 PNU
    source: Optional[str] = None         # "mois"
    candidates: Optional[List[str]] = None
    version: Optional[str] = None

class BuildingBundle(BaseModel):
    pnu: str
    building: Optional[dict] = None
    lastUpdatedAt: str

# ================== FastAPI 앱 ==================
app = FastAPI(
    title="RealEstate Toolkit (MOIS API only, one key)",
    description="행안부 법정표준코드 API 기반 주소→PNU, 국토부 API로 PNU→건축물대장(표제부) — 공공데이터 서비스키 하나로 통합",
    version=APP_VERSION,
)

@app.get("/")
async def root():
    return {"service": "RealEstate Toolkit (MOIS, one key)", "version": APP_VERSION}

@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "time": utc_now_iso(),
        "mois": {
            "base": MOIS_BASE_URL,
            "path": MOIS_LIST_PATH,
            "has_key": bool(MOIS_SERVICE_KEY),
        },
        "publicdata": bool(PUBLICDATA_KEY),
        "version": APP_VERSION,
    }

@app.get("/version")
async def version():
    return {"version": APP_VERSION, "who": "fastapi-index", "time": utc_now_iso()}

@app.get("/_healthz")
async def _healthz():
    return await healthz()

# ---- 공통: 19자리 PNU 조립 ----
def build_pnu19(code10: str, mt: int, bun: int, ji: int) -> str:
    return f"{code10}{mt}{bun:04d}{ji:04d}"

# -------- 주소→PNU 변환 (MOIS만 사용) --------
def _convert_base(addr: str, res10: Dict[str, Any], mt: Optional[int], bun: Optional[int], ji: Optional[int]) -> ConvertResp:
    base = {
        "ok": False,
        "input": addr,
        "normalized": normalize_address(addr),
        "full": res10.get("matched"),
        "admCd10": res10.get("admCd10"),
        "bun": f"{bun:04d}" if isinstance(bun, int) else None,
        "ji": f"{ji:04d}" if isinstance(ji, int) else None,
        "mtYn": (str(mt) if isinstance(mt, int) else None),
        "pnu": None,
        "source": "mois",
        "candidates": res10.get("candidates"),
        "version": APP_VERSION,
    }
    if not res10.get("ok"):
        return ConvertResp(**base)

    if bun is None:
        base["ok"] = True
        return ConvertResp(**base)

    pnu19 = build_pnu19(res10["admCd10"], mt or 0, bun, (ji if ji is not None else 0))
    base.update({"ok": True, "pnu": pnu19})
    return ConvertResp(**base)

async def _convert_impl_async(address: str) -> ConvertResp:
    addr = normalize_address(address)
    mt, bun, ji = parse_bunjib(addr)
    name_part = _strip_bunjib(addr)
    name_part = _norm_spaces(name_part)
    res10 = await _mois.find_adm10(name_part if name_part else addr)
    return _convert_base(address, res10, mt, bun, ji)

@app.post("/pnu/convert", response_model=ConvertResp)
async def pnu_convert_post(body: ConvertReq, x_api_key: Optional[str] = Header(None)):
    require_api_key(x_api_key)
    if not PUBLICDATA_KEY:
        raise HTTPException(status_code=500, detail="PUBLICDATA_KEY not set")
    fixed = _fix_input_text(body.text)
    return await _convert_impl_async(fixed)

@app.get("/pnu/convert", response_model=ConvertResp)
async def pnu_convert_get(
    text: str = Query(..., description="예: '서울특별시 서초구 양재동 2-14'"),
    x_api_key: Optional[str] = Header(None),
):
    require_api_key(x_api_key)
    if not PUBLICDATA_KEY:
        raise HTTPException(status_code=500, detail="PUBLICDATA_KEY not set")
    fixed = _fix_input_text(text)
    return await _convert_impl_async(fixed)

# -------- PNU→건축물대장(표제부) --------
def _split_pnu(pnu: str) -> Tuple[str, str, str, str]:
    if not re.fullmatch(r"\d{19}", pnu or ""):
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
        body = (data.get("response") or {}).get("body") or {}
        items = (body.get("items") or {})
        item = items.get("item")
        if isinstance(item, list):
            building = item[0] if item else None
        elif isinstance(item, dict):
            building = item
    except Exception:
        building = None

    return BuildingBundle(pnu=pnu, building=building, lastUpdatedAt=utc_now_iso())
