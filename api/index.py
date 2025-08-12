import os, time, re
from typing import Optional, Dict, Any
import httpx
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime, timezone

app = FastAPI(title="RealEstate Aggregator", version="1.0.0")

PUBLICDATA_KEY = os.environ.get("PUBLICDATA_KEY")  # URL-encoded serviceKey
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY")  # our API protection key

def require_api_key(x_api_key: Optional[str]):
    if not SERVICE_API_KEY:
        # if not configured, allow all for local testing
        return
    if not x_api_key or x_api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ---------- PNU helpers ----------
def split_pnu(pnu: str) -> Dict[str, str]:
    if not re.fullmatch(r"\d{19}", pnu):
        raise HTTPException(400, "pnu must be 19 digits")
    return {
        "bjd_code": pnu[:10],
        "plat_gb": pnu[10],
        "bun": pnu[11:15],
        "ji": pnu[15:19],
    }

async def call_public(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if not PUBLICDATA_KEY:
        raise HTTPException(500, "PUBLICDATA_KEY is not set")
    params = {**params}
    params["serviceKey"] = PUBLICDATA_KEY
    async with httpx.AsyncClient(timeout=20) as client:
        for i in range(3):
            r = await client.get(url, params=params)
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    raise HTTPException(502, "Non-JSON response from upstream")
            await asyncio.sleep(0.3 * (i + 1))
    raise HTTPException(502, "Upstream timeout/error")

class RealEstateBundle(BaseModel):
    pnu: str
    summary: Dict[str, Any]
    lastUpdatedAt: str

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

@app.get("/healthz")
async def healthz():
    return {"ok": True, "time": now_iso()}

@app.get("/realestate/by-pnu/{pnu}", response_model=RealEstateBundle)
async def by_pnu(
    pnu: str,
    x_api_key: Optional[str] = Header(None),
    stdrYear: str = Query("2025")
):
    require_api_key(x_api_key)
    parts = split_pnu(pnu)

    # 1) 개별공시지가 (속성)
    # NOTE: If this endpoint changes, adjust here.
    price_resp = await call_public(
        "https://apis.data.go.kr/1611000/nsdi/IndvdLandPriceService/attr/getIndvdLandPriceAttr",
        {"pnu": pnu, "stdrYear": stdrYear, "format": "json", "numOfRows": 1, "pageNo": 1}
    )

    # 2) 토지대장/건축물대장 - TODO: map to actual endpoints and extract key fields
    land_resp = {"note": "TODO map to NSDI land register endpoint using sigunguCd/bjdongCd/bun/ji"}
    bld_resp = {"note": "TODO map to ArchBldgInfoService_v2 (표제부/전유부 등) with matching params"}

    # 3) extract price per_sqm (best-effort, depends on official schema)
    per_sqm = None
    try:
        # typical structure: response->body->items->item list
        items = price_resp.get("response", {}).get("body", {}).get("items", {})
        if isinstance(items, dict):
            # items may be {"item":[{...}]}
            item = None
            if "item" in items:
                if isinstance(items["item"], list) and items["item"]:
                    item = items["item"][0]
                elif isinstance(items["item"], dict):
                    item = items["item"]
            if item and "pblntfPclnd" in item:
                per_sqm = float(item["pblntfPclnd"])
    except Exception:
        per_sqm = None

    summary = {
        "land": land_resp,
        "building": bld_resp,
        "price": {
            "year": stdrYear,
            "per_sqm": per_sqm,
            "source": "개별공시지가"
        }
    }
    return {"pnu": pnu, "summary": summary, "lastUpdatedAt": now_iso()}

@app.get("/")
async def root():
    return {"service": "RealEstate Aggregator", "version": "1.0.0"}
