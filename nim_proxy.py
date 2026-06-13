"""
nim_proxy.py — transparent reverse proxy for NVIDIA NIM (or any OpenAI-compatible API).

Why: HF Spaces shares an egress IP range with thousands of other users.
NIM IP-throttles that range regardless of how clean the user's key is.
This proxy runs on Render (clean egress IP per service) and tunnels
NIM calls through, so the throttle doesn't apply.

Drop-in deployment:
  Render → New Web Service → Python
  Build:  pip install -r requirements.txt
  Start:  uvicorn nim_proxy:app --host 0.0.0.0 --port $PORT
  Env:    NIM_PROXY_API_KEY=<random-hex>     (optional, see auth below)
          NIM_UPSTREAM=https://integrate.api.nvidia.com/v1   (default)
"""

import os
import logging
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# --- Config ---
NVIDIA_BASE_URL = os.environ.get(
    "NIM_UPSTREAM", "https://integrate.api.nvidia.com"
).rstrip("/")
PROXY_API_KEY = os.environ.get("NIM_PROXY_API_KEY", "").strip()
TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=60.0, pool=10.0)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("nim-proxy")

# --- Hop-by-hop headers (per RFC 7230 §6.1) ---
HOP_BY_HOP = frozenset({
    "host", "content-length", "transfer-encoding", "connection",
    "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "upgrade",
})

# --- App lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f"nim-proxy up; upstream={NVIDIA_BASE_URL}  auth={'on' if PROXY_API_KEY else 'off'}")
    yield
    await client.aclose()

app = FastAPI(title="NIM Proxy", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

client = httpx.AsyncClient(timeout=TIMEOUT, http2=False, follow_redirects=False)


def authorized(request: Request) -> bool:
    """Two ways to be authorized:
       1. Caller supplied the proxy's X-API-Key (NIM_PROXY_API_KEY), OR
       2. Caller supplied a real NIM key in Authorization (Bearer nvapi-...)
          — proof they have quota, no need for the proxy key.

    If NIM_PROXY_API_KEY is unset, only option 2 is accepted.
    """
    if PROXY_API_KEY and request.headers.get("X-API-Key", "") == PROXY_API_KEY:
        return True
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return True
    return False


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "nim-proxy",
        "upstream": NVIDIA_BASE_URL,
        "auth": "required" if PROXY_API_KEY else "bearer-only",
    }


@app.get("/")
async def root():
    return {"status": "ok", "service": "nim-proxy"}


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy(path: str, request: Request):
    # CORS preflight — middleware handles it, return early
    if request.method == "OPTIONS":
        return JSONResponse({}, status_code=204)

    if not authorized(request):
        return JSONResponse(
            {"status": 401, "title": "Unauthorized",
             "detail": "Provide X-API-Key or a Bearer token"},
            status_code=401,
        )

    target = f"{NVIDIA_BASE_URL}/{path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"

    body = await request.body()

    fwd_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP
    }

    ua = request.headers.get("user-agent", "-")[:50]
    log.info(f"[NIM] {request.method} {path}  body={len(body)}B  ua={ua}")

    # Build + send upstream as a stream so SSE/chat-completions stream through
    upstream_req = client.build_request(
        method=request.method,
        url=target,
        headers=fwd_headers,
        content=body,
    )

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.RequestError as e:
        log.error(f"[NIM] upstream connect failed: {e}")
        return JSONResponse(
            {"status": 502, "title": "Bad Gateway",
             "detail": f"upstream error: {e}"},
            status_code=502,
        )

    async def relay():
        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
        finally:
            await upstream_resp.aclose()

    # Forward response headers (drop hop-by-hop and content-type — media_type handles that)
    resp_headers = {
        k: v for k, v in upstream_resp.headers.items()
        if k.lower() not in (HOP_BY_HOP | {"content-type"})
    }

    return StreamingResponse(
        relay(),
        status_code=upstream_resp.status_code,
        headers=resp_headers,
        media_type=upstream_resp.headers.get("content-type", "application/json"),
  )
