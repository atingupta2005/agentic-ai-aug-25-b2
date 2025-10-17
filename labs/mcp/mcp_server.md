# 1 — `mcp_server.py` (single-file MCP Server)

Save this as `mcp_server.py`.

```python
"""
mcp_server.py
A reusable, multi-user MCP Server (FastAPI).
Features:
 - /tools             GET : list registered tools
 - /tools/register    POST: register a new tool (runtime)
 - /tool/{name}       GET/POST: execute tool (GET uses query, POST uses JSON)
 - /context/{prov}    POST: fetch context from a provider (payload contains user/project ids)
 - /providers         GET : list providers
 - /providers/register POST: register provider dynamically
 - /health            GET : simple health check
 - Optional API key auth (MCP_API_KEY env var or header)
 - Pluggable LLM hook for ai_summarizer (via OPENAI_API_KEY)
"""

import os
import asyncio
import inspect
from typing import Any, Callable, Dict, Optional
from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import logging

# Optional: If you want to call OpenAI, install `openai` and set OPENAI_API_KEY env var
# import openai

logger = logging.getLogger("uvicorn")
logging.basicConfig(level=logging.INFO)

APP_TITLE = "MCP Server"
APP_VERSION = "0.1.0"

app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# Allow CORS by default for convenience (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET","POST","PUT","DELETE","OPTIONS"],
    allow_headers=["*"],
)

# ---------- Simple API key auth dependency ----------
MCP_API_KEY = os.environ.get("MCP_API_KEY")  # set for production if needed

async def require_api_key(x_api_key: Optional[str] = Header(None)):
    if MCP_API_KEY:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Missing X-API-Key header")
        if x_api_key != MCP_API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")
    return True


# ---------- Registries (in-memory) ----------
TOOLS: Dict[str, Dict[str, Any]] = {}
PROVIDERS: Dict[str, Callable[..., Any]] = {}
REG_LOCK = asyncio.Lock()  # protect registry operations


# ---------- Pydantic models ----------
class RegisterToolModel(BaseModel):
    name: str
    description: Optional[str] = None
    # For runtime registration we accept a "type" that indicates built-in behavior
    type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RegisterProviderModel(BaseModel):
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ContextRequest(BaseModel):
    user_id: Optional[str] = None
    project_id: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


# ---------- Helper utilities ----------
async def register_tool_builtin(name: str, handler: Callable, description: str = "", metadata: Optional[Dict]=None):
    async with REG_LOCK:
        TOOLS[name] = {
            "name": name,
            "description": description or getattr(handler, "__doc__", ""),
            "handler": handler,
            "metadata": metadata or {}
        }
        logger.info("Registered tool: %s", name)

async def register_provider_builtin(name: str, handler: Callable, description: str = "", metadata: Optional[Dict]=None):
    async with REG_LOCK:
        PROVIDERS[name] = handler
        logger.info("Registered provider: %s", name)


# ---------- Built-in example providers ----------
async def prov_user(context: ContextRequest):
    """Example user provider returning profile & preferences (simulated)."""
    # In production: fetch from DB or service
    user_id = context.user_id or "anonymous"
    return {
        "user_id": user_id,
        "display_name": f"User {user_id}",
        "preferences": {"locale": "en-IN", "timezone": "Asia/Kolkata"},
    }

async def prov_finance(context: ContextRequest):
    """Example finance provider returning project budgets."""
    # In production: query finance DB or API
    project = (context.project_id or context.params or {}).get("project", "default-project")
    # simulated budgets
    budgets = {
        "default-project": {"budget": 100000, "spent": 40000},
        "Launch": {"budget": 150000, "spent": 40000}
    }
    return {"project_id": context.project_id, "budget_info": budgets.get(project, budgets["default-project"])}

async def prov_project(context: ContextRequest):
    """Example project provider returning metadata."""
    pid = context.project_id or "p000"
    return {"project_id": pid, "name": f"Project {pid}", "owner": "team@example.com", "phase": "planning"}


# ---------- Built-in example tools ----------
async def tool_weather(params: Dict[str, Any]):
    """Simulated weather tool. In prod, call a weather API."""
    city = params.get("city") or params.get("q") or "unknown"
    # simulated weather payload
    return {"city": city, "forecast": "Sunny", "temp_c": 29, "note": "Simulated data. Swap for real API."}

async def tool_budget(params: Dict[str, Any]):
    """Check budget thresholds and return status."""
    project = params.get("project") or "default-project"
    cost = float(params.get("cost", 0))
    # Example: ask finance provider internally if available
    finance_data = await prov_finance(ContextRequest(project_id=project, params={"project": project}))
    budget = finance_data["budget_info"]["budget"]
    status = "within" if cost <= budget else "over"
    return {"project": project, "cost": cost, "budget": budget, "status": status}

async def tool_ai_summarizer(params: Dict[str, Any]):
    """AI summarizer tool: uses LLM if configured, otherwise returns a simple extractive summary."""
    content = params.get("content", "")
    goal = params.get("goal", "Summarize")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        # Optional: call OpenAI's completion/chat here.
        # Example (pseudo-code, uncomment after pip install openai):
        # import openai
        # openai.api_key = openai_key
        # resp = openai.ChatCompletion.create(
        #   model="gpt-5-agentic",
        #   messages=[{"role":"system","content":"You are a summarizer."},{"role":"user","content":f"{goal}\n\n{content}"}],
        #   max_tokens=400
        # )
        # return {"summary": resp.choices[0].message['content'], "via": "openai"}
        return {"summary": f"(LLM would summarize: {content[:200]}...)", "via": "openai-simulated"}
    # fallback naive summarization
    snippet = content[:300] + ("..." if len(content) > 300 else "")
    return {"summary": f"{goal}: {snippet}", "via": "naive"}


# Register builtins on startup
@app.on_event("startup")
async def startup_event():
    await register_provider_builtin("user", prov_user, "User profile & preferences provider")
    await register_provider_builtin("finance", prov_finance, "Finance / budgets provider")
    await register_provider_builtin("project", prov_project, "Project metadata provider")

    await register_tool_builtin("weather", tool_weather, "Weather lookup (simulated)")
    await register_tool_builtin("budget", tool_budget, "Budget check tool")
    await register_tool_builtin("ai_summarizer", tool_ai_summarizer, "LLM summarizer (uses OPENAI_API_KEY if set)")

    logger.info("MCP Server started with %d providers and %d tools", len(PROVIDERS), len(TOOLS))


# ---------- Endpoints ----------
@app.get("/health")
async def health():
    return {"status": "ok", "tools": list(TOOLS.keys()), "providers": list(PROVIDERS.keys())}


@app.get("/tools")
async def list_tools():
    return [{"name": v["name"], "description": v["description"], "metadata": v["metadata"]} for v in TOOLS.values()]


@app.post("/tools/register")
async def register_tool(payload: RegisterToolModel, auth=Depends(require_api_key)):
    """
    Runtime register of a *link* to a tool. For safety, we only allow registering built-in types here.
    In a real system we'd accept a webhook URL or service ID.
    """
    name = payload.name
    t = payload.type or "simulated"
    if name in TOOLS:
        raise HTTPException(400, "tool already registered")
    # For demo support a few named types:
    if t == "weather":
        handler = tool_weather
    elif t == "budget":
        handler = tool_budget
    elif t == "ai_summarizer":
        handler = tool_ai_summarizer
    else:
        raise HTTPException(400, "unsupported tool type for runtime register (demo)")
    await register_tool_builtin(name, handler, payload.description, payload.metadata)
    return {"status": "ok", "name": name}


@app.get("/providers")
async def list_providers():
    return [{"name": name, "doc": PROVIDERS[name].__doc__} for name in PROVIDERS]


@app.post("/providers/register")
async def register_provider(payload: RegisterProviderModel, auth=Depends(require_api_key)):
    # For security and simplicity, only support built-in provider types in this demo
    name = payload.name
    if name in PROVIDERS:
        raise HTTPException(400, "provider already exists")
    t = payload.type or "simulated"
    if t == "user":
        handler = prov_user
    elif t == "finance":
        handler = prov_finance
    elif t == "project":
        handler = prov_project
    else:
        raise HTTPException(400, "unsupported provider type (demo)")
    await register_provider_builtin(name, handler, payload.description, payload.metadata)
    return {"status": "ok", "provider": name}


@app.post("/context/{provider}")
async def get_context(provider: str, payload: ContextRequest, auth=Depends(require_api_key)):
    """
    Retrieve context from a named provider. Accepts user_id, project_id and params in body.
    """
    if provider not in PROVIDERS:
        raise HTTPException(404, "provider not found")
    handler = PROVIDERS[provider]
    # Handler may be sync or async
    if inspect.iscoroutinefunction(handler):
        result = await handler(payload)
    else:
        result = handler(payload)
    return {"provider": provider, "data": result}


@app.get("/tool/{tool_name}")
async def run_tool_get(tool_name: str, request: Request, auth=Depends(require_api_key)):
    """
    Execute a tool using query parameters.
    """
    if tool_name not in TOOLS:
        raise HTTPException(404, "tool not found")
    params = dict(request.query_params)
    handler = TOOLS[tool_name]["handler"]
    if inspect.iscoroutinefunction(handler):
        result = await handler(params)
    else:
        result = handler(params)
    return {"tool": tool_name, "result": result}


@app.post("/tool/{tool_name}")
async def run_tool_post(tool_name: str, payload: Dict[str, Any], auth=Depends(require_api_key)):
    """
    Execute a tool using JSON payload.
    """
    if tool_name not in TOOLS:
        raise HTTPException(404, "tool not found")
    handler = TOOLS[tool_name]["handler"]
    if inspect.iscoroutinefunction(handler):
        result = await handler(payload)
    else:
        result = handler(payload)
    return {"tool": tool_name, "result": result}


# Graceful fallback for root
@app.get("/")
async def root():
    return JSONResponse({
        "service": APP_TITLE,
        "version": APP_VERSION,
        "description": "MCP Server — OpenAPI available at /docs"
    })


# ---------- Run if invoked directly ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9000"))
    uvicorn.run("mcp_server:app", host="0.0.0.0", port=port, reload=True)
```

---

# 2 — Docker + docker-compose

`Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY mcp_server.py /app/mcp_server.py
RUN pip install --no-cache-dir fastapi uvicorn pydantic
EXPOSE 9000
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn","mcp_server:app","--host","0.0.0.0","--port","9000"]
```

`docker-compose.yml`:

```yaml
version: "3.8"
services:
  mcp:
    build: .
    ports:
      - "9000:9000"
    environment:
      - MCP_API_KEY=${MCP_API_KEY:-}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
    restart: unless-stopped
```

Run:

```bash
docker compose up --build
# or locally: MCP_API_KEY=mykey python mcp_server.py
```

---

# 3 — Usage examples (curl + LangGraph client snippet)

## a) curl (no API key configured)

List tools:

```bash
curl http://localhost:9000/tools
```

Get user context:

```bash
curl -X POST http://localhost:9000/context/user -H "Content-Type: application/json" -d '{"user_id":"u001"}'
```

Run weather:

```bash
curl "http://localhost:9000/tool/weather?city=Mumbai"
```

Use budget tool:

```bash
curl -X POST http://localhost:9000/tool/budget -H "Content-Type: application/json" -d '{"project":"Launch","cost":40000}'
```

## b) If MCP_API_KEY is set (e.g. `MCP_API_KEY=secret`), include header:

```bash
curl -H "X-API-Key: secret" http://localhost:9000/tools
```

## c) LangGraph / Python client snippet (how you'd call it from your earlier LangGraph nodes)

```python
import requests

SERVER_URL = "http://localhost:9000"
HEADERS = {"X-API-Key": "secret"}  # if used

def gather_contexts(user_id="u001", project_id="p001"):
    contexts = {}
    for provider in ["user", "finance", "project"]:
        r = requests.post(f"{SERVER_URL}/context/{provider}", json={"user_id": user_id, "project_id": project_id}, headers=HEADERS)
        r.raise_for_status()
        contexts[provider] = r.json()["data"]
    return contexts

def discover_tools():
    r = requests.get(f"{SERVER_URL}/tools", headers=HEADERS)
    r.raise_for_status()
    return r.json()

print(gather_contexts())
print(discover_tools())
```

You can plug these calls into LangGraph nodes the same way you showed earlier.

---

# 4 — Extensibility & Deployment notes

* **Providers & Tools**: In this demo they are in-memory. In production:

  * Load plugins from a folder or register via a service catalog (consul, k8s service discovery).
  * Use service-to-service auth (mTLS, JWKS).
* **Persistence**: Use Postgres/Redis for provider metadata, tool registry, and rate limiting.
* **LLM integration**: `ai_summarizer` contains a placeholder that uses `OPENAI_API_KEY`. Replace with your LLM client of choice (OpenAI, Anthropic, local LLM).
* **Observability**: Add structured logs, metrics (Prometheus), and tracing.
* **Security**: Replace the simple X-API-Key with OAuth2/JWT or mTLS for multi-tenant usage.

---

# 5 — Simple async test (pytest + httpx)

Save this as `test_mcp.py` and run `pytest -q`:

```python
import pytest
from fastapi.testclient import TestClient
from mcp_server import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "tools" in body and "providers" in body

def test_user_context():
    r = client.post("/context/user", json={"user_id":"u123"})
    assert r.status_code == 200
    assert r.json()["data"]["user_id"] == "u123"

def test_weather_tool():
    r = client.get("/tool/weather?city=Delhi")
    assert r.status_code == 200
    assert r.json()["result"]["city"] == "Delhi"
```

