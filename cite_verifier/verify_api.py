"""
CITE — Clinical Inference Tethered to Evidence (RCT Arm-3 intervention API)
==========================================================================
Adjusted from the old ANCHOR v2 API: the deterministic concordance core (shown null in the retro) is
REPLACED by the validated grounded verifier (verifier_llm.py). CITE reads the AI care-plan TEXT and
returns flags (commission | confabulation | unsupported | omission) with a verbatim quote, evidence
basis, and severity. Deployed on Modal (BAA-covered) at the existing 'anchor-v2-verify' endpoint:
  https://waymark--anchor-v2-verify-fastapi-app.modal.run

Endpoints:  POST /v1/verify {case_id, arm, plan_text, context?} -> {status, flags[], latency_ms}
            GET  /v1/health
Auth: Bearer token in CITE_API_KEYS (cite-keys secret). No key configured -> 503 (safe default for PHI).
Mock: CITE_MOCK=1 returns the response SHAPE with no LLM call / no PHI (for Replit integration tests).
Deploy: `modal deploy verify_api.py`  (updates the existing endpoint in place).
"""
from __future__ import annotations

import os
import time

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

CITE_MODEL = os.environ.get("CITE_MODEL", "claude-opus-4-8")
_GRADE_SEV = {"high": 3, "moderate": 2, "low": 1, "very-low": 1, "": 2}


class VerifyRequest(BaseModel):
    case_id: str
    arm: str | None = None
    plan_text: str
    context: dict | None = None


class Flag(BaseModel):
    type: str
    quote: str
    evidence_citation: str
    severity: int


class VerifyResponse(BaseModel):
    case_id: str
    version: str
    status: str
    flags: list[Flag]
    latency_ms: float


def _context_to_text(ctx: dict | None) -> str:
    if not ctx:
        return ""
    return "; ".join(f"{k}: {ctx[k]}" for k in ("age", "sex", "problems", "meds", "labs", "history") if k in ctx)


def _to_cite_flags(v: dict) -> list[dict]:
    ver = v.get("verification", {}) or {}
    out: list[dict] = []
    for it in ver.get("inappropriate_action", []) or []:
        out.append({"type": "commission", "quote": it.get("quote", ""),
                    "evidence_citation": it.get("why", ""), "severity": 2})
    for it in ver.get("evidence_overreach", []) or []:
        why = (it.get("why") or "").lower()
        typ = "confabulation" if any(w in why for w in ("fabricat", "not in", "vignette", "record", "unsupported by")) else "unsupported"
        out.append({"type": typ, "quote": it.get("quote", ""), "evidence_citation": it.get("why", ""), "severity": 2})
    for it in ver.get("high_value_omission", []) or []:
        out.append({"type": "omission", "quote": it.get("action", ""), "evidence_citation": it.get("why", ""),
                    "severity": _GRADE_SEV.get((it.get("grade") or "").lower(), 2)})
    for it in ver.get("therapeutic_inertia", []) or []:
        out.append({"type": "omission", "quote": it.get("quote", ""), "evidence_citation": it.get("why", ""),
                    "severity": _GRADE_SEV.get((it.get("grade") or "").lower(), 2)})
    return out


def run_cite(req: "VerifyRequest") -> "VerifyResponse":
    t0 = time.perf_counter()
    if os.environ.get("CITE_MOCK") == "1":
        flags = [{"type": "commission", "quote": "(mock) example flagged recommendation",
                  "evidence_citation": "(mock) evidence basis", "severity": 2}]
        status = "REVIEW"
    else:
        import config as _cfg
        _cfg.LEDGER_PATH = os.environ.get("CITE_LEDGER", "/tmp/cite_ledger.jsonl")
        try:
            _cfg.BUDGET_USD_CEILING = float(os.environ.get("CITE_CEILING", "100000"))
        except ValueError:
            pass
        open(_cfg.LEDGER_PATH, "a").close()  # ensure writable ledger exists
        import verifier_llm as vf
        v = vf.verify(_context_to_text(req.context), req.plan_text, model=CITE_MODEL,
                      case_id=req.case_id, ledger_path=_cfg.LEDGER_PATH)
        flags = _to_cite_flags(v)
        status = "ESCALATE" if any(f["severity"] >= 3 for f in flags) else ("REVIEW" if flags else "SAFE")
    return VerifyResponse(case_id=req.case_id, version="cite-1.0", status=status,
                          flags=[Flag(**f) for f in flags], latency_ms=(time.perf_counter() - t0) * 1000)


security = HTTPBearer(auto_error=False)


def _auth(credentials: HTTPAuthorizationCredentials = Security(security)):
    keys = {k.strip() for k in os.environ.get("CITE_API_KEYS", "").split(",") if k.strip()}
    if not keys:
        if os.environ.get("CITE_ALLOW_NOAUTH") == "1":
            return "dev"
        raise HTTPException(status_code=503, detail="Auth not configured (set CITE_API_KEYS)")
    if credentials is None or credentials.credentials not in keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


web = FastAPI(title="CITE Verification API", version="cite-1.0")


@web.get("/v1/health")
def health():
    return {"status": "ok", "version": "cite-1.0", "mock": os.environ.get("CITE_MOCK") == "1",
            "auth_configured": bool(os.environ.get("CITE_API_KEYS"))}


@web.post("/v1/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest, _key: str = Depends(_auth)):
    try:
        return run_cite(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


# ── Modal deployment (BAA-covered) — updates the existing 'anchor-v2-verify' endpoint in place ──
import modal  # noqa: E402

_STUDY = os.path.dirname(os.path.abspath(__file__))
app = modal.App("anchor-v2-verify")
_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("anthropic>=0.86", "fastapi", "pydantic")
    .add_local_file(f"{_STUDY}/verify_api.py", "/root/study/verify_api.py")
    .add_local_file(f"{_STUDY}/verifier_llm.py", "/root/study/verifier_llm.py")
    .add_local_file(f"{_STUDY}/llm_client.py", "/root/study/llm_client.py")
    .add_local_file(f"{_STUDY}/config.py", "/root/study/config.py")
    .add_local_file(f"{_STUDY}/cost_ledger.py", "/root/study/cost_ledger.py")
)


@app.function(image=_image,
                    secrets=[modal.Secret.from_name("ebmv-keys"), modal.Secret.from_name("cite-keys")],
                    min_containers=1, timeout=120)
@modal.asgi_app()
def fastapi_app():
    import sys
    if "/root/study" not in sys.path:
        sys.path.insert(0, "/root/study")
    os.environ.setdefault("CITE_LEDGER", "/tmp/cite_ledger.jsonl")
    os.environ.setdefault("CITE_CEILING", "100000")
    return web


if __name__ == "__main__":
    import json, sys
    sys.path.insert(0, _STUDY)
    demo = VerifyRequest(case_id="demo", arm="3",
                         plan_text="Assessment: HFrEF EF30. Plan: continue lisinopril, metoprolol, empagliflozin; "
                                   "monitor and consider an MRA at a future visit.",
                         context={"age": 61, "problems": ["HFrEF EF30"], "labs": "K 4.1, eGFR 64"})
    print(json.dumps(run_cite(demo).model_dump(), indent=2))
