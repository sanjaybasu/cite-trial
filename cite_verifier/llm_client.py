"""
Multi-vendor, budget-gated LLM client. One entrypoint (`call`) routes by model id to Anthropic
(SDK), OpenAI, or Google Gemini (both via stdlib HTTP — no extra deps). Every call is budget-checked
before and cost-recorded after, uniformly across vendors, so the kill switch and ledger are
provider-agnostic.

Cross-vendor is what makes the study vendor-independent AND removes self-judging bias (the layer's
verifier can be a different vendor than the generator).

Keys: read from the environment; if missing, auto-loaded from ~/.config/ebm-verify/secrets.env
(chmod 600, outside the repo). On Modal these become Modal secrets instead.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error

import cost_ledger as cl
import config

_SECRETS = os.path.expanduser("~/.config/ebm-verify/secrets.env")


def _load_secrets():
    if not os.path.exists(_SECRETS):
        return
    for line in open(_SECRETS):
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_secrets()


class _RateLimited(Exception):
    pass


class _Transient(Exception):
    pass


def provider(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt", "o3", "o4")):
        return "openai"
    if model.startswith("gemini"):
        return "gemini"
    raise ValueError(f"unknown provider for model {model!r}")


# ---- Anthropic (SDK, pooled client) ----
_ANTH = None


def _anthropic(model, system, user, max_tokens):
    global _ANTH
    import anthropic
    if _ANTH is None:
        _ANTH = anthropic.Anthropic(max_retries=0)
    kw = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": user}])
    if system:
        kw["system"] = system
    try:
        r = _ANTH.messages.create(**kw)
    except anthropic.RateLimitError as e:
        raise _RateLimited(str(e))
    except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
        raise _Transient(str(e))
    text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
    return text, r.usage.input_tokens, r.usage.output_tokens


def _http_post(url, payload, headers, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        code = e.code
        body = e.read().decode(errors="ignore")
        if code == 429:
            raise _RateLimited(f"429 {body[:200]}")
        if code in (500, 502, 503, 529):
            raise _Transient(f"{code} {body[:200]}")
        raise RuntimeError(f"HTTP {code}: {body[:300]}")
    except urllib.error.URLError as e:
        raise _Transient(str(e))


def _openai(model, system, user, max_tokens):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    payload = {"model": model, "messages": msgs, "max_completion_tokens": max_tokens}
    d = _http_post("https://api.openai.com/v1/chat/completions", payload,
                   {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    text = d["choices"][0]["message"]["content"] or ""
    u = d.get("usage", {})
    return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def _gemini(model, system, user, max_tokens):
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {"contents": [{"role": "user", "parts": [{"text": user}]}],
               "generationConfig": {"maxOutputTokens": max_tokens}}
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}
    d = _http_post(url, payload, {})
    text = ""
    cand = (d.get("candidates") or [{}])[0]
    for p in cand.get("content", {}).get("parts", []):
        text += p.get("text", "")
    um = d.get("usageMetadata", {})
    out = um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0)
    return text, um.get("promptTokenCount", 0), out


_DISPATCH = {"anthropic": _anthropic, "openai": _openai, "gemini": _gemini}


def _est_input_tokens(system: str, user: str) -> int:
    return int((len(system) + len(user)) / 3.5) + 8


def call(*, model, user, system="", max_tokens=1000, stage, case_id="",
         ledger_path=None, ceiling=None, batch=False) -> dict:
    ledger_path = ledger_path or config.LEDGER_PATH
    ceiling = config.BUDGET_USD_CEILING if ceiling is None else ceiling
    est = cl.cost_usd(model, _est_input_tokens(system, user), max_tokens, batch=batch)
    cl.check_budget(ledger_path, next_call_estimate_usd=est, ceiling=ceiling)

    fn = _DISPATCH[provider(model)]
    last = None
    for attempt in range(6):
        try:
            text, in_tok, out_tok = fn(model, system, user, max_tokens)
            break
        except _RateLimited as e:
            last = e
            if attempt == 5:
                raise
            time.sleep(min(60.0, 8.0 * (attempt + 1)))
        except _Transient as e:
            last = e
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    else:  # pragma: no cover
        raise last

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    total = cl.record(ledger_path, stage=stage, model=model, case_id=case_id,
                      input_tokens=in_tok, output_tokens=out_tok, batch=batch, ts=ts)
    return {"text": text, "input_tokens": in_tok, "output_tokens": out_tok,
            "usd": cl.cost_usd(model, in_tok, out_tok, batch=batch),
            "cumulative_usd": total, "model": model}
