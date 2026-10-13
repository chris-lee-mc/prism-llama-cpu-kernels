"""Hugging Face repo ids used by the experiments.

Pinned values were checked on 2026-10-13 with https://huggingface.co/api/models?search=Qwen3.5
(also exist: -Base variants, 4B, 9B, 27B). resolve() re-checks at run time and falls back to the pinned
value when offline; the result records what exists so the result JSON shows what was used.
"""
from __future__ import annotations

import datetime as _dt
import json
import urllib.error
import urllib.request

API = "https://huggingface.co/api"
QWEN = {"0.8B": "Qwen/Qwen3.5-0.8B", "2B": "Qwen/Qwen3.5-2B"}
BONSAI2_GGUF = ("prism-ml/Ternary-Bonsai-2-27B-gguf", {
    "PQ2_0": "Ternary-Bonsai-2-27B-PQ2_0.gguf",    # 7.21 GB
    "PTQ1_0": "Ternary-Bonsai-2-27B-PTQ1_0.gguf",  # 5.95 GB
})
DRAFT_GGUF = ("ggml-org/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-BF16.gguf")
DRAFT_GGUF_FALLBACK = ("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-BF16.gguf")


def _http_json(url: str, timeout: float = 15.0):
    req = urllib.request.Request(url, headers={"User-Agent": "cr-exps/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _siblings(info) -> list[str]:
    return [s.get("rfilename", "") for s in (info or {}).get("siblings", [])]


def _check_repo(repo: str, fetch) -> tuple[bool | None, list[str], str | None]:
    """Return (exists, file list, error). exists is None when the API cannot be reached."""
    try:
        info = fetch(f"{API}/models/{repo}")
        return True, _siblings(info), None
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False, [], "404"
        return None, [], f"HTTP {e.code}"
    except Exception as e:  # network, proxy, json
        return None, [], f"{type(e).__name__}: {e}"


def _search_ids(query: str, fetch, extra: str = "") -> list[str]:
    try:
        data = fetch(f"{API}/models?search={query}{extra}&limit=100")
        return [m.get("id", "") for m in data]
    except Exception:
        return []


def _base(kind, key, repo, fname, offline):
    return {"kind": kind, "key": key, "repo": repo, "file": fname, "exists": None, "source": "pinned",
            "checked_utc": None if offline else _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "alternatives": [], "note": None}


def resolve(kind: str, key: str | None = None, offline: bool = False, fetch=None) -> dict:
    """Return {kind, key, repo, file, exists, source: api|pinned, alternatives, note}.

    kind: qwen (key 0.8B|2B), bonsai2 (key PQ2_0|PTQ1_0), draft (key ignored). fetch(url) -> json is injectable.
    """
    fetch = fetch or _http_json
    if kind == "qwen":
        repo = QWEN[key]
        out = _base(kind, key, repo, None, offline)
    elif kind == "bonsai2":
        repo, files = BONSAI2_GGUF
        out = _base(kind, key, repo, files[key], offline)
    elif kind == "draft":
        repo, fname = DRAFT_GGUF
        out = _base(kind, key, repo, fname, offline)
    else:
        raise KeyError(f"unknown kind {kind!r}")
    if offline:
        out["note"] = "offline: pinned value, not verified"
        return out
    exists, files, err = _check_repo(out["repo"], fetch)
    if exists is None:
        out["note"] = f"HF API unreachable ({err}); using pinned value"
        return out
    out["source"] = "api"
    out["exists"] = exists
    if exists and out["file"] and out["file"] not in files:
        out["exists"] = False
        out["note"] = f"repo exists but file {out['file']} missing"
    if out["exists"]:
        return out
    # Not found: look for alternatives and record them.
    if kind == "qwen":
        ids = _search_ids("Qwen3.5", fetch)
        alts = [i for i in ids if i.startswith("Qwen/Qwen3.5-") and key.lower() in i.lower() and "base" not in i.lower()]
        out["alternatives"] = alts
    elif kind == "bonsai2":
        ids = _search_ids("Bonsai", fetch, "&author=prism-ml")
        out["alternatives"] = [i for i in ids if "gguf" in i.lower() and "27b" in i.lower()]
    else:
        e2, f2, _ = _check_repo(DRAFT_GGUF_FALLBACK[0], fetch)
        out["alternatives"] = [DRAFT_GGUF_FALLBACK[0]] if e2 else []
        if e2 and DRAFT_GGUF_FALLBACK[1] in f2:
            out.update(repo=DRAFT_GGUF_FALLBACK[0], file=DRAFT_GGUF_FALLBACK[1], exists=True)
            out["note"] = f"primary {DRAFT_GGUF[0]} missing; fell back"
    return out
