"""GPU name -> tag table, peak DRAM bandwidth and RunPod candidate cascade."""
from __future__ import annotations

import re

# Order matters: more specific names first.
TAGS: dict[str, str] = {
    "RTX PRO 6000 Blackwell Max-Q": "pro6000-maxq",
    "RTX PRO 6000 Blackwell Server": "pro6000-server",
    "RTX PRO 6000 Blackwell": "pro6000-ws",
    "RTX 5090": "rtx5090",
    "RTX 4090": "rtx4090",
    "RTX 3090": "rtx3090",
    "L40S": "l40s",
    "RTX 6000 Ada": "rtx6000ada",
}

_PEAK_GBPS: dict[str, float] = {
    "pro6000-maxq": 1792.0,
    "pro6000-server": 1792.0,
    "pro6000-ws": 1792.0,
    "rtx5090": 1792.0,
    "rtx4090": 1008.0,
    "rtx3090": 936.0,
    "l40s": 864.0,
    "rtx6000ada": 960.0,
}

RUNPOD_CANDIDATES: dict[str, list[str]] = {
    # 600 W editions first: E4/E5 timings are from a 600 W Workstation host; Max-Q (300 W) is the fallback
    "pro6000": [
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    ],
    "rtx5090": ["NVIDIA GeForce RTX 5090"],
    "rtx4090": ["NVIDIA GeForce RTX 4090"],
    "rtx3090": ["NVIDIA GeForce RTX 3090"],
    "l40s": ["NVIDIA L40S"],
}


def tag_from_name(name: str | None) -> str:
    """Map a device name to a short tag; unknown names become a slug; empty -> cpu."""
    if not name or not str(name).strip():
        return "cpu"
    low = str(name).lower()
    for sub, tag in TAGS.items():
        if sub.lower() in low:
            return tag
    slug = re.sub(r"[^a-z0-9]+", "-", low).strip("-")
    return slug or "cpu"


def peak_dram_gbps(tag: str) -> float | None:
    return _PEAK_GBPS.get(tag)


def cascade(key: str, cloud: str = "any") -> list[tuple[str, str]]:
    """Ordered (gpu_type_name, cloud) pairs. cloud: any (community then secure), community, secure."""
    if key not in RUNPOD_CANDIDATES:
        raise KeyError(f"unknown gpu cascade key: {key}")
    clouds = {"any": ["COMMUNITY", "SECURE"], "community": ["COMMUNITY"], "secure": ["SECURE"]}
    if cloud not in clouds:
        raise ValueError(f"cloud must be any|community|secure, got {cloud}")
    return [(n, c) for n in RUNPOD_CANDIDATES[key] for c in clouds[cloud]]
