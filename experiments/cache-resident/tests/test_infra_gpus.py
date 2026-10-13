from __future__ import annotations

import pytest

from infra import gpus


@pytest.mark.parametrize("name,tag", [
    ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "pro6000-maxq"),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "pro6000-server"),
    ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "pro6000-ws"),
    ("NVIDIA GeForce RTX 5090", "rtx5090"),
    ("NVIDIA GeForce RTX 4090", "rtx4090"),
    ("NVIDIA GeForce RTX 3090", "rtx3090"),
    ("NVIDIA L40S", "l40s"),
    ("NVIDIA RTX 6000 Ada Generation", "rtx6000ada"),
])
def test_known_tags(name, tag):
    assert gpus.tag_from_name(name) == tag


def test_max_q_before_generic():
    keys = list(gpus.TAGS)
    assert keys.index("RTX PRO 6000 Blackwell Max-Q") < keys.index("RTX PRO 6000 Blackwell")
    assert keys.index("RTX PRO 6000 Blackwell Server") < keys.index("RTX PRO 6000 Blackwell")


def test_slug_and_cpu():
    assert gpus.tag_from_name("Tesla T4") == "tesla-t4"
    assert gpus.tag_from_name("NVIDIA A100-SXM4-80GB") == "nvidia-a100-sxm4-80gb"
    assert gpus.tag_from_name("") == "cpu"
    assert gpus.tag_from_name(None) == "cpu"


def test_peak():
    assert gpus.peak_dram_gbps("rtx5090") == 1792
    assert gpus.peak_dram_gbps("rtx3090") == 936
    assert gpus.peak_dram_gbps("pro6000-maxq") == 1792
    assert gpus.peak_dram_gbps("nope") is None
    for tag in gpus.TAGS.values():
        assert gpus.peak_dram_gbps(tag) is not None


def test_cascade_order():
    c = gpus.cascade("pro6000")
    assert [x[1] for x in c] == ["COMMUNITY", "SECURE"] * 3
    assert "Max-Q" in c[0][0] and c[0][0] == c[1][0]
    assert "Max-Q" not in c[2][0] and "Workstation" in c[2][0]
    assert "Server" in c[4][0]
    assert gpus.cascade("rtx5090") == [("NVIDIA GeForce RTX 5090", "COMMUNITY"), ("NVIDIA GeForce RTX 5090", "SECURE")]
    assert [x[1] for x in gpus.cascade("l40s", "secure")] == ["SECURE"]
    assert [x[1] for x in gpus.cascade("rtx4090", "community")] == ["COMMUNITY"]


def test_cascade_errors():
    with pytest.raises(KeyError):
        gpus.cascade("h100")
    with pytest.raises(ValueError):
        gpus.cascade("rtx5090", "moon")
    assert set(gpus.RUNPOD_CANDIDATES) >= {"pro6000", "rtx5090", "rtx4090", "rtx3090", "l40s"}
