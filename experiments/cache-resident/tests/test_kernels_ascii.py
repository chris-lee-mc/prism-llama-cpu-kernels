from __future__ import annotations

import pathlib

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent


def test_kernels_and_tests_are_ascii():
    files = [f for f in (EXP_DIR / "kernels").rglob("*") if f.is_file()]
    files += list(EXP_DIR.glob("tests/test_kernels_*.py"))
    bad = []
    for f in files:
        if "__pycache__" in f.parts or f.suffix in (".o", ".so", ".pyc"):
            continue
        if any(b > 0x7F for b in f.read_bytes()):
            bad.append(str(f))
    assert not bad, bad
