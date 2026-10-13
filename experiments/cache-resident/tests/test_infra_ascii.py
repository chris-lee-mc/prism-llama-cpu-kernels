from __future__ import annotations

import pathlib

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent


def test_infra_and_tests_are_ascii():
    files = list((EXP_DIR / "infra").rglob("*")) + list(EXP_DIR.glob("tests/test_infra_*.py"))
    files += [EXP_DIR / "tests" / "conftest.py", EXP_DIR / "pytest.ini", EXP_DIR / ".gitignore"]
    bad = []
    for f in files:
        if f.is_file() and "__pycache__" not in f.parts:
            data = f.read_bytes()
            if any(b > 0x7F for b in data):
                bad.append(str(f))
    assert not bad, bad
