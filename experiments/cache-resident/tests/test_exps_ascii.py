import pathlib

EXP = pathlib.Path(__file__).resolve().parent.parent


def _non_ascii(paths):
    bad = []
    for p in paths:
        if p.is_file() and "__pycache__" not in p.parts and any(b > 0x7F for b in p.read_bytes()):
            bad.append(str(p))
    return bad


def test_exps_ascii_only():
    assert not _non_ascii(sorted((EXP / "exps").rglob("*")))


def test_exps_tests_ascii_only():
    assert not _non_ascii(sorted(pathlib.Path(__file__).parent.glob("test_exps_*.py")))
