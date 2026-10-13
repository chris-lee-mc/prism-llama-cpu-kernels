import pathlib

EXP = pathlib.Path(__file__).resolve().parent.parent


def test_harness_ascii_only():
    bad = []
    for p in sorted((EXP / "harness").rglob("*")):
        if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".cu", ".md", ".sh", ".txt"):
            for i, b in enumerate(p.read_bytes()):
                if b > 0x7F:
                    bad.append("%s offset %d" % (p, i))
                    break
    assert not bad, bad


def test_harness_tests_ascii_only():
    bad = []
    for p in sorted(pathlib.Path(__file__).parent.glob("test_harness_*.py")):
        if any(b > 0x7F for b in p.read_bytes()):
            bad.append(str(p))
    assert not bad, bad
