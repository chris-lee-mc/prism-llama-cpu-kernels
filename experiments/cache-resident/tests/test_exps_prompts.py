import hashlib
import json
import pathlib

from exps import make_prompts

FILE = pathlib.Path(make_prompts.DEFAULT_OUT)


def test_generation_is_deterministic():
    a = make_prompts.to_text(make_prompts.generate())
    b = make_prompts.to_text(make_prompts.generate())
    assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()


def test_shape_and_categories():
    rows = make_prompts.generate()
    assert len(rows) == 200
    assert len({r["id"] for r in rows}) == 200
    assert rows[0]["id"] == "p000" and rows[-1]["id"] == "p199"
    for cat in make_prompts.CATEGORIES:
        assert sum(r["category"] == cat for r in rows) == 25
    assert len({r["prompt"] for r in rows}) == 200
    assert all(set(r) == {"id", "category", "prompt"} and r["prompt"].isascii() for r in rows)


def test_committed_file_matches_generator():
    assert FILE.exists()
    assert FILE.read_text() == make_prompts.to_text(make_prompts.generate())
    rows = [json.loads(ln) for ln in FILE.read_text().splitlines()]
    assert len(rows) == 200


def test_load_limit():
    assert len(make_prompts.load(FILE, 7)) == 7
