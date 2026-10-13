"""Write exps/data/prompts_e5.jsonl: 200 deterministic agentic/code prompts (8 categories x 25).

  python -m exps.make_prompts [--out exps/data/prompts_e5.jsonl] [--check]

Only random.Random(1234) is used; the output is byte-stable across runs and Python versions >= 3.10.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

SEED = 1234
PER_CAT = 25
CATEGORIES = ["code_write", "code_fix", "code_explain", "shell", "tool_call", "json_extract", "plan", "refactor"]
DEFAULT_OUT = pathlib.Path(__file__).resolve().parent / "data" / "prompts_e5.jsonl"

LANGS = ["Python", "TypeScript", "Go", "Rust", "C++", "Java", "Bash", "SQL"]
WRITE_TASKS = [
    "parses an ISO 8601 date string and returns the day of the week",
    "merges two sorted lists into one sorted list without using sort",
    "removes duplicate lines from a text file while keeping the first occurrence",
    "computes the Levenshtein distance between two strings",
    "implements an LRU cache with O(1) get and put",
    "retries an HTTP request with exponential backoff and jitter",
    "validates an IPv4 address string",
    "flattens an arbitrarily nested list or array",
    "reads a CSV file and returns the average of a named numeric column",
    "debounces a callback so it runs once after calls stop for N milliseconds",
    "finds the longest common prefix of an array of strings",
    "converts a number of seconds into a human readable duration like 2h 3m 4s",
    "implements binary search and returns the insertion point when the key is missing",
    "counts word frequencies in a text and returns the top K words",
    "checks whether a string of brackets is balanced",
    "generates all permutations of a short string",
    "rate limits calls with a token bucket",
    "computes a rolling median over a stream of numbers",
]
BUGS = [
    ("Python", "def mean(xs):\n    total = 0\n    for i in range(1, len(xs)):\n        total += xs[i]\n    return total / len(xs)",
     "the mean is wrong for every input"),
    ("Python", "def add_item(item, bucket=[]):\n    bucket.append(item)\n    return bucket",
     "items leak between unrelated calls"),
    ("JavaScript", "async function load(ids) {\n  const out = [];\n  ids.forEach(async (id) => {\n    out.push(await fetch('/api/' + id));\n  });\n  return out;\n}",
     "the function returns an empty array"),
    ("C", "char *copy(const char *s) {\n    char *p = malloc(strlen(s));\n    strcpy(p, s);\n    return p;\n}",
     "it sometimes crashes under valgrind"),
    ("Go", "func sum(ch chan int) int {\n    total := 0\n    for v := range ch {\n        total += v\n    }\n    return total\n}\n// producer never closes ch",
     "the program hangs forever"),
    ("Python", "def is_palindrome(s):\n    s = s.lower()\n    return s == s.reverse()",
     "it raises AttributeError"),
    ("Java", "public static boolean eq(String a, String b) {\n    return a == b;\n}",
     "it returns false for equal strings built at run time"),
    ("TypeScript", "function last<T>(xs: T[]): T {\n  return xs[xs.length];\n}",
     "it always returns undefined"),
    ("Python", "def chunks(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs) - n, n)]",
     "the final chunk is missing"),
    ("SQL", "SELECT user_id, COUNT(*) FROM orders WHERE COUNT(*) > 3 GROUP BY user_id;",
     "the database reports an invalid use of an aggregate"),
    ("Rust", "fn first_word(s: &String) -> &str {\n    let t = s.clone();\n    &t[..t.find(' ').unwrap_or(t.len())]\n}",
     "it does not compile"),
    ("Bash", "for f in $(ls *.txt); do\n  wc -l $f\ndone",
     "it breaks on file names with spaces"),
]
EXPLAIN = [
    ("Python", "from functools import reduce\nprint(reduce(lambda a, b: a * b, range(1, 8)))"),
    ("Python", "def f(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"),
    ("JavaScript", "const uniq = (xs) => [...new Set(xs)].sort((a, b) => a - b);"),
    ("C", "int f(unsigned x) { int c = 0; while (x) { x &= x - 1; c++; } return c; }"),
    ("Go", "func worker(id int, jobs <-chan int, out chan<- int) {\n    for j := range jobs {\n        out <- j * 2\n    }\n}"),
    ("SQL", "SELECT dept, AVG(salary) OVER (PARTITION BY dept ORDER BY hired ROWS 2 PRECEDING) FROM staff;"),
    ("Rust", "let v: Vec<u32> = (1..=10).filter(|x| x % 3 == 0).map(|x| x * x).collect();"),
    ("Bash", "find . -name '*.log' -mtime +7 -print0 | xargs -0 gzip"),
    ("Python", "import heapq\ndef top(xs, k):\n    return heapq.nlargest(k, xs)"),
    ("TypeScript", "type Json = string | number | boolean | null | Json[] | { [k: string]: Json };"),
]
SHELL = [
    "lists the 10 largest files under the current directory",
    "counts the lines of code in all .py files recursively",
    "finds all files modified in the last 24 hours",
    "prints the 5 most frequent words in a file called notes.txt",
    "renames every .jpeg file in a folder to .jpg",
    "shows which process is listening on TCP port 8080",
    "creates a gzip compressed tar archive of a directory called src",
    "replaces the string foo with bar in all .txt files in place",
    "prints the disk usage of each subdirectory sorted by size",
    "kills every process whose name contains node",
    "shows the last 50 lines of a log file and keeps following it",
    "downloads a file with curl and verifies its sha256 against a known value",
    "lists git branches merged into main",
    "extracts the third column of a CSV file and removes duplicates",
    "prints the total memory used by all python processes",
    "checks whether a TCP port is open on a remote host",
]
TOOLS = ('Tools: read_file(path), write_file(path, content), run_shell(cmd), search_code(query), '
         'http_get(url), list_dir(path).')
TOOL_TASKS = [
    "find where the function parse_config is defined",
    "show the contents of README.md",
    "run the unit tests with pytest and report the result",
    "create a file hello.py that prints hello world",
    "list the files in the src directory",
    "fetch https://example.com/api/status",
    "search the codebase for TODO comments",
    "check the current git branch",
    "install the package requests with pip",
    "append a line to CHANGELOG.md saying the release is done",
    "find all usages of the class UserSession",
    "run the linter on the file app/main.py",
]
PEOPLE = ["Maria Lopez", "Chen Wei", "Aisha Khan", "Tom Becker", "Ivan Petrov", "Sara Novak", "Liam O'Brien", "Yuki Tanaka"]
CITIES = ["Lisbon", "Osaka", "Denver", "Nairobi", "Hamburg", "Austin", "Lyon", "Quito"]
ITEMS = ["a standing desk", "two monitors", "a mechanical keyboard", "a laptop stand", "an ergonomic chair", "a webcam"]
GOALS = [
    "migrate a small Flask app from SQLite to PostgreSQL",
    "add request logging to an existing Express server",
    "find and fix a memory leak in a long running Go service",
    "set up CI that runs tests and a linter on every pull request",
    "split a 2000 line Python module into a package",
    "add pagination to a REST endpoint that returns all rows",
    "write a script that backs up a directory to cloud storage every night",
    "upgrade a project from Python 3.8 to 3.12 and fix deprecations",
    "profile a slow SQL query and add the right index",
    "add type hints to an untyped Python codebase step by step",
    "containerize a Node service with a small production image",
    "debug a flaky integration test that fails one run in ten",
]
REFACTORS = [
    ("Python", "def f(a, b, c):\n    if a:\n        if b:\n            if c:\n                return 1\n    return 0", "remove the nesting with early returns"),
    ("JavaScript", "function total(xs) {\n  var t = 0;\n  for (var i = 0; i < xs.length; i++) { t += xs[i].price * xs[i].qty; }\n  return t;\n}", "use modern array methods and const"),
    ("Python", "class A:\n    def __init__(self):\n        self.items = []\n    def add(self, x):\n        self.items.append(x)\n    def get(self):\n        return self.items", "make it a dataclass"),
    ("Java", "String s = \"\";\nfor (String w : words) { s = s + w + \",\"; }", "use a StringBuilder or String.join"),
    ("Go", "func Max(a, b int) int {\n    if a > b {\n        return a\n    } else {\n        return b\n    }\n}", "simplify and make it generic"),
    ("Python", "def read(p):\n    f = open(p)\n    d = f.read()\n    f.close()\n    return d", "use a context manager and pathlib"),
    ("TypeScript", "function area(s: any) {\n  if (s.kind == 'c') return 3.14 * s.r * s.r;\n  if (s.kind == 's') return s.w * s.w;\n}", "use a discriminated union"),
    ("C++", "for (int i = 0; i < v.size(); i++) { out.push_back(f(v[i])); }", "use std::transform and reserve"),
    ("Python", "result = []\nfor x in data:\n    if x % 2 == 0:\n        result.append(x * x)", "use a comprehension"),
    ("SQL", "SELECT * FROM a, b WHERE a.id = b.a_id AND b.flag = 1", "use explicit joins and select only needed columns"),
]


def _unique(rng: random.Random, make, n: int) -> list[str]:
    out, seen, tries = [], set(), 0
    while len(out) < n:
        tries += 1
        if tries > 10000:
            raise RuntimeError("could not build enough unique prompts")
        p = make(rng)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _code_write(rng):
    return f"Write a {rng.choice(LANGS)} function that {rng.choice(WRITE_TASKS)}. Return only the code and a short usage example."


def _code_fix(rng):
    lang, code, symptom = rng.choice(BUGS)
    ask = rng.choice(["Explain the bug in one sentence and give the corrected code.",
                      "Give only the corrected code, with a one line comment on the change.",
                      "List the root cause, then show a minimal diff.",
                      "Fix it and add one test that would have caught it."])
    return f"This {lang} code has a bug: {symptom}.\n\n{code}\n\n{ask}"


def _code_explain(rng):
    lang, code = rng.choice(EXPLAIN)
    extra = rng.choice(["Mention the time complexity.", "Mention any edge cases.", "Explain it to a junior developer.",
                        "Answer in three bullet points."])
    return f"Explain what this {lang} snippet does.\n\n{code}\n\n{extra}"


def _shell(rng):
    ctx = rng.choice(["on Linux", "on macOS", "using only POSIX tools", "that is safe to run in a script with set -e"])
    return f"Give one bash command {ctx} that {rng.choice(SHELL)}. Then explain each flag in one short line."


def _tool_call(rng):
    where = rng.choice(["The repo root is /work/app.", "The project uses Python 3.12.", "You are in a fresh checkout.",
                        "The working directory is /srv/site."])
    return (f"{TOOLS} {where}\nRespond with exactly one JSON object {{\"tool\": name, \"args\": {{...}}}} and nothing else.\n"
            f"Task: {rng.choice(TOOL_TASKS)}.")


def _json_extract(rng):
    who, city, item = rng.choice(PEOPLE), rng.choice(CITIES), rng.choice(ITEMS)
    qty, price, day = rng.randint(1, 5), rng.randint(20, 900), rng.randint(1, 28)
    text = (f"On March {day}, {who} from {city} ordered {item} (quantity {qty}) for ${price}. "
            f"Delivery should reach {city} within {rng.randint(2, 9)} business days.")
    return (f"Extract JSON with the keys customer, city, item, quantity, price_usd, order_day from the text below. "
            f"Output only the JSON.\n\n{text}")


def _plan(rng):
    steps = rng.choice([4, 5, 6, 7])
    return (f"You are a coding agent with read_file, write_file, run_shell and search_code tools. "
            f"Plan how to {rng.choice(GOALS)}. Give {steps} numbered steps and name the tool used in each step.")


def _refactor(rng):
    lang, code, goal = rng.choice(REFACTORS)
    tail = rng.choice(["Return only the new code.", "Return the new code and one sentence on why it is better.",
                       "Return the new code and list any behavior risks.", "Return a unified diff."])
    return f"Refactor this {lang} code to {goal}. Keep the behavior the same.\n\n{code}\n\n{tail}"


MAKERS = {"code_write": _code_write, "code_fix": _code_fix, "code_explain": _code_explain, "shell": _shell,
          "tool_call": _tool_call, "json_extract": _json_extract, "plan": _plan, "refactor": _refactor}


def generate() -> list[dict]:
    rng = random.Random(SEED)
    rows = []
    for cat in CATEGORIES:
        for p in _unique(rng, MAKERS[cat], PER_CAT):
            rows.append({"id": "p%03d" % len(rows), "category": cat, "prompt": p})
    return rows


def to_text(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=True, sort_keys=False) + "\n" for r in rows)


def load(path=DEFAULT_OUT, limit: int | None = None) -> list[dict]:
    rows = [json.loads(ln) for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]
    return rows[:limit] if limit else rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="exps.make_prompts", description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--check", action="store_true", help="fail if the file on disk differs from a fresh generation")
    a = ap.parse_args(argv)
    text = to_text(generate())
    out = pathlib.Path(a.out)
    if a.check:
        ok = out.exists() and out.read_text() == text
        print("ok" if ok else "MISMATCH")
        return 0 if ok else 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(text.splitlines())} prompts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
