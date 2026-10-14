"""Write exps/data/prompts_e5.jsonl: 200 deterministic agentic/code prompts (8 categories x 25).

  python -m exps.make_prompts [--out exps/data/prompts_e5.jsonl] [--check]
  python -m exps.make_prompts --longctx [--out exps/data/prompts_r2a.jsonl] [--check] [--stats]

Only random.Random(1234) is used; the output is byte-stable across runs and Python versions >= 3.10.
--longctx writes the R2a set (60 long-context prompts built from files of this repo, see the R2a section below).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import statistics
import subprocess
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


def shuffled(rows: list[dict], seed: int, stratify: bool = False) -> list[dict]:
    """Fixed-seed permutation. load(limit) keeps file order, which is category blocks of 25: a prefix is not a sample.

    stratify: shuffle inside each category, then deal one prompt per category in turn, so that every prefix is
    category balanced (a run cut by a time budget still covers all categories).
    """
    rng = random.Random(seed)
    if not stratify:
        return rng.sample(rows, len(rows))
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    decks = [rng.sample(v, len(v)) for _, v in sorted(by_cat.items())]
    out = []
    for i in range(max((len(d) for d in decks), default=0)):
        out += [d[i] for d in decks if i < len(d)]
    return out


# ---------------------------------------------------------------- R2a: long-context agentic prompts
#
# 40 code-edit prompts (one real source file of 2-8K tokens each, 20 asking for the full modified file and 20 for a unified diff)
# and 20 multi-turn tool-call traces (tool schema, 3-5 prior calls with real results, ask for the next call). Everything is built
# from files of this repo, so no network is needed. The data file is the authority: the file lists below are pinned, and every row
# records the sha of each source it used, so a later edit of a source file shows up as drift (see drift_rows), not as a silent change.

R2A_OUT = DEFAULT_OUT.parent / "prompts_r2a.jsonl"
R2A_SEED = 20261014
R2A_CATEGORIES = ["edit_full_file", "edit_diff", "toolcall_trace"]
R2A_N_PREDICT = {"edit_full_file": 320, "edit_diff": 512, "toolcall_trace": 128}
R2A_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent
CHARS_PER_TOKEN = 3.5
EDIT_TOKENS = (2000, 8000)
TRACE_MIN_TOKENS = 2100        # estimated tokens (chars / 3.5) a trace is lengthened to, when its file is long enough
START_BEFORE = 3            # lines of copy before the first edited line in an edit_full_file answer
ANCHOR_MIN_CHARS, ANCHOR_SHOW = 8, 100     # the start line of a full_file answer is named by its text: at least 8 chars, shown up to 100
ANCHOR_MAX_BACK, ANCHOR_MAX_TOKENS = 15, 60     # the walk back to a unique start line stops at 15 lines or 60 tokens
EDIT_KINDS = ["rename", "add_param", "fix_bug", "add_logging", "refactor"]
TRACE_TURNS = [3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]
LANG_OF_EXT = {".py": "python", ".c": "c", ".cpp": "cpp", ".h": "cpp", ".cu": "cpp", ".cuh": "cpp"}
CODE_EXTS = tuple(LANG_OF_EXT)
C_NOT_FUNCS = {"if", "for", "while", "switch", "return", "sizeof", "defined", "else", "do", "catch", "static_assert", "typedef",
               "struct", "class", "enum", "union", "namespace", "using", "extern", "template", "operator"}
SUFFIXES = ["_v2", "_ex", "_checked", "_impl"]

# files chosen once with random.Random(R2A_SEED) among tracked ASCII-only sources of 7-28K chars (see select_files)
R2A_EDIT_FILES: list[str] = [
    "src/llama-mmap.cpp", "src/llama-kv-cache-dsa-iswa.cpp", "src/llama-model-saver.cpp", "src/llama-memory-hybrid.cpp",
    "src/llama-kv-cache-iswa.cpp", "src/llama-memory-hybrid-iswa.cpp", "src/llama.cpp", "src/llama-kv-cache-msa.cpp",
    "src/llama-hparams.cpp", "common/ngram-cache.cpp", "common/jinja/runtime.h", "common/dspark-markov.cu", "common/json.cpp",
    "common/llguidance.cpp", "common/preset.cpp", "common/hf-cache.cpp", "common/jinja/caps.cpp", "tools/server/bench/bench.py",
    "tools/rpc/rpc-server.cpp", "tools/export-lora/export-lora.cpp", "tools/server/server-queue.cpp", "tools/server/server.cpp",
    "tools/tuning/fa-vec.cpp", "tools/tokenize/tokenize.cpp", "tools/mtmd/mtmd-cli.cpp", "ggml/src/ggml-cuda/upscale.cu",
    "ggml/src/ggml-cuda/set-rows.cu", "ggml/src/ggml-cuda/unary.cu", "ggml/src/ggml-cuda/ssm-conv.cu",
    "ggml/src/ggml-cuda/mmid.cu", "ggml/src/ggml-cuda/im2col.cu", "ggml/src/ggml-cuda/fwht.cu",
    "ggml/src/ggml-cuda/mmq-hopper-q1.cu", "ggml/src/ggml-cuda/dsv4-hc.cu", "experiments/cache-resident/harness/counters.py",
    "experiments/cache-resident/harness/l2persist.py", "experiments/cache-resident/harness/p0_l2_check.py",
    "experiments/cache-resident/kernels/toy_hybrid.py", "tools/kv-mean-center/kv-mean-center.cpp",
    "experiments/cache-resident/kernels/bench_gemv.py",
]
R2A_TRACE_FILES: list[str] = [
    "src/llama-batch.cpp", "src/llama-context.cpp", "src/models/delta-net-base.cpp", "src/llama-kv-cache-dsa.cpp",
    "src/llama-kv-cache.cpp", "common/chat-auto-parser-generator.cpp", "common/jinja/value.cpp", "common/ngram-map.cpp",
    "common/arg.cpp", "common/sampling.cpp", "tools/mtmd/legacy-models/llava_surgery_v2.py", "tools/ui/embed.cpp",
    "tools/server/server-tools.cpp", "tools/tuning/bench.cpp", "tools/mtmd/models/pockettts-seanet.cpp",
    "ggml/src/ggml-cuda/ggml-cuda.cu", "ggml/src/ggml-cuda/argsort.cu", "ggml/src/ggml-cuda/vecdotq.cuh",
    "ggml/src/ggml-cuda/cpy-utils.cuh", "ggml/src/ggml-cuda/ssm-scan.cu",
]

TOOL_SCHEMA = [
    {"name": "list_dir", "description": "List the entries of a directory (directories end with /).",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "read_file", "description": "Read a file, optionally a 1-based inclusive line range, with line numbers.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"},
                                                     "end_line": {"type": "integer"}}, "required": ["path"]}},
    {"name": "search_code", "description": "Search for a regular expression in the files of a directory (grep -n style).",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["query"]}},
    {"name": "run_shell", "description": "Run a shell command in the repository root and return its output.",
     "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}},
    {"name": "write_file", "description": "Replace the whole content of a file.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}},
    {"name": "apply_patch", "description": "Apply a unified diff to the working tree.",
     "parameters": {"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]}},
]
TRACE_TOOLS = [{"type": "function", "function": t} for t in TOOL_SCHEMA]      # OpenAI format: the chat template renders the tools block
TRACE_SYSTEM = ("You are a coding agent working in a repository checked out at /work/repo. Use the tools to inspect and change the "
                "code. Call one tool per turn.")


def approx_tokens(chars: int) -> int:
    return int(round(chars / CHARS_PER_TOKEN))


def messages_chars(messages: list[dict], tools: list[dict] | None = None) -> int:
    """Characters the chat template will render from the messages: contents, tool call JSON and the tool schema."""
    n = sum(len(m.get("content") or "") + (len(json.dumps(m["tool_calls"])) if m.get("tool_calls") else 0) for m in messages)
    return n + (len(json.dumps(tools)) if tools else 0)


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def lang_of(path: str) -> str | None:
    return LANG_OF_EXT.get(pathlib.Path(path).suffix)


def _split_params(params: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in params:
        if ch in "(<[{":
            depth += 1
        elif ch in ")>]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + ([cur] if cur.strip() else [])


def _first_param(params: str) -> tuple[str | None, bool]:
    """(name, is_pointer) of the first parameter of a C/C++ or Python parameter list."""
    parts = [p.strip() for p in _split_params(params) if p.strip()]
    parts = [p for p in parts if p not in ("self", "cls", "void")]
    if not parts:
        return None, False
    first = re.split(r"=", parts[0])[0].split(":")[0].strip()
    m = re.search(r"([A-Za-z_]\w*)\s*(\[\])?$", first)
    return (m.group(1) if m else None), ("*" in first or "[" in first)


def _ret_kind(head: str) -> str:
    h = re.sub(r"\b(static|inline|extern|constexpr|__global__|__device__|__host__|__forceinline__|template\s*<[^>]*>)\b", "", head)
    if "*" in h:
        return "ptr"
    if re.search(r"\bvoid\b", h):
        return "void"
    if re.search(r"\bbool\b", h):
        return "bool"
    return "other"


def find_functions(text: str, lang: str) -> list[dict]:
    """Top-level function definitions: name, 0-based line, end line of the body, first parameter, call count, loops."""
    lines = text.split("\n")
    funcs: list[dict] = []
    for i, ln in enumerate(lines):
        if lang == "python":
            m = re.match(r"^(\s*)def (\w+)\((.*)", ln)
            if not m:
                continue
            indent, name = len(m.group(1)), m.group(2)
            j = i
            sig = ln
            while not re.search(r"\)\s*(->.*)?:\s*(#.*)?$", lines[j]) and j < min(i + 12, len(lines) - 1):
                j += 1
                sig += " " + lines[j].strip()
            end = j
            for k in range(j + 1, len(lines)):
                if lines[k].strip() and len(lines[k]) - len(lines[k].lstrip()) <= indent and not lines[k].lstrip().startswith(("#", ")")):
                    break
                end = k
            params = sig[sig.index("(") + 1:]
            fp, ptr = _first_param(params.rsplit(")", 1)[0])
            ret = "other"
        else:
            if not ln or ln[0] in " \t#}/*" or "(" not in ln:
                continue
            head, j = ln, i
            while "{" not in head and ";" not in head and j < min(i + 8, len(lines) - 1):
                j += 1
                head += " " + lines[j].strip()
            cut = min([x for x in (head.find("{"), head.find(";")) if x >= 0] or [len(head)])
            if head[cut:cut + 1] != "{":
                continue
            pre = head[:head.index("(")]
            m = re.search(r"([A-Za-z_]\w*)\s*$", pre)
            if not m or m.group(1) in C_NOT_FUNCS or "=" in pre or re.match(r"^(struct|class|enum|union|namespace|using|typedef)\b", head):
                continue
            name = m.group(1)
            end = j
            for k in range(j + 1, len(lines)):
                if lines[k].startswith("}"):
                    end = k
                    break
            else:
                continue
            params = head[head.index("(") + 1:cut]
            params = params[:params.rfind(")")] if ")" in params else params
            fp, ptr = _first_param(params)
            ret = _ret_kind(pre[:m.start(1)])
        body = lines[j + 1:end + 1]
        loop_re = r"^\s+for .+ in .+:\s*$" if lang == "python" else r"^\s+for \(.*\)\s*\{?\s*$"
        loops = [(j + 1 + n, b) for n, b in enumerate(body) if re.match(loop_re, b)]
        lt = [lp for lp in loops if (re.search(r"range\(", lp[1]) if lang == "python" else re.search(r";[^;=]*[^<]<[^<=;][^;]*;", lp[1]))]
        funcs.append({"name": name, "line": i, "body_start": j + 1, "end": end, "n_body": end - j, "first_param": fp, "first_ptr": ptr,
                      "ret": ret, "loop": loops[0] if loops else None, "loop_lt": lt[0] if lt else None})
    for f in funcs:
        f["calls"] = len(re.findall(r"\b%s\s*\(" % re.escape(f["name"]), text)) - 1
    return funcs


def _log_macro(text: str, lang: str) -> str:
    if lang == "python":
        return "logging.debug" if "import logging" in text else "print(..., file=sys.stderr)"
    for m in ("LOG_DBG", "LOG_INF", "GGML_LOG_DEBUG", "GGML_LOG_INFO", "LLAMA_LOG_DEBUG", "LLAMA_LOG_INFO"):
        if m + "(" in text:
            return m
    return "fprintf(stderr, ...)"


def _pick(rng: random.Random, funcs: list[dict], pred) -> dict | None:
    ok = [f for f in funcs if pred(f)]
    return rng.choice(ok) if ok else None


def edit_instruction(kind: str, text: str, lang: str, rng: random.Random) -> tuple[str, dict] | None:
    """(instruction, meta) for one edit kind, or None when the file has nothing for it."""
    funcs = find_functions(text, lang)
    med = lambda f: 6 <= f["n_body"] <= 140
    lines = text.split("\n")
    first_use = lambda name: next(i for i, ln in enumerate(lines) if re.search(r"\b%s\b" % re.escape(name), ln))
    if kind == "rename":
        f = _pick(rng, funcs, lambda f: f["calls"] >= 1 and len(f["name"]) >= 5 and f["name"] != "main")
        if not f:
            return None
        new = next(f["name"] + s for s in SUFFIXES if f["name"] + s not in text)
        msg = rng.choice([f"Rename the function `{f['name']}` to `{new}` and update every call site in this file.",
                          f"Rename `{f['name']}` to `{new}`. Update the definition and all {f['calls']} call site(s) in this file."])
        return msg, {"function": f["name"], "new_name": new, "edit_line": first_use(f["name"])}
    if kind == "add_param":
        f = _pick(rng, funcs, lambda f: f["calls"] >= 1 and med(f) and f["name"] != "main")
        if not f:
            return None
        if lang == "python":
            msg = (f"Add a new keyword parameter `verbose: bool = False` to `{f['name']}`. When it is true, print one line to stderr "
                   f"at the start of the function. Existing call sites must keep working.")
        else:
            msg = (f"Add a new trailing parameter `bool verbose` to `{f['name']}`. Give it the default value `false` when the language "
                   f"allows default arguments, otherwise pass `false` at every existing call site in this file. When `verbose` is true, "
                   f"print one line to stderr at the start of the function.")
        return msg, {"function": f["name"], "edit_line": first_use(f["name"])}
    if kind == "fix_bug":
        f = _pick(rng, funcs, lambda f: med(f) and f["loop_lt"])
        if f:
            return (f"Bug report: the loop `{f['loop_lt'][1].strip()}` in `{f['name']}` skips its last element (off-by-one in the loop "
                    f"bound). Fix the bound and keep everything else unchanged."), {"function": f["name"], "edit_line": f["loop_lt"][0]}
        f = _pick(rng, funcs, lambda f: med(f) and f["first_param"] and (f["first_ptr"] or lang == "python"))
        if not f:
            return None
        if lang == "python":
            msg = (f"Bug report: `{f['name']}` raises an exception when `{f['first_param']}` is None. Add an early return of a "
                   f"sensible default when `{f['first_param']}` is None.")
        else:
            msg = (f"Bug report: `{f['name']}` crashes when its argument `{f['first_param']}` is NULL. Add an early return with a "
                   f"safe default value (pick it from the return type) when `{f['first_param']}` is NULL.")
        return msg, {"function": f["name"], "edit_line": f["line"]}
    if kind == "add_logging":
        f = _pick(rng, funcs, lambda f: med(f) and f["first_param"])
        if not f:
            return None
        return (f"Add a debug log line at the start of `{f['name']}` that prints the function name and the value of `{f['first_param']}` "
                f"(use {_log_macro(text, lang)}). Do not change anything else."), {"function": f["name"], "edit_line": f["body_start"]}
    if kind == "refactor":
        f = _pick(rng, funcs, lambda f: med(f) and f["loop"])
        if f:
            helper = f["name"] + "_loop"
            msg = (f"Refactor `{f['name']}`: extract its first loop (`{f['loop'][1].strip()}`) into a new helper function `{helper}` "
                   f"defined right above `{f['name']}`, pass what it needs as parameters, and call it from `{f['name']}`. "
                   f"The behavior must stay the same.")
            return msg, {"function": f["name"], "helper": helper, "edit_line": f["line"]}
        f = _pick(rng, funcs, lambda f: 16 <= f["n_body"] <= 140)
        if not f:
            return None
        helper = f["name"] + "_tail"
        return (f"Refactor `{f['name']}`: move the second half of its body into a new helper function `{helper}` defined right above "
                f"it and call it from `{f['name']}`. The behavior must stay the same."), {"function": f["name"], "helper": helper, "edit_line": f["line"]}
    raise ValueError(kind)


def start_line_of(lines: list[str], edit_line: int, before: int = START_BEFORE, text: str | None = None) -> int | None:
    """0-based first line of a full_file answer. The prompt names this line by its text, so the text must occur once in the file.
    Walk back from `before` lines above the edit to the nearest line of >= ANCHOR_MIN_CHARS chars that occurs once; the walk is capped
    by ANCHOR_MAX_BACK lines and ANCHOR_MAX_TOKENS tokens. If none is found, try the lines from there up to the edit line. None when
    no line is unique."""
    text = "\n".join(lines) if text is None else text
    top = max(0, edit_line - before)

    def ok(i: int) -> bool:
        a = lines[i].strip()[:ANCHOR_SHOW]
        return len(a) >= ANCHOR_MIN_CHARS and text.count(a) == 1
    for i in range(top, max(0, edit_line - ANCHOR_MAX_BACK) - 1, -1):
        if approx_tokens(len("\n".join(lines[i:edit_line]))) > ANCHOR_MAX_TOKENS:
            break
        if ok(i):
            return i
    return next((i for i in range(top + 1, edit_line + 1) if ok(i)), None)


def edit_prompt(path: str, text: str, kind: str, variant: str, lang: str, rng: random.Random) -> tuple[list[dict], dict] | None:
    ins = edit_instruction(kind, text, lang, rng)
    if ins is None:
        return None
    instruction, meta = ins
    fence = lang
    meta["edit_offset_tokens"] = None
    if variant == "full_file":
        # a 320-token answer that starts at line 1 would only copy the file head (the edit is often thousands of tokens down),
        # so the answer starts a few lines before the first edited line, at a line whose text occurs once in the file
        lines = text.split("\n")
        start = start_line_of(lines, meta["edit_line"], text=text)
        if start is None:
            return None
        shown = lines[start].strip()
        shown = shown if len(shown) <= ANCHOR_SHOW else shown[:ANCHOR_SHOW] + "..."
        meta.update(start_line=start, edit_offset_tokens=approx_tokens(len("\n".join(lines[start:meta["edit_line"]]))))
        spec = (f"Output the modified file starting with the line `{shown}` (line {start + 1} of the original) and continuing to the "
                f"last line of the file, in a single ```{fence} code block. Do not output the lines before it. Keep every unchanged "
                f"line exactly as it is and do not add any explanation.")
    else:
        spec = (f"Output only a unified diff for this single file (`--- a/{path}` and `+++ b/{path}` headers, 3 lines of context) in a "
                f"single ```diff code block. Do not output the whole file and do not add any explanation.")
    body = f"File: {path}\n\n```{fence}\n{text}{'' if text.endswith(chr(10)) else chr(10)}```\n\nTask: {instruction}\n\n{spec}"
    return [{"role": "user", "content": body}], dict(meta, kind=kind, variant=variant, lang=lang)


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def read_source(root, rel: str) -> str | None:
    p = pathlib.Path(root) / rel
    if not p.is_file():
        return None
    try:
        t = p.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return None
    return t if t.isascii() and "\r" not in t and "\0" not in t else None


def tracked_files(root) -> list[str]:
    """Files tracked by git under root. Listings and greps use only these, so an untracked or ignored file (a .env, results/,
    a scratch file) can never get into a prompt."""
    p = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"R2a: {root} is not a git checkout; directory listings use tracked files only")
    return [f for f in p.stdout.split("\0") if f]


def dir_listing(root, rel_dir: str, limit: int = 40) -> str:
    """ls of a directory as the tracked files see it (sub directories end with /)."""
    pre = rel_dir.rstrip("/") + "/"
    kids: dict[str, bool] = {}
    for f in tracked_files(root):
        if f.startswith(pre):
            name, _, rest = f[len(pre):].partition("/")
            if not name.startswith("."):
                kids[name] = kids.get(name, False) or bool(rest)
    return "\n".join(_ascii(n + "/" if kids[n] else n) for n in sorted(kids)[:limit])


def grep_dir(root, rel_dir: str, first: str, pattern: str, limit: int = 12) -> str:
    """grep -n of a word in `first` (a path) and in the code files of its directory, sorted, at most `limit` hits."""
    pre = rel_dir.rstrip("/") + "/"
    near = sorted(f for f in tracked_files(root) if f.startswith(pre) and "/" not in f[len(pre):] and f.endswith(CODE_EXTS) and f != first)
    files = [first] + near[:8]
    rx = re.compile(r"\b%s\b" % re.escape(pattern))
    hits = []
    for rel in sorted(files):
        src = read_source(root, rel)
        if src is None:
            continue
        hits += [f"{rel}:{n}:{_ascii(ln.strip())[:150]}" for n, ln in enumerate(src.split("\n"), 1) if rx.search(ln)]
    return "\n".join(hits[:limit]) or "(no matches)"


def numbered(text: str, a: int, b: int) -> str:
    lines = text.split("\n")
    return "\n".join(f"{n:6d}\t{ln[:200]}" for n, ln in enumerate(lines[a - 1:b], a))


def trace_prompt(root, rel: str, text: str, n_turns: int, rng: random.Random) -> tuple[list[dict], dict] | None:
    lang = lang_of(rel)
    funcs = [f for f in find_functions(text, lang) if f["calls"] >= 1 and f["n_body"] >= 6 and len(f["name"]) >= 5]
    if len(funcs) < 2:
        return None
    fn, other = rng.sample(funcs, 2)
    d = str(pathlib.PurePosixPath(rel).parent)
    n_lines = len(text.split("\n"))

    def region(f):
        a = max(1, f["line"] + 1 - rng.randint(0, 6))
        return a, min(n_lines, a + rng.randint(110, 170))
    ra, rb = region(fn)
    oa, ob = region(other)
    plan = rng.choice([["ls", "grep", "read", "grep2", "read2"], ["grep", "read", "wc", "grep2", "read2"],
                       ["head", "grep", "read", "grep2", "read2"]])[:n_turns]
    task = rng.choice([f"Add a debug log line to `{fn['name']}` in `{rel}`.",
                       f"I want to rename `{fn['name']}` in `{rel}` to `{fn['name']}_v2`. Check where it is used before changing anything.",
                       f"Fix the bug where `{fn['name']}` in `{rel}` mishandles an empty input.",
                       f"Explain what `{fn['name']}` in `{rel}` does and check how `{other['name']}` relates to it."])

    def build(more: int) -> list[dict]:
        """more: extra lines of every read range. A short trace is lengthened this way until it reaches TRACE_MIN_TOKENS."""
        rb2, ob2 = min(n_lines, rb + more), min(n_lines, ob + more)
        steps = {
            "ls": ("list_dir", {"path": d}, dir_listing(root, d)),
            "grep": ("search_code", {"query": fn["name"], "path": d}, grep_dir(root, d, rel, fn["name"])),
            "read": ("read_file", {"path": rel, "start_line": ra, "end_line": rb2}, numbered(text, ra, rb2)),
            "wc": ("run_shell", {"cmd": f"wc -l {rel}"}, f"{n_lines} {rel}"),
            "grep2": ("search_code", {"query": other["name"], "path": d}, grep_dir(root, d, rel, other["name"], 10)),
            "read2": ("read_file", {"path": rel, "start_line": oa, "end_line": ob2}, numbered(text, oa, ob2)),
            "head": ("read_file", {"path": rel, "start_line": 1, "end_line": 40}, numbered(text, 1, 40)),
        }
        msgs = [{"role": "system", "content": TRACE_SYSTEM}, {"role": "user", "content": task}]
        for k, key in enumerate(plan):
            tool, args, result = steps[key]
            call = {"id": f"call_{k + 1}", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}
            msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
            msgs.append({"role": "tool", "tool_call_id": call["id"], "content": result})
        return msgs
    more = 0
    msgs = build(more)
    while approx_tokens(messages_chars(msgs, TRACE_TOOLS)) < TRACE_MIN_TOKENS and more < n_lines:
        more += 40
        msgs = build(more)
    srcs = {rel: sha16(text), "dir:" + d: sha16(dir_listing(root, d)),
            "grep:" + fn["name"]: sha16(grep_dir(root, d, rel, fn["name"])),
            "grep:" + other["name"]: sha16(grep_dir(root, d, rel, other["name"], 10))}
    return msgs, {"function": fn["name"], "other_function": other["name"], "n_turns": len(plan), "plan": plan, "lang": lang, "file": rel,
                  "sources": srcs}


def _kind_order(i: int) -> list[str]:
    return EDIT_KINDS[i % len(EDIT_KINDS):] + EDIT_KINDS[:i % len(EDIT_KINDS)]


def generate_longctx(root=R2A_ROOT, edit_files=None, trace_files=None, seed: int = R2A_SEED) -> list[dict]:
    """The R2a prompt rows: 20 edit_full_file, 20 edit_diff, 20 toolcall_trace, ids r000-r059. Raises when a pinned file is
    missing or unusable (non ASCII, no usable function) so a broken tree never gives a short set."""
    edit_files = R2A_EDIT_FILES if edit_files is None else edit_files
    trace_files = R2A_TRACE_FILES if trace_files is None else trace_files
    root = pathlib.Path(root)
    rows: list[dict] = []
    sized = []
    for rel in edit_files:
        text = read_source(root, rel)
        if text is None:
            raise RuntimeError(f"R2a edit file missing or not plain ASCII: {rel}")
        sized.append((len(text), rel, text))
    sized.sort()
    by_variant = {"full_file": [], "diff": []}
    for i, item in enumerate(sized):              # alternate by size so both variants span the same sizes
        by_variant["full_file" if i % 2 == 0 else "diff"].append(item)
    for variant, cat in (("full_file", "edit_full_file"), ("diff", "edit_diff")):
        for i, (n, rel, text) in enumerate(by_variant[variant]):
            rng = random.Random(f"{seed}:{rel}")
            res = None
            for kind in _kind_order(i):
                res = edit_prompt(rel, text, kind, variant, lang_of(rel), rng)
                if res:
                    break
            if not res:
                raise RuntimeError(f"R2a: no edit instruction fits {rel}")
            msgs, meta = res
            chars = messages_chars(msgs)
            rows.append({"id": "", "category": cat, "n_predict": R2A_N_PREDICT[cat], "approx_tokens": approx_tokens(chars), "chars": chars,
                         "messages": msgs, "tools": None, "meta": dict(meta, file=rel, file_lines=len(text.split("\n")), sources={rel: sha16(text)})})
    for i, rel in enumerate(trace_files):
        text = read_source(root, rel)
        if text is None:
            raise RuntimeError(f"R2a trace file missing or not plain ASCII: {rel}")
        res = trace_prompt(root, rel, text, TRACE_TURNS[(i * 7) % len(TRACE_TURNS)], random.Random(f"{seed}:t:{rel}"))
        if not res:
            raise RuntimeError(f"R2a: no usable functions in trace file {rel}")
        msgs, meta = res
        chars = messages_chars(msgs, TRACE_TOOLS)
        rows.append({"id": "", "category": "toolcall_trace", "n_predict": R2A_N_PREDICT["toolcall_trace"], "approx_tokens": approx_tokens(chars),
                     "chars": chars, "messages": msgs, "tools": TRACE_TOOLS, "meta": meta})
    for n, r in enumerate(rows):
        r["id"] = "r%03d" % n
    return rows


def longctx_stats(rows: list[dict]) -> dict:
    """Counts and approximate prompt tokens per category, edit kinds, source files."""
    def q(xs, p):
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else None
    out: dict = {"n": len(rows), "by_category": {}, "chars_per_token": CHARS_PER_TOKEN}
    for cat in sorted({r["category"] for r in rows}):
        toks = [r["approx_tokens"] for r in rows if r["category"] == cat]
        out["by_category"][cat] = {"n": len(toks), "tokens_min": min(toks), "tokens_median": int(statistics.median(toks)),
                                   "tokens_p90": q(toks, 0.9), "tokens_max": max(toks), "tokens_sum": sum(toks),
                                   "n_predict": R2A_N_PREDICT.get(cat)}
    toks_all = [r["approx_tokens"] for r in rows]
    out["tokens_all"] = {"min": min(toks_all), "median": int(statistics.median(toks_all)), "max": max(toks_all), "sum": sum(toks_all)} if toks_all else {}
    kinds: dict[str, int] = {}
    for r in rows:
        k = r["meta"].get("kind")
        if k:
            kinds[k] = kinds.get(k, 0) + 1
    out["edit_kinds"] = kinds
    offs = [r["meta"]["edit_offset_tokens"] for r in rows if r["meta"].get("edit_offset_tokens") is not None]
    out["full_file_edit_offset_tokens"] = {"median": int(statistics.median(offs)), "max": max(offs)} if offs else None
    out["trace_turns"] = {str(t): sum(1 for r in rows if r["meta"].get("n_turns") == t) for t in (3, 4, 5)}
    out["files"] = len({s for r in rows for s in r["meta"].get("sources", {}) if not s.startswith(("dir:", "grep:"))})
    out["languages"] = {lg: sum(1 for r in rows if r["meta"].get("lang") == lg) for lg in sorted({r["meta"].get("lang") for r in rows if r["meta"].get("lang")})}
    out["dirs"] = {}
    for r in rows:
        f = r["meta"].get("file")
        if f:
            top = "/".join(f.split("/")[:3] if f.startswith(("ggml/", "experiments/")) else f.split("/")[:1])
            out["dirs"][top] = out["dirs"].get(top, 0) + 1
    return out


def drift_rows(committed: list[dict], fresh: list[dict]) -> dict:
    """Compare a regenerated set with the committed one. A row whose recorded sources differ is drift (a source file changed since the
    set was built); any other difference is a real mismatch. Returns {"drift": [ids], "mismatch": [ids]}."""
    out: dict = {"drift": [], "mismatch": []}
    if [r["id"] for r in committed] != [r["id"] for r in fresh]:
        out["mismatch"] = ["ids"]
        return out
    for a, b in zip(committed, fresh):
        if a == b:
            continue
        (out["drift"] if a["meta"].get("sources") != b["meta"].get("sources") else out["mismatch"]).append(a["id"])
    return out


def longctx_to_text(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows)


def load_longctx(path=R2A_OUT, limit: int | None = None) -> list[dict]:
    rows = [json.loads(ln) for ln in pathlib.Path(path).read_text().splitlines() if ln.strip()]
    return rows[:limit] if limit else rows


def select_files(root=R2A_ROOT, seed: int = R2A_SEED) -> dict:
    """How the pinned lists were made (kept for the record, not used at run time): tracked ASCII-only sources of 7-28K chars in
    src, common, tools, ggml/src/ggml-cuda (and a few stable ones of experiments/), at least three usable functions; seeded sample."""
    import subprocess
    tracked = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    quota = {"src": 9, "common": 8, "tools": 8, "ggml/src/ggml-cuda": 9, "experiments/cache-resident/harness": 3,
             "experiments/cache-resident/kernels": 3}
    skip = ("/tests/", "/webui/", "template-instances", "/vendor/", "/results/", "/data/", "/infra/")   # infra/ handles secrets
    rng = random.Random(seed)
    edit, used = [], set()
    for d, n in quota.items():
        cand = []
        for rel in sorted(tracked):
            if not rel.startswith(d + "/") or not rel.endswith(CODE_EXTS) or any(s in rel for s in skip + ("/models/",)):
                continue
            t = read_source(root, rel)
            if t is None or not (EDIT_TOKENS[0] * CHARS_PER_TOKEN <= len(t) <= EDIT_TOKENS[1] * CHARS_PER_TOKEN):
                continue
            if len([f for f in find_functions(t, lang_of(rel)) if 6 <= f["n_body"] <= 140]) >= 3:
                cand.append(rel)
        pick = rng.sample(cand, min(n, len(cand)))
        edit += pick
        used.update(pick)
    trace = []
    for d in ("src", "common", "tools", "ggml/src/ggml-cuda"):
        cand = []
        for rel in sorted(tracked):
            if not rel.startswith(d + "/") or rel in used or not rel.endswith(CODE_EXTS) or any(s in rel for s in skip):
                continue
            t = read_source(root, rel)
            if t is None or len(t) < 6000:
                continue
            if len([f for f in find_functions(t, lang_of(rel)) if f["calls"] >= 1 and f["n_body"] >= 6 and len(f["name"]) >= 5]) >= 3:
                cand.append(rel)
        trace += rng.sample(cand, 5)
    return {"edit": edit, "trace": trace}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="exps.make_prompts", description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="fail if the file on disk differs from a fresh generation")
    ap.add_argument("--longctx", action="store_true", help="the R2a long-context set instead of the E5 set")
    ap.add_argument("--stats", action="store_true", help="with --longctx: print the stats of the file on disk (or of --out)")
    a = ap.parse_args(argv)
    if a.longctx:
        out = pathlib.Path(a.out or R2A_OUT)
        if a.stats:
            print(json.dumps(longctx_stats(load_longctx(out)), indent=1))
            return 0
        fresh = generate_longctx()
        if a.check:
            if not out.exists():
                print("MISSING")
                return 1
            d = drift_rows(load_longctx(out), fresh)
            print("ok" if not d["mismatch"] else "MISMATCH", json.dumps(d))
            return 0 if not d["mismatch"] else 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(longctx_to_text(fresh))
        print(f"wrote {out} ({len(fresh)} prompts)")
        return 0
    text = to_text(generate())
    out = pathlib.Path(a.out or DEFAULT_OUT)
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
