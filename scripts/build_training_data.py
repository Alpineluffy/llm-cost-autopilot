#!/usr/bin/env python3
"""Build a length-balanced training set for the complexity classifier.

The original 180-prompt set had tier encoded in prompt length: tier 1 topped
out at 63 characters, tier 3 started at 191, and a single length threshold
scored 93%. The classifier learned "long = complex" and reported 100% accuracy
against a test split drawn from the same skew.

This rebuilds the corpus so length carries no tier signal. Every tier spans
short, medium, and long prompts, and the bulk that makes a prompt long —
pasted code, a traceback, a requirements list, a rambling preamble — is drawn
from the same tier-agnostic pool for all three tiers.

    python scripts/build_training_data.py
    python scripts/build_training_data.py --check   # validate, write nothing
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as stats
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_TRAIN = PROJECT_ROOT / "data" / "training_prompts.json"
OUT_EVAL = PROJECT_ROOT / "data" / "eval_length_adversarial.json"

SEED = 20260916


# ── Tier 1: simple — reformatting, lookup, syntax, boilerplate ──────────

TIER1 = [
    ("What is a variable in Python?", "basic_qa"),
    ("What does len() do?", "basic_qa"),
    ("How do I declare a constant in Java?", "syntax"),
    ("What is the difference between = and == ?", "basic_qa"),
    ("Print hello world in Go.", "simple_code"),
    ("How do I import a module in Python?", "syntax"),
    ("What is a list?", "basic_qa"),
    ("Convert this string to uppercase.", "simple_code"),
    ("Rename this variable to userCount.", "refactor"),
    ("What does the % operator do?", "basic_qa"),
    ("How do I comment out a line in SQL?", "syntax"),
    ("Format this JSON.", "format"),
    ("What is a for loop?", "basic_qa"),
    ("Write a function that adds two numbers.", "simple_code"),
    ("Fix the typo in this variable name.", "syntax_fix"),
    ("What is an integer?", "basic_qa"),
    ("How do I create an empty dictionary?", "syntax"),
    ("Convert this list to a set.", "simple_code"),
    ("What does return do?", "basic_qa"),
    ("Give me the boilerplate for a Python script.", "boilerplate"),
    ("How do I check the Python version?", "basic_qa"),
    ("What is a string?", "basic_qa"),
    ("Capitalize the first letter of this word.", "simple_code"),
    ("What does += mean?", "basic_qa"),
    ("How do I write a single line if statement?", "syntax"),
    ("Make this variable name snake_case.", "refactor"),
    ("What is None in Python?", "basic_qa"),
    ("Write an empty class definition.", "boilerplate"),
    ("How do I concatenate two strings?", "syntax"),
    ("What does append() do?", "basic_qa"),
    ("Turn this into a one-line list comprehension.", "simple_code"),
    ("What is a boolean?", "basic_qa"),
    ("How do I get the length of an array in JavaScript?", "syntax"),
    ("Add a docstring to this function.", "format"),
    ("What does break do in a loop?", "basic_qa"),
    ("Write the shebang line for a bash script.", "boilerplate"),
    ("How do I round a number to 2 decimals?", "syntax"),
    ("What is a tuple?", "basic_qa"),
    ("Sort this list alphabetically.", "simple_code"),
    ("What does strip() do?", "basic_qa"),
    ("Change these double quotes to single quotes.", "format"),
    ("How do I declare an array in C?", "syntax"),
    ("What is a function parameter?", "basic_qa"),
    ("Write a gitignore entry for node_modules.", "boilerplate"),
    ("Extract the numbers from this string.", "extraction"),
    ("What is a float?", "basic_qa"),
    ("How do I reverse a string?", "syntax"),
    ("What does continue do in a loop?", "basic_qa"),
    ("Convert this dictionary to JSON.", "simple_code"),
    ("How do I read a file line by line?", "syntax"),
    ("What is an index in a list?", "basic_qa"),
    ("Write a function that returns the max of two numbers.", "simple_code"),
    ("What does upper() do?", "basic_qa"),
    ("How do I check if a key exists in a dict?", "syntax"),
    ("Indent this code block properly.", "format"),
    ("What is a nested list?", "basic_qa"),
    ("Write the import line for datetime.", "boilerplate"),
    ("How do I join a list into a string?", "syntax"),
    ("What does int() do to a string?", "basic_qa"),
    ("Replace all tabs with spaces here.", "format"),
    ("What is a keyword argument?", "basic_qa"),
    ("How do I get today's date?", "syntax"),
    ("Extract all email addresses from this text.", "extraction"),
    ("What does pop() do on a list?", "basic_qa"),
    ("Write an empty try/except block.", "boilerplate"),
    ("How do I make a number negative?", "syntax"),
    ("What is a comment?", "basic_qa"),
    ("Turn this snake_case name into camelCase.", "refactor"),
    ("What does split() return?", "basic_qa"),
    ("How do I check if a string is empty?", "syntax"),
    ("List the keys of this dictionary.", "extraction"),
    ("What is a module?", "basic_qa"),
    ("Write a print statement with an f-string.", "simple_code"),
    ("How do I exit a Python script early?", "syntax"),
    ("What does abs() do?", "basic_qa"),
]


# ── Tier 2: moderate — review, debug, explain, compare, summarize ───────

TIER2 = [
    ("Why is my loop running one extra time?", "debugging"),
    ("Review this function for readability.", "code_review"),
    ("Summarize what this module does.", "summarization"),
    ("Explain the difference between a list and a tuple.", "explanation"),
    ("Debug this off-by-one error.", "debugging"),
    ("What are the pros and cons of using an ORM?", "comparison"),
    ("Refactor this to remove the nested ifs.", "refactoring"),
    ("Walk me through what this regex matches.", "explanation"),
    ("Why does this return None instead of a value?", "debugging"),
    ("Compare REST and GraphQL for a small app.", "comparison"),
    ("Write unit tests for this function.", "testing"),
    ("Explain how Python decorators work.", "explanation"),
    ("Is this function doing too much? Review it.", "code_review"),
    ("Trace what this recursive call returns.", "trace"),
    ("When should I use a set instead of a list?", "comparison"),
    ("Why am I getting a KeyError here?", "debugging"),
    ("Summarize the changes in this diff.", "summarization"),
    ("Explain what a context manager is and when to use one.", "explanation"),
    ("Improve the error handling in this function.", "code_review"),
    ("What edge cases am I missing in this validation?", "code_review"),
    ("Explain how garbage collection works in Python.", "explanation"),
    ("Why is this test flaky?", "debugging"),
    ("Convert this callback code to async/await.", "refactoring"),
    ("What's the difference between a process and a thread?", "explanation"),
    ("Review this SQL query for correctness.", "code_review"),
    ("Explain why this raises a TypeError.", "debugging"),
    ("Suggest better names for these variables.", "code_review"),
    ("Compare pytest and unittest.", "comparison"),
    ("Summarize this error log.", "summarization"),
    ("Explain what this list comprehension does step by step.", "explanation"),
    ("Why is my API returning 500 instead of 404?", "debugging"),
    ("Add type hints to this function.", "refactoring"),
    ("What does this stack trace tell me?", "debugging"),
    ("Explain the difference between deep and shallow copy.", "explanation"),
    ("Review this for SQL injection risk.", "code_review"),
    ("When is a dataclass better than a dict?", "comparison"),
    ("Why does this dictionary lose its ordering?", "debugging"),
    ("Explain what idempotent means for an HTTP endpoint.", "explanation"),
    ("Summarize the tradeoffs in this design doc.", "summarization"),
    ("Rewrite this loop to be more readable.", "refactoring"),
    ("What's wrong with catching a bare except?", "explanation"),
    ("Check this function for off-by-one errors.", "code_review"),
    ("Explain how JWT authentication works.", "explanation"),
    ("Why is this query slower than expected?", "debugging"),
    ("Compare Docker volumes and bind mounts.", "comparison"),
    ("Why does this function mutate its input?", "debugging"),
    ("Explain what a memory leak looks like in Python.", "explanation"),
    ("Review this class for single responsibility.", "code_review"),
    ("Summarize what this migration script changes.", "summarization"),
    ("Why is my regex matching too much?", "debugging"),
    ("Compare synchronous and asynchronous I/O.", "comparison"),
    ("Explain what happens when I call super().", "explanation"),
    ("Refactor these three near-identical functions.", "refactoring"),
    ("What's causing this circular import?", "debugging"),
    ("Review this API response schema.", "code_review"),
    ("Explain the difference between authentication and authorization.", "explanation"),
    ("Why does this return a copy instead of a reference?", "debugging"),
    ("Summarize the failure modes in this runbook.", "summarization"),
    ("Compare a queue and a stack for this use case.", "comparison"),
    ("Explain what a race condition is with an example.", "explanation"),
    ("Review this retry logic.", "code_review"),
    ("Why is this container restarting?", "debugging"),
    ("Add logging to this function in the right places.", "refactoring"),
    ("Explain how indexes speed up a database query.", "explanation"),
    ("What's the bug in this sorting comparison?", "debugging"),
    ("Compare environment variables and a config file.", "comparison"),
    ("Explain what a transaction isolation level does.", "explanation"),
    ("Review this for missing null checks.", "code_review"),
    ("Why does this date parse wrong in another timezone?", "debugging"),
    ("Summarize what this class is responsible for.", "summarization"),
    ("Explain when to use composition over inheritance.", "explanation"),
    ("Refactor this to remove the global variable.", "refactoring"),
    ("Why is this cache never hitting?", "debugging"),
    ("Explain what connection pooling does.", "explanation"),
    ("Review this function's parameter list.", "code_review"),
]


# ── Tier 3: complex — architecture, algorithms, multi-step reasoning ────
# Deliberately includes very short prompts. "Implement Raft." is four words
# and genuinely tier 3; the old corpus had nothing like it.

TIER3 = [
    ("Implement Raft consensus.", "algorithm"),
    ("Design a CRDT for collaborative text editing.", "architecture"),
    ("Build a lock-free queue.", "concurrency"),
    ("Design a distributed rate limiter.", "system_design"),
    ("Implement a B-tree with deletion.", "algorithm"),
    ("Architect a multi-tenant SaaS database schema.", "architecture"),
    ("Write a garbage collector.", "algorithm"),
    ("Design an event sourcing system.", "architecture"),
    ("Implement consistent hashing with virtual nodes.", "algorithm"),
    ("Prove this algorithm is O(n log n).", "algorithm"),
    ("Design a job scheduler with priorities and retries.", "system_design"),
    ("Implement a query planner.", "algorithm"),
    ("Build a connection pool with backpressure.", "concurrency"),
    ("Design an idempotent payment pipeline.", "architecture"),
    ("Implement a trie with fuzzy matching.", "algorithm"),
    ("Architect a real-time collaborative editor.", "architecture"),
    ("Write a dependency resolver that detects cycles.", "algorithm"),
    ("Design a sharded counter that survives partitions.", "system_design"),
    ("Implement optimistic concurrency control.", "concurrency"),
    ("Design a feature flag system with gradual rollout.", "architecture"),
    ("Build a retry policy with exponential backoff and jitter.", "system_design"),
    ("Implement a streaming median over a sliding window.", "algorithm"),
    ("Design a zero-downtime schema migration strategy.", "architecture"),
    ("Write a thread-safe LRU cache.", "concurrency"),
    ("Design an audit log that cannot be tampered with.", "security"),
    ("Implement backpressure across a async pipeline.", "concurrency"),
    ("Architect a service mesh migration.", "devops"),
    ("Design a recommendation system cold-start strategy.", "machine_learning"),
    ("Implement a distributed lock with fencing tokens.", "concurrency"),
    ("Design a multi-region failover strategy.", "system_design"),
    ("Build an incremental compiler.", "algorithm"),
    ("Design a deduplicating backup system.", "architecture"),
    ("Implement a rate-limited crawler that respects robots.txt.", "system_design"),
    ("Design a time-series store with downsampling.", "architecture"),
    ("Write a scheduler that avoids priority inversion.", "concurrency"),
    ("Design an OAuth token refresh flow that handles races.", "security"),
    ("Implement copy-on-write for a persistent data structure.", "algorithm"),
    ("Architect a data pipeline with exactly-once delivery.", "architecture"),
    ("Design a circuit breaker for a fan-out service.", "system_design"),
    ("Implement a parser with error recovery.", "algorithm"),
    ("Design a leader election protocol.", "system_design"),
    ("Implement a bloom filter with counting.", "algorithm"),
    ("Architect a read-heavy service with cache invalidation.", "architecture"),
    ("Write a work-stealing scheduler.", "concurrency"),
    ("Design a schema for hierarchical permissions.", "architecture"),
    ("Implement a merge algorithm for concurrent edits.", "algorithm"),
    ("Design a webhook delivery system with ordering guarantees.", "system_design"),
    ("Build a memory allocator.", "algorithm"),
    ("Design an anomaly detection pipeline.", "machine_learning"),
    ("Implement a skip list.", "algorithm"),
    ("Architect a blue-green deployment with database changes.", "devops"),
    ("Design a secret rotation system with zero downtime.", "security"),
    ("Implement a topological sort with cycle reporting.", "algorithm"),
    ("Design a multi-level cache with write-through semantics.", "architecture"),
    ("Build a rate limiter using the token bucket algorithm.", "system_design"),
    ("Implement a red-black tree.", "algorithm"),
    ("Design a saga pattern for distributed transactions.", "architecture"),
    ("Write a load balancer with health checking.", "system_design"),
    ("Implement a concurrent hash map with striped locking.", "concurrency"),
    ("Design a versioned document store with conflict resolution.", "architecture"),
    ("Build a diff algorithm for structured data.", "algorithm"),
    ("Design a tenant isolation strategy for shared infrastructure.", "security"),
    ("Implement a priority queue with decrease-key.", "algorithm"),
    ("Architect a migration from a monolith to services.", "architecture"),
    ("Design a replication protocol tolerant of network partitions.", "system_design"),
    ("Implement a bytecode interpreter.", "algorithm"),
    ("Design a fair queuing system across noisy tenants.", "system_design"),
    ("Build a static analyzer that detects unused code paths.", "algorithm"),
    ("Design an encryption-at-rest scheme with key rotation.", "security"),
    ("Implement a coroutine scheduler.", "concurrency"),
]


# ── Tier-agnostic bulk ──────────────────────────────────────────────────
# Applied across all three tiers so that "long" never implies "complex".

PREAMBLES = [
    "I'm still pretty new to all of this and I've been going back and forth "
    "reading different tutorials, and honestly a lot of them contradict each "
    "other, so I wanted to just ask directly.",

    "My team lead mentioned this in standup yesterday and I nodded along, but "
    "I want to actually understand it before the next sprint planning meeting.",

    "I've been staring at this for about two hours now and I think I'm too "
    "deep in it to see the obvious answer, so a fresh pair of eyes would help.",

    "Context: this is for a side project I'm building on weekends, nothing "
    "production critical, but I'd still like to do it properly rather than "
    "hacking something together that I'll regret later.",

    "We're doing a code review tomorrow and I want to get ahead of the "
    "feedback rather than having the whole thing torn apart in the meeting.",

    "I searched for this and found three different answers on Stack Overflow, "
    "all from different years, and I can't tell which one still applies.",

    "This is part of a larger refactor we've been putting off for months, and "
    "I finally have a week of runway to actually deal with it properly.",

    "Quick bit of background: the codebase is about four years old, has been "
    "through two rewrites, and the original authors have all left the company.",

    "I'm preparing for a technical interview next week and this came up in a "
    "practice problem that I completely fumbled.",

    "We inherited this service from another team during a reorg and the "
    "documentation is, generously, a single README with a TODO in it.",
]

CODE_CONTEXT = [
    "Here's the relevant code:\n\n```python\ndef process(items):\n"
    "    results = []\n    for i in range(len(items)):\n"
    "        if items[i] is not None:\n            results.append(items[i])\n"
    "    return results\n```",

    "The function in question:\n\n```javascript\nasync function fetchAll(ids) {\n"
    "  const out = [];\n  for (const id of ids) {\n"
    "    out.push(await fetch(`/api/${id}`));\n  }\n  return out;\n}\n```",

    "This is what I have so far:\n\n```python\nclass Cache:\n"
    "    def __init__(self):\n        self.store = {}\n\n"
    "    def get(self, k):\n        return self.store.get(k)\n```",

    "Current implementation:\n\n```sql\nSELECT u.*, o.total\nFROM users u\n"
    "LEFT JOIN orders o ON o.user_id = u.id\nWHERE u.created_at > '2024-01-01'\n"
    "ORDER BY o.total DESC;\n```",

    "The snippet:\n\n```go\nfunc worker(jobs <-chan int, out chan<- int) {\n"
    "    for j := range jobs {\n        out <- j * 2\n    }\n}\n```",
]

ERROR_CONTEXT = [
    "The traceback I'm getting:\n\nTraceback (most recent call last):\n"
    '  File "app.py", line 42, in <module>\n    main()\n'
    '  File "app.py", line 31, in main\n    result = process(data)\n'
    "TypeError: 'NoneType' object is not iterable",

    "Logs from the failure:\n\n2026-03-04 11:22:01 ERROR  request failed "
    "status=500 trace_id=8f2a upstream=payments latency_ms=30012\n"
    "2026-03-04 11:22:01 ERROR  context deadline exceeded",

    "It fails intermittently with:\n\nConnectionResetError: [Errno 104] "
    "Connection reset by peer\n  during handshake with pool worker 3",
]

REQUIREMENTS = [
    "Constraints: must stay backward compatible with the existing API, cannot "
    "add new third-party dependencies, and has to pass the existing test suite "
    "without modification.",

    "Requirements: keep it readable for junior engineers, no clever one-liners, "
    "and add comments only where the intent isn't obvious from the code.",

    "A few things to keep in mind: we're on an older runtime version, the CI "
    "budget is tight so nothing that doubles the test time, and the team "
    "prefers explicit over implicit.",

    "Please keep the answer practical — we ship on Friday and I need something "
    "I can actually apply, not a theoretical overview of the whole problem space.",
]


def _rng() -> random.Random:
    return random.Random(SEED)


def _build_tier(
    cores: list[tuple[str, str]], tier: int, rng: random.Random
) -> list[dict]:
    """Expand a tier's cores across short, medium, and long length bands.

    Each core appears once. Which band it lands in is assigned round-robin so
    every tier ends up with roughly a third of its prompts in each band.
    """
    entries = []
    for i, (core, category) in enumerate(cores):
        band = i % 3

        if band == 0:                       # short — the bare core
            prompt = core
        elif band == 1:                     # medium — one piece of context
            extra = rng.choice(PREAMBLES + REQUIREMENTS)
            prompt = f"{extra}\n\n{core}"
        else:                               # long — several pieces
            parts = [
                rng.choice(PREAMBLES),
                rng.choice(CODE_CONTEXT + ERROR_CONTEXT),
                core,
                rng.choice(REQUIREMENTS),
            ]
            prompt = "\n\n".join(parts)

        entries.append(
            {
                "prompt": prompt,
                "tier": tier,
                "category": category,
                "length_band": ("short", "medium", "long")[band],
            }
        )
    return entries


def build() -> list[dict]:
    rng = _rng()
    data = (
        _build_tier(TIER1, 1, rng)
        + _build_tier(TIER2, 2, rng)
        + _build_tier(TIER3, 3, rng)
    )
    rng.shuffle(data)
    return data


def build_adversarial_eval() -> list[dict]:
    """Held-out prompts where length actively points the wrong way.

    Short prompts that are genuinely complex, long prompts that are genuinely
    trivial. A length-shortcut classifier fails this set badly; a classifier
    that learned the actual signal does not.
    """
    rng = random.Random(SEED + 1)
    short_complex = [
        ("Implement Paxos.", 3, "algorithm"),
        ("Design a CRDT.", 3, "architecture"),
        ("Write a JIT compiler.", 3, "algorithm"),
        ("Build a distributed tracer.", 3, "system_design"),
        ("Implement MVCC.", 3, "concurrency"),
        ("Design a sharding strategy.", 3, "architecture"),
        ("Write a regex engine.", 3, "algorithm"),
        ("Implement two-phase commit.", 3, "concurrency"),
        ("Design a quorum read protocol.", 3, "system_design"),
        ("Build an LSM tree.", 3, "algorithm"),
    ]

    long_simple = []
    for core, category in [
        ("What is a variable?", "basic_qa"),
        ("How do I import a module?", "syntax"),
        ("What does len() return?", "basic_qa"),
        ("Print hello world.", "simple_code"),
        ("What is a dictionary?", "basic_qa"),
        ("How do I write a comment?", "syntax"),
        ("What does += do?", "basic_qa"),
        ("Make this variable name lowercase.", "refactor"),
        ("What is a boolean?", "basic_qa"),
        ("How do I check a list's length?", "syntax"),
    ]:
        padded = "\n\n".join(
            [
                rng.choice(PREAMBLES),
                rng.choice(CODE_CONTEXT),
                core,
                rng.choice(REQUIREMENTS),
            ]
        )
        long_simple.append((padded, 1, category))

    return [
        {"prompt": p, "tier": t, "category": c, "adversarial": True}
        for p, t, c in short_complex + long_simple
    ]


# ── Validation ──────────────────────────────────────────────────────────

def validate(data: list[dict]) -> bool:
    """Assert that length no longer separates the tiers."""
    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    ok = True
    print("Length distribution by tier:")
    print(f"  {'tier':<6}{'n':>5}{'min':>7}{'med':>7}{'mean':>8}{'max':>7}")
    spans = {}
    for tier in (1, 2, 3):
        lengths = [len(d["prompt"]) for d in data if d["tier"] == tier]
        spans[tier] = (min(lengths), max(lengths))
        print(
            f"  {tier:<6}{len(lengths):>5}{min(lengths):>7}"
            f"{int(stats.median(lengths)):>7}{stats.mean(lengths):>8.0f}"
            f"{max(lengths):>7}"
        )

    # Every pair of tiers must overlap in length.
    for a, b in ((1, 2), (2, 3), (1, 3)):
        lo = max(spans[a][0], spans[b][0])
        hi = min(spans[a][1], spans[b][1])
        if lo >= hi:
            print(f"  FAIL: tier {a} and tier {b} lengths do not overlap")
            ok = False
    if ok:
        print("  OK: all tier pairs overlap in length")

    # A length-only model must be near chance (33%).
    X = np.array([[len(d["prompt"])] for d in data])
    y = np.array([d["tier"] for d in data])
    baseline = cross_val_score(
        DecisionTreeClassifier(max_depth=3, random_state=0), X, y, cv=5
    ).mean()
    print(f"\n  Length-only baseline: {baseline:.1%} (chance is 33.3%)")
    if baseline > 0.50:
        print("  FAIL: length alone still predicts tier too well")
        ok = False
    else:
        print("  OK: length alone carries little tier signal")

    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="Validate only, write nothing"
    )
    args = parser.parse_args()

    data = build()
    print(f"Built {len(data)} training prompts\n")
    ok = validate(data)

    if args.check:
        return 0 if ok else 1

    if not ok:
        print("\nRefusing to write an unbalanced corpus.")
        return 1

    OUT_TRAIN.write_text(json.dumps(data, indent=2))
    print(f"\nWrote {len(data)} → {OUT_TRAIN.relative_to(PROJECT_ROOT)}")

    adversarial = build_adversarial_eval()
    OUT_EVAL.write_text(json.dumps(adversarial, indent=2))
    print(f"Wrote {len(adversarial)} → {OUT_EVAL.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
