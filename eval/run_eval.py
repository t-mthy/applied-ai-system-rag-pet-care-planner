"""
PawPal+ AI — Evaluation Harness
───────────────────────────────
Runs every case in `eval/cases.json` through the full RAG pipeline
(validate → retrieve → generate → guard) and asserts a list of
expected behaviors per case. Prints a per-case + total summary, and
exits non-zero if any check fails (so CI can pick this up).

Usage:
    python eval/run_eval.py            # run all cases
    python -m eval.run_eval            # equivalent

The runner is offline like every other module — no network, no API
keys, no model downloads. With a deterministic Tier C pipeline,
results are repeatable across runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

# Make the project's src/ importable when this file is invoked from
# anywhere (project root, eval/, or as `python -m eval.run_eval`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails import compute_confidence  # noqa: E402
from src.pawpal_system import Pet  # noqa: E402
from src.rag_planner import suggest_tasks_for_pet  # noqa: E402
from src.retriever import Retriever  # noqa: E402

# Where the cases live. Constant so users can `import eval.run_eval`
# and reuse the runner programmatically (e.g. in a notebook).
CASES_FILE = Path(__file__).resolve().parent / "cases.json"


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def _topic_of(source_id: str) -> str:
    """Pull the topic out of a chunk id like 'dog_puppy_exercise'.

    Returns the empty string for unexpectedly-shaped ids. Defensive
    only — the Phase 2 KB validator already enforces this shape.
    """
    parts = source_id.split("_")
    return parts[2] if len(parts) >= 3 else ""


# ──────────────────────────────────────────────────────────────────
# Check registry
# ──────────────────────────────────────────────────────────────────
# Each check is a function (suggestions, value) -> (passed, message).
# `value` carries whatever the cases.json declared for that check —
# an int, a string, a list, or a small dict, depending on the check.
def _check_min_suggestions(suggestions, value):
    n = len(suggestions)
    return n >= value, f"got {n}, expected >= {value}"


def _check_max_suggestions(suggestions, value):
    n = len(suggestions)
    return n <= value, f"got {n}, expected <= {value}"


def _check_all_have_citations(suggestions, _):
    if not suggestions:
        return False, "no suggestions to check citations on"
    for s in suggestions:
        if not (s.source_id and s.source and s.source_url):
            return False, (
                f"suggestion {s.description!r} missing one of "
                f"source_id / source / source_url"
            )
    return True, f"all {len(suggestions)} suggestions cite a source"


def _check_topic_present(suggestions, value):
    topics = {_topic_of(s.source_id) for s in suggestions}
    return value in topics, (
        f"topics found: {sorted(topics)}, expected {value!r}"
    )


def _check_topic_count_min(suggestions, spec):
    topic = spec["topic"]
    minimum = spec["min"]
    count = sum(1 for s in suggestions if _topic_of(s.source_id) == topic)
    return count >= minimum, (
        f"topic={topic!r}: got {count}, expected >= {minimum}"
    )


def _check_description_contains_any(suggestions, keywords):
    """At least one suggestion's description contains any of the keywords."""
    keywords_lc = [k.lower() for k in keywords]
    for s in suggestions:
        for k in keywords_lc:
            if k in s.description.lower():
                return True, f"matched {k!r} in {s.description!r}"
    return False, f"none of {keywords} found in any description"


def _check_max_duration_for_topic(suggestions, spec):
    topic = spec["topic"]
    max_dur = spec["max"]
    relevant = [s for s in suggestions if _topic_of(s.source_id) == topic]
    if not relevant:
        return False, f"no suggestions for topic={topic!r}"
    over = [s for s in relevant if s.duration_minutes > max_dur]
    if over:
        bad = ", ".join(
            f"{s.description}={s.duration_minutes}min" for s in over
        )
        return False, f"durations exceed {max_dur} min: {bad}"
    return True, (
        f"all {len(relevant)} {topic!r} task(s) <= {max_dur} min"
    )


def _check_expect_empty(suggestions, _):
    n = len(suggestions)
    return n == 0, f"expected empty, got {n} suggestion(s)"


def _check_min_confidence(suggestions, value):
    if not suggestions:
        return False, "no suggestions to check confidence on"
    bad = [
        (s, compute_confidence(s.retrieval_score))
        for s in suggestions
        if compute_confidence(s.retrieval_score) < value
    ]
    if bad:
        bad_desc = ", ".join(
            f"{s.description}={c:.2f}" for s, c in bad
        )
        return False, f"some confidences < {value}: {bad_desc}"
    return True, f"all {len(suggestions)} suggestion(s) >= {value:.2f}"


# Registry — referenced by `check` field in cases.json.
CHECKS: dict[str, Callable[[list, Any], tuple[bool, str]]] = {
    "min_suggestions": _check_min_suggestions,
    "max_suggestions": _check_max_suggestions,
    "all_have_citations": _check_all_have_citations,
    "topic_present": _check_topic_present,
    "topic_count_min": _check_topic_count_min,
    "description_contains_any": _check_description_contains_any,
    "max_duration_for_topic": _check_max_duration_for_topic,
    "expect_empty": _check_expect_empty,
    "min_confidence": _check_min_confidence,
}


# ──────────────────────────────────────────────────────────────────
# Per-case runner
# ──────────────────────────────────────────────────────────────────
def run_case(case: dict, retriever: Retriever) -> dict:
    """Run one case end-to-end and return a structured summary.

    Returned dict shape (also useful for programmatic callers):
      {
        "id": str,
        "description": str,
        "n_suggestions": int,
        "n_passed": int,
        "n_failed": int,
        "checks": [ {check, passed, message, rationale}, ... ],
      }
    """
    pet_spec = case["pet"]
    pet = Pet(
        name=pet_spec["name"],
        species=pet_spec["species"],
        age=pet_spec["age"],
    )
    query = case.get("query")

    # Run the full pipeline. The retriever is shared across cases for
    # speed — building the TF-IDF index once is enough.
    suggestions = suggest_tasks_for_pet(
        pet, query=query, retriever=retriever,
    )

    check_results = []
    n_passed = 0
    n_failed = 0
    for exp in case["expectations"]:
        check_name = exp["check"]
        value = exp.get("value")
        check_fn = CHECKS.get(check_name)
        if check_fn is None:
            check_results.append({
                "check": check_name,
                "passed": False,
                "message": f"unknown check {check_name!r}",
                "rationale": exp.get("rationale", ""),
            })
            n_failed += 1
            continue

        passed, message = check_fn(suggestions, value)
        check_results.append({
            "check": check_name,
            "passed": passed,
            "message": message,
            "rationale": exp.get("rationale", ""),
        })
        if passed:
            n_passed += 1
        else:
            n_failed += 1

    return {
        "id": case["id"],
        "description": case.get("description", ""),
        "n_suggestions": len(suggestions),
        "n_passed": n_passed,
        "n_failed": n_failed,
        "checks": check_results,
    }


# ──────────────────────────────────────────────────────────────────
# Top-level runner
# ──────────────────────────────────────────────────────────────────
def run_all() -> tuple[list[dict], int, int]:
    """Run every case in `cases.json`. Returns (per_case_results,
    total_passed, total_failed)."""
    with CASES_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"]

    retriever = Retriever()

    results: list[dict] = []
    total_passed = 0
    total_failed = 0
    for case in cases:
        case_result = run_case(case, retriever)
        results.append(case_result)
        total_passed += case_result["n_passed"]
        total_failed += case_result["n_failed"]

    return results, total_passed, total_failed


def _print_case(result: dict) -> None:
    """Pretty-print one case's outcome to stdout."""
    print("-" * 70)
    print(f"CASE: {result['id']}")
    if result["description"]:
        print(f"  {result['description']}")
    print(f"  Generated {result['n_suggestions']} suggestion(s)")
    print()
    for check in result["checks"]:
        marker = "[PASS]" if check["passed"] else "[FAIL]"
        print(f"    {marker} {check['check']}: {check['message']}")
        # Print rationale only on failures — keeps a passing run quiet.
        if not check["passed"] and check["rationale"]:
            print(f"           why this matters: {check['rationale']}")
    print()


def main() -> int:
    print("=" * 70)
    print("  PawPal+ AI — Evaluation Harness")
    print("=" * 70)

    results, total_passed, total_failed = run_all()
    print(f"\nRan {len(results)} case(s) from {CASES_FILE.name}\n")

    for r in results:
        _print_case(r)

    # ── Summary table ──
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["n_failed"] == 0 else "FAIL"
        total_checks = r["n_passed"] + r["n_failed"]
        print(
            f"  [{status}] {r['id']:<28} "
            f"{r['n_passed']}/{total_checks} checks "
            f"({r['n_suggestions']} suggestion(s))"
        )

    grand_total = total_passed + total_failed
    n_cases_passed = sum(1 for r in results if r["n_failed"] == 0)
    print()
    print(
        f"  Total: {total_passed}/{grand_total} checks passed "
        f"across {len(results)} case(s) "
        f"({n_cases_passed}/{len(results)} cases fully green)"
    )
    print("=" * 70)

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
