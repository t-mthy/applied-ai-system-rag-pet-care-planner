"""
Guardrails Tests
────────────────
Unit tests for the Phase 6 reliability layer:
  - validate_pet_profile / validate_query   (input validation)
  - apply_output_guardrail                  (citation backstop)
  - compute_confidence                      (display-time scoring)
  - configure_logger / log_event            (JSON-lines logger)

Plus a small integration test confirming the planner emits the
expected structured events when it runs end-to-end.

Run with:  pytest tests/test_guardrails.py -v
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.generator import Suggestion
from src.guardrails import (
    ALLOWED_SPECIES,
    GuardResult,
    MAX_AGE,
    MAX_QUERY_LEN,
    MIN_AGE,
    ValidationResult,
    apply_output_guardrail,
    compute_confidence,
    configure_logger,
    log_event,
    validate_pet_profile,
    validate_query,
)
from src.retriever import KBChunk, RetrievalResult


# ──────────────────────────────────────────────────────────────────
# Helper: minimal Suggestion / RetrievalResult builders
# ──────────────────────────────────────────────────────────────────
def _make_chunk(id_: str = "dog_puppy_exercise") -> KBChunk:
    return KBChunk(
        id=id_,
        species="dog", life_stage="puppy", topic="exercise",
        tags=("puppy",),
        source="AKC (illustrative)",
        source_url="https://www.akc.org/expert-advice/",
        retrieved_on="2026-04-26",
        task_templates=({"description": "Walk", "duration_minutes": 15,
                         "frequency": "daily", "priority": "high",
                         "suggested_time": "08:00"},),
        body="Body.",
    )


def _make_suggestion(source_id: str = "dog_puppy_exercise",
                     retrieval_score: float = 0.4) -> Suggestion:
    return Suggestion(
        description="Walk", duration_minutes=15,
        frequency="daily", priority="high", suggested_time="08:00",
        source_id=source_id,
        source="AKC (illustrative)",
        source_url="https://www.akc.org/expert-advice/",
        retrieval_score=retrieval_score,
        rationale=f"grounded in {source_id}",
    )


# ──────────────────────────────────────────────────────────────────
# validate_pet_profile
# ──────────────────────────────────────────────────────────────────
def test_validate_pet_profile_accepts_supported_species_and_age():
    assert validate_pet_profile("dog", 3).valid
    assert validate_pet_profile("cat", 0).valid
    assert validate_pet_profile("rabbit", 5).valid


def test_validate_pet_profile_is_case_insensitive():
    assert validate_pet_profile("Dog", 3).valid
    assert validate_pet_profile("CAT", 1).valid
    assert validate_pet_profile("  rabbit ", 2).valid


def test_validate_pet_profile_rejects_unsupported_species():
    r = validate_pet_profile("hamster", 1)
    assert not r.valid
    assert any("hamster" in e for e in r.errors)


def test_validate_pet_profile_rejects_empty_species():
    r = validate_pet_profile("", 3)
    assert not r.valid
    assert any("empty" in e for e in r.errors)


def test_validate_pet_profile_rejects_none_species():
    r = validate_pet_profile(None, 3)  # type: ignore[arg-type]
    assert not r.valid


def test_validate_pet_profile_rejects_negative_age():
    r = validate_pet_profile("dog", -1)
    assert not r.valid
    assert any(">= " in e for e in r.errors)


def test_validate_pet_profile_rejects_implausibly_large_age():
    """The architecture's example: a 250-year-old cat."""
    r = validate_pet_profile("cat", 250)
    assert not r.valid
    assert any("<= " in e for e in r.errors)


def test_validate_pet_profile_rejects_non_int_age():
    r = validate_pet_profile("dog", "five")  # type: ignore[arg-type]
    assert not r.valid


def test_validate_pet_profile_rejects_bool_age():
    """Python's bool is a subclass of int — guard against True/False
    sneaking in as age=1 / age=0."""
    r = validate_pet_profile("dog", True)  # type: ignore[arg-type]
    assert not r.valid


def test_validate_pet_profile_age_boundaries():
    assert validate_pet_profile("dog", MIN_AGE).valid
    assert validate_pet_profile("dog", MAX_AGE).valid
    assert not validate_pet_profile("dog", MIN_AGE - 1).valid
    assert not validate_pet_profile("dog", MAX_AGE + 1).valid


def test_allowed_species_set_matches_kb():
    """Canary so we notice if the KB grows but the validator forgets."""
    assert ALLOWED_SPECIES == frozenset({"dog", "cat", "rabbit"})


# ──────────────────────────────────────────────────────────────────
# validate_query
# ──────────────────────────────────────────────────────────────────
def test_validate_query_none_is_valid():
    assert validate_query(None).valid


def test_validate_query_empty_is_valid():
    assert validate_query("").valid


def test_validate_query_short_is_valid():
    assert validate_query("feeding nutrition").valid


def test_validate_query_at_max_length_is_valid():
    assert validate_query("x" * MAX_QUERY_LEN).valid


def test_validate_query_over_max_length_is_invalid():
    r = validate_query("x" * (MAX_QUERY_LEN + 1))
    assert not r.valid


def test_validate_query_non_string_is_invalid():
    r = validate_query(12345)  # type: ignore[arg-type]
    assert not r.valid


# ──────────────────────────────────────────────────────────────────
# compute_confidence
# ──────────────────────────────────────────────────────────────────
def test_confidence_floor_is_half_when_score_is_zero():
    assert compute_confidence(0.0) == 0.5


def test_confidence_is_one_when_score_is_one():
    assert compute_confidence(1.0) == 1.0


def test_confidence_is_linear_in_score():
    """0.5 + 0.5 * score → 0.3 should map to 0.65."""
    assert compute_confidence(0.3) == pytest.approx(0.65)


def test_confidence_clamps_negative_score_to_floor():
    assert compute_confidence(-0.1) == 0.5


def test_confidence_clamps_oversized_score_to_one():
    assert compute_confidence(1.5) == 1.0


def test_confidence_is_monotonic():
    """Higher retrieval score → higher confidence, no surprises."""
    a = compute_confidence(0.1)
    b = compute_confidence(0.4)
    c = compute_confidence(0.8)
    assert a < b < c


# ──────────────────────────────────────────────────────────────────
# apply_output_guardrail
# ──────────────────────────────────────────────────────────────────
def test_guardrail_accepts_all_when_every_citation_matches():
    chunk = _make_chunk("dog_puppy_exercise")
    suggestions = [
        _make_suggestion("dog_puppy_exercise"),
        _make_suggestion("dog_puppy_exercise"),
    ]
    results = [RetrievalResult(chunk=chunk, score=0.4)]
    g = apply_output_guardrail(suggestions, results)
    assert len(g.accepted) == 2
    assert g.rejected == []


def test_guardrail_rejects_unknown_source_id():
    chunk = _make_chunk("dog_puppy_exercise")
    suggestions = [
        _make_suggestion("dog_puppy_exercise"),
        _make_suggestion("dog_puppy_FAKE"),
    ]
    results = [RetrievalResult(chunk=chunk, score=0.4)]
    g = apply_output_guardrail(suggestions, results)
    assert len(g.accepted) == 1
    assert len(g.rejected) == 1
    bad, reason = g.rejected[0]
    assert bad.source_id == "dog_puppy_FAKE"
    assert "dog_puppy_FAKE" in reason


def test_guardrail_with_empty_inputs_returns_empty_result():
    g = apply_output_guardrail([], [])
    assert g.accepted == []
    assert g.rejected == []


def test_guardrail_is_pure_does_not_mutate_input_suggestions():
    chunk = _make_chunk("dog_puppy_exercise")
    s = _make_suggestion("dog_puppy_exercise", retrieval_score=0.4)
    results = [RetrievalResult(chunk=chunk, score=0.4)]
    apply_output_guardrail([s], results)
    # Suggestion is frozen; assert all fields unchanged.
    assert s.source_id == "dog_puppy_exercise"
    assert s.retrieval_score == 0.4


# ──────────────────────────────────────────────────────────────────
# Structured logger
# ──────────────────────────────────────────────────────────────────
def test_log_event_writes_json_line(tmp_path: Path):
    log_file = tmp_path / "events.jsonl"
    configure_logger(path=log_file, enabled=True)

    log_event("retrieve", pet="Mochi", n_results=3)

    contents = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    record = json.loads(contents[0])
    assert record["event"] == "retrieve"
    assert record["pet"] == "Mochi"
    assert record["n_results"] == 3
    assert "ts" in record


def test_log_event_appends_multiple_lines(tmp_path: Path):
    log_file = tmp_path / "events.jsonl"
    configure_logger(path=log_file, enabled=True)

    log_event("a", x=1)
    log_event("b", y=2)
    log_event("c", z=3)

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    events = [json.loads(line)["event"] for line in lines]
    assert events == ["a", "b", "c"]


def test_log_event_disabled_does_not_write(tmp_path: Path):
    log_file = tmp_path / "events.jsonl"
    configure_logger(path=log_file, enabled=False)

    log_event("retrieve", pet="Mochi")

    assert not log_file.exists()


def test_log_event_creates_parent_directory(tmp_path: Path):
    """Lazy directory creation — if the parent doesn't exist yet, the
    logger should create it on first write rather than raise."""
    nested = tmp_path / "deep" / "nested" / "events.jsonl"
    configure_logger(path=nested, enabled=True)

    log_event("retrieve")

    assert nested.exists()


def test_log_event_serializes_unknown_types_via_default_str(tmp_path: Path):
    """`default=str` should keep us from blowing up on Path / etc."""
    log_file = tmp_path / "events.jsonl"
    configure_logger(path=log_file, enabled=True)

    log_event("retrieve", weird_path=Path("/tmp/x"))

    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert "weird_path" in record  # serialized as a string


# ──────────────────────────────────────────────────────────────────
# End-to-end: planner emits the expected structured events
# ──────────────────────────────────────────────────────────────────
def test_planner_emits_retrieve_generate_guard_events(tmp_path: Path):
    """Run the real planner and verify the JSON-lines log shows the
    expected sequence of events for a happy-path call."""
    from src.pawpal_system import Pet
    from src.rag_planner import suggest_tasks_for_pet
    from src.retriever import Retriever

    log_file = tmp_path / "events.jsonl"
    configure_logger(path=log_file, enabled=True)

    pet = Pet(name="Mochi", species="dog", age=3)
    suggestions = suggest_tasks_for_pet(pet, retriever=Retriever())
    assert suggestions  # the call should succeed

    events = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").strip().splitlines()
    ]
    event_names = [e["event"] for e in events]
    assert "retrieve" in event_names
    assert "generate" in event_names
    assert "guard" in event_names

    # Verify the guard event captured the accepted/rejected counts.
    guard = next(e for e in events if e["event"] == "guard")
    assert guard["n_accepted"] == len(suggestions)
    assert guard["n_rejected"] == 0  # Tier C never produces ungrounded output


def test_planner_emits_input_rejected_for_bad_species(tmp_path: Path):
    """An unsupported species should be rejected at the input stage."""
    from src.pawpal_system import Pet
    from src.rag_planner import suggest_tasks_for_pet
    from src.retriever import Retriever

    log_file = tmp_path / "events.jsonl"
    configure_logger(path=log_file, enabled=True)

    pet = Pet(name="Hammy", species="hamster", age=2)
    suggestions = suggest_tasks_for_pet(pet, retriever=Retriever())
    assert suggestions == []

    events = [
        json.loads(line)
        for line in log_file.read_text(encoding="utf-8").strip().splitlines()
    ]
    rejected = [e for e in events if e["event"] == "input.rejected"]
    assert rejected
    assert rejected[0]["stage"] == "pet_profile"


# ──────────────────────────────────────────────────────────────────
# Cleanup: re-point the logger at the default after tests so we
# don't leave a tmp_path stuck in the module-level state.
# ──────────────────────────────────────────────────────────────────
def teardown_module(module):
    configure_logger(path=None, enabled=True)
