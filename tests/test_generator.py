"""
Generator Tests
───────────────
Unit tests for the Phase 4 Generator module. These verify that
template-based synthesis correctly fans chunks out into Suggestion
objects, attaches the right citations, and remains deterministic.

Run with:  pytest tests/test_generator.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retriever import KBChunk, RetrievalResult
from src.generator import Suggestion, generate_suggestions


# ──────────────────────────────────────────────────────────────────
# Helper: build a synthetic chunk without going through the KB on disk
# ──────────────────────────────────────────────────────────────────
def _make_chunk(
    id_: str = "dog_puppy_exercise",
    species: str = "dog",
    life_stage: str = "puppy",
    topic: str = "exercise",
    tags: tuple = ("puppy", "exercise"),
    source: str = "AKC puppy exercise guidance (illustrative)",
    source_url: str = "https://www.akc.org/expert-advice/",
    retrieved_on: str = "2026-04-26",
    task_templates: tuple = (
        {"description": "Short walk", "duration_minutes": 15,
         "frequency": "daily", "priority": "high",
         "suggested_time": "08:00"},
    ),
    body: str = "Puppies need short walks.",
) -> KBChunk:
    """Build a minimal valid KBChunk for unit tests."""
    return KBChunk(
        id=id_,
        species=species, life_stage=life_stage, topic=topic,
        tags=tags, source=source, source_url=source_url,
        retrieved_on=retrieved_on,
        task_templates=task_templates,
        body=body,
    )


# ──────────────────────────────────────────────────────────────────
# Empty input
# ──────────────────────────────────────────────────────────────────
def test_empty_retrieval_returns_empty_suggestions():
    assert generate_suggestions([]) == []


# ──────────────────────────────────────────────────────────────────
# Single chunk, single template
# ──────────────────────────────────────────────────────────────────
def test_single_template_produces_one_suggestion():
    chunk = _make_chunk()
    suggestions = generate_suggestions(
        [RetrievalResult(chunk=chunk, score=0.5)]
    )
    assert len(suggestions) == 1


def test_suggestion_carries_template_fields():
    chunk = _make_chunk(task_templates=(
        {"description": "Morning walk", "duration_minutes": 30,
         "frequency": "daily", "priority": "high",
         "suggested_time": "07:30"},
    ))
    s = generate_suggestions(
        [RetrievalResult(chunk=chunk, score=0.5)]
    )[0]
    assert s.description == "Morning walk"
    assert s.duration_minutes == 30
    assert s.frequency == "daily"
    assert s.priority == "high"
    assert s.suggested_time == "07:30"


def test_suggestion_carries_chunk_id_as_source_id():
    chunk = _make_chunk(id_="dog_puppy_exercise")
    s = generate_suggestions(
        [RetrievalResult(chunk=chunk, score=0.5)]
    )[0]
    assert s.source_id == "dog_puppy_exercise"


def test_suggestion_carries_source_attribution():
    chunk = _make_chunk(
        source="ASPCA Puppy Care",
        source_url="https://www.aspca.org/pet-care",
    )
    s = generate_suggestions(
        [RetrievalResult(chunk=chunk, score=0.5)]
    )[0]
    assert s.source == "ASPCA Puppy Care"
    assert s.source_url == "https://www.aspca.org/pet-care"


def test_suggestion_carries_retrieval_score():
    chunk = _make_chunk()
    s = generate_suggestions(
        [RetrievalResult(chunk=chunk, score=0.42)]
    )[0]
    assert s.retrieval_score == 0.42


# ──────────────────────────────────────────────────────────────────
# Fan-out: multiple templates per chunk
# ──────────────────────────────────────────────────────────────────
def test_multiple_templates_fan_out_in_declared_order():
    chunk = _make_chunk(task_templates=(
        {"description": "Morning walk", "duration_minutes": 30,
         "frequency": "daily", "priority": "high",
         "suggested_time": "07:30"},
        {"description": "Evening walk", "duration_minutes": 30,
         "frequency": "daily", "priority": "medium",
         "suggested_time": "18:00"},
        {"description": "Free play", "duration_minutes": 20,
         "frequency": "daily", "priority": "medium",
         "suggested_time": "11:00"},
    ))
    suggestions = generate_suggestions(
        [RetrievalResult(chunk=chunk, score=0.5)]
    )
    assert len(suggestions) == 3
    # Generator preserves the order task_templates were declared in
    # the KB doc; chronological re-ordering is the caller's job.
    times = [s.suggested_time for s in suggestions]
    assert times == ["07:30", "18:00", "11:00"]


# ──────────────────────────────────────────────────────────────────
# Multiple chunks combine into one flat list
# ──────────────────────────────────────────────────────────────────
def test_multiple_chunks_combine_into_one_flat_list():
    chunk_a = _make_chunk(
        id_="dog_puppy_feeding", topic="feeding",
        task_templates=(
            {"description": "Breakfast", "duration_minutes": 10,
             "frequency": "daily", "priority": "high",
             "suggested_time": "07:00"},
        ),
    )
    chunk_b = _make_chunk(
        id_="dog_puppy_exercise", topic="exercise",
        task_templates=(
            {"description": "Walk", "duration_minutes": 15,
             "frequency": "daily", "priority": "high",
             "suggested_time": "08:00"},
        ),
    )
    suggestions = generate_suggestions([
        RetrievalResult(chunk=chunk_a, score=0.4),
        RetrievalResult(chunk=chunk_b, score=0.3),
    ])
    assert len(suggestions) == 2
    assert {s.source_id for s in suggestions} == {
        "dog_puppy_feeding", "dog_puppy_exercise"
    }


def test_chunks_with_different_scores_propagate_individually():
    chunk_a = _make_chunk(id_="a")
    chunk_b = _make_chunk(id_="b")
    suggestions = generate_suggestions([
        RetrievalResult(chunk=chunk_a, score=0.9),
        RetrievalResult(chunk=chunk_b, score=0.1),
    ])
    by_id = {s.source_id: s.retrieval_score for s in suggestions}
    assert by_id["a"] == 0.9
    assert by_id["b"] == 0.1


# ──────────────────────────────────────────────────────────────────
# Rationale
# ──────────────────────────────────────────────────────────────────
def test_rationale_includes_source_id_and_source():
    chunk = _make_chunk(
        id_="dog_puppy_exercise",
        source="AKC guidance",
    )
    s = generate_suggestions(
        [RetrievalResult(chunk=chunk, score=0.5)]
    )[0]
    assert "dog_puppy_exercise" in s.rationale
    assert "AKC guidance" in s.rationale


def test_rationale_includes_pet_name_when_provided():
    chunk = _make_chunk()
    s = generate_suggestions(
        [RetrievalResult(chunk=chunk, score=0.5)],
        pet_name="Mochi",
    )[0]
    assert "Mochi" in s.rationale


def test_rationale_omits_pet_name_when_none():
    chunk = _make_chunk()
    s = generate_suggestions(
        [RetrievalResult(chunk=chunk, score=0.5)]
    )[0]
    # Without a pet name, the rationale should still be non-empty,
    # and shouldn't contain the literal word 'None'.
    assert len(s.rationale) > 0
    assert "None" not in s.rationale


# ──────────────────────────────────────────────────────────────────
# Determinism (Tier C contract)
# ──────────────────────────────────────────────────────────────────
def test_same_input_produces_identical_output_twice():
    chunk = _make_chunk(task_templates=(
        {"description": "A", "duration_minutes": 10,
         "frequency": "daily", "priority": "high",
         "suggested_time": "07:00"},
        {"description": "B", "duration_minutes": 20,
         "frequency": "weekly", "priority": "low",
         "suggested_time": "18:00"},
    ))
    results = [RetrievalResult(chunk=chunk, score=0.3)]
    a = generate_suggestions(results)
    b = generate_suggestions(results)
    assert a == b


# ──────────────────────────────────────────────────────────────────
# End-to-end: real Retriever → Generator
# ──────────────────────────────────────────────────────────────────
def test_end_to_end_with_real_retriever():
    """Catch any data-shape mismatch between the modules by running
    the full retrieve → generate path against the real KB."""
    from src.retriever import Retriever
    r = Retriever()
    results = r.retrieve(
        species="dog", life_stage="puppy",
        query="exercise walks", top_k=3,
    )
    suggestions = generate_suggestions(results, pet_name="Buddy")

    # dog_puppy_exercise alone has 3 task_templates, plus templates
    # from feeding (3) and grooming (2) → 8 total when top_k=3.
    assert len(suggestions) >= 3

    # Every suggestion's source_id must trace back to a retrieved
    # chunk — this is the citation invariant the Phase 6 guardrail
    # will enforce as a backstop.
    retrieved_ids = {res.chunk.id for res in results}
    for s in suggestions:
        assert s.source_id in retrieved_ids
        assert "Buddy" in s.rationale
        assert s.source.strip() != ""
        assert s.source_url.startswith("http")
