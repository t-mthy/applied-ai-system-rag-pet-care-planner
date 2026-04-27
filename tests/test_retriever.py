"""
Retriever Tests
───────────────
Unit tests for the Phase 3 Retriever module.

Run with:  pytest tests/test_retriever.py -v
"""

import os
import sys

# Make `src.retriever` importable when pytest runs from anywhere.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retriever import Retriever


# ──────────────────────────────────────────────────────────────────
# Shared retriever fixture
# ──────────────────────────────────────────────────────────────────
# Building the TF-IDF index is fast (~ms on this KB), but doing it
# once per test session keeps the suite snappy and removes incidental
# variance between tests.
_retriever: Retriever | None = None


def _r() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# ──────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────
def test_loads_all_sixteen_kb_docs():
    assert len(_r().chunks) == 16


def test_chunk_id_matches_filename_convention():
    """Every id should be species_life_stage_topic — the same shape
    used by Phase 2's filename validator."""
    for c in _r().chunks:
        parts = c.id.split("_")
        assert len(parts) == 3, f"unexpected id shape: {c.id!r}"
        assert parts[0] == c.species
        assert parts[1] == c.life_stage
        assert parts[2] == c.topic


def test_every_chunk_has_at_least_one_task_template():
    for c in _r().chunks:
        assert len(c.task_templates) >= 1


# ──────────────────────────────────────────────────────────────────
# Metadata filtering
# ──────────────────────────────────────────────────────────────────
def test_filter_dog_puppy_returns_only_dog_puppy_chunks():
    results = _r().retrieve(species="dog", life_stage="puppy")
    assert results, "expected some dog/puppy chunks"
    for res in results:
        assert res.chunk.species == "dog"
        assert res.chunk.life_stage == "puppy"


def test_filter_unknown_species_returns_empty():
    assert _r().retrieve(species="hamster", life_stage="adult") == []


def test_filter_unknown_life_stage_returns_empty():
    # 'kitten' is a valid life_stage but no dog chunks use it.
    assert _r().retrieve(species="dog", life_stage="kitten") == []


# ──────────────────────────────────────────────────────────────────
# Query ranking
# ──────────────────────────────────────────────────────────────────
def test_feeding_query_ranks_feeding_above_exercise():
    """A feeding-flavored query should put the feeding chunk on top."""
    results = _r().retrieve(
        species="dog", life_stage="adult",
        query="feeding meals nutrition portions",
    )
    by_topic = {res.chunk.topic: i for i, res in enumerate(results)}
    assert by_topic["feeding"] < by_topic["exercise"]


def test_exercise_query_ranks_exercise_above_feeding():
    results = _r().retrieve(
        species="dog", life_stage="adult",
        query="walks exercise activity training",
    )
    by_topic = {res.chunk.topic: i for i, res in enumerate(results)}
    assert by_topic["exercise"] < by_topic["feeding"]


def test_query_produces_some_nonzero_score():
    results = _r().retrieve(
        species="dog", life_stage="puppy",
        query="puppy exercise walks growth plates",
    )
    assert any(r.score > 0 for r in results)


def test_score_is_in_unit_interval():
    """Cosine similarity on a non-negative TF-IDF space lies in [0, 1]."""
    results = _r().retrieve(
        species="cat", life_stage="adult",
        query="play wand toys",
    )
    for r in results:
        assert 0.0 <= r.score <= 1.0


# ──────────────────────────────────────────────────────────────────
# Tag-driven synonym recall
# ──────────────────────────────────────────────────────────────────
def test_synonym_query_young_dog_finds_puppy_exercise():
    """The puppy exercise doc tags include 'young dog' — searching for
    that synonym (which never appears in the body) should still surface
    the chunk with a positive score."""
    results = _r().retrieve(
        species="dog", life_stage="puppy",
        query="young dog exercise",
    )
    assert results
    # The exercise chunk should rank above the feeding/grooming ones.
    assert results[0].chunk.topic == "exercise"
    assert results[0].score > 0


def test_synonym_query_older_dog_finds_senior_chunks():
    """Searching with 'older dog' (a synonym tag) under life_stage=senior
    should still produce positive ranking signal."""
    results = _r().retrieve(
        species="dog", life_stage="senior",
        query="older dog walks",
    )
    assert results
    assert results[0].chunk.topic == "exercise"


# ──────────────────────────────────────────────────────────────────
# No-query / empty-query behavior
# ──────────────────────────────────────────────────────────────────
def test_no_query_returns_all_filtered_chunks_with_score_one():
    results = _r().retrieve(species="cat", life_stage="adult")
    # cat / adult should have at least feeding and play.
    assert len(results) >= 2
    for r in results:
        assert r.score == 1.0


def test_empty_string_query_treated_as_no_query():
    a = _r().retrieve(species="rabbit", life_stage="adult", query="")
    b = _r().retrieve(species="rabbit", life_stage="adult")
    assert [r.chunk.id for r in a] == [r.chunk.id for r in b]


def test_whitespace_only_query_treated_as_no_query():
    a = _r().retrieve(species="rabbit", life_stage="adult", query="   ")
    b = _r().retrieve(species="rabbit", life_stage="adult")
    assert [r.chunk.id for r in a] == [r.chunk.id for r in b]


# ──────────────────────────────────────────────────────────────────
# top_k
# ──────────────────────────────────────────────────────────────────
def test_top_k_limits_result_count():
    results = _r().retrieve(species="dog", life_stage="adult", top_k=1)
    assert len(results) == 1


def test_top_k_none_returns_all_candidates():
    # dog / adult covers feeding, exercise, grooming → 3 chunks.
    results = _r().retrieve(species="dog", life_stage="adult")
    assert len(results) == 3


def test_top_k_larger_than_candidates_does_not_pad():
    results = _r().retrieve(species="dog", life_stage="adult", top_k=99)
    assert len(results) == 3
