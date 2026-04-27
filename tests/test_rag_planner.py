"""
RAG Planner Tests
─────────────────
Unit tests for the Phase 5 orchestrator that ties Retriever + Generator
into the existing PawPal+ Core.

Run with:  pytest tests/test_rag_planner.py -v
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generator import Suggestion
from src.pawpal_system import Pet, Task
from src.rag_planner import (
    apply_suggestions_to_pet,
    derive_life_stage,
    suggest_tasks_for_pet,
    suggestion_to_task,
)
from src.retriever import Retriever


# ──────────────────────────────────────────────────────────────────
# Shared retriever fixture (avoids re-building TF-IDF for every test)
# ──────────────────────────────────────────────────────────────────
_retriever: Retriever | None = None


def _r() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# ──────────────────────────────────────────────────────────────────
# Life-stage derivation
# ──────────────────────────────────────────────────────────────────
def test_dog_under_one_is_puppy():
    assert derive_life_stage("dog", 0) == "puppy"


def test_dog_one_through_six_is_adult():
    assert derive_life_stage("dog", 1) == "adult"
    assert derive_life_stage("dog", 5) == "adult"
    assert derive_life_stage("dog", 6) == "adult"


def test_dog_seven_plus_is_senior():
    assert derive_life_stage("dog", 7) == "senior"
    assert derive_life_stage("dog", 12) == "senior"


def test_cat_under_one_is_kitten():
    assert derive_life_stage("cat", 0) == "kitten"


def test_cat_one_through_nine_is_adult():
    assert derive_life_stage("cat", 1) == "adult"
    assert derive_life_stage("cat", 9) == "adult"


def test_cat_ten_plus_is_senior():
    assert derive_life_stage("cat", 10) == "senior"
    assert derive_life_stage("cat", 18) == "senior"


def test_rabbit_any_age_is_adult():
    """The KB only covers adult rabbits, and derivation reflects that."""
    assert derive_life_stage("rabbit", 0) == "adult"
    assert derive_life_stage("rabbit", 5) == "adult"


def test_unknown_species_returns_none():
    assert derive_life_stage("hamster", 1) is None
    assert derive_life_stage("ferret", 3) is None


def test_species_lookup_is_case_insensitive():
    assert derive_life_stage("Dog", 0) == "puppy"
    assert derive_life_stage("CAT", 1) == "adult"
    assert derive_life_stage("  rabbit ", 2) == "adult"


# ──────────────────────────────────────────────────────────────────
# suggest_tasks_for_pet — the orchestrator
# ──────────────────────────────────────────────────────────────────
def test_suggest_for_puppy_dog_returns_only_puppy_chunks():
    pet = Pet(name="Buddy", species="dog", age=0)
    suggestions = suggest_tasks_for_pet(pet, retriever=_r())
    assert suggestions
    for s in suggestions:
        assert s.source_id.startswith("dog_puppy_")


def test_suggest_for_adult_cat_returns_only_adult_cat_chunks():
    pet = Pet(name="Whiskers", species="cat", age=5)
    suggestions = suggest_tasks_for_pet(pet, retriever=_r())
    assert suggestions
    for s in suggestions:
        assert s.source_id.startswith("cat_adult_")


def test_suggest_for_senior_dog_returns_only_senior_dog_chunks():
    pet = Pet(name="Rex", species="dog", age=10)
    suggestions = suggest_tasks_for_pet(pet, retriever=_r())
    assert suggestions
    for s in suggestions:
        assert s.source_id.startswith("dog_senior_")


def test_suggest_for_unknown_species_returns_empty_list():
    pet = Pet(name="Hammy", species="hamster", age=2)
    assert suggest_tasks_for_pet(pet, retriever=_r()) == []


def test_suggest_passes_pet_name_into_rationale():
    pet = Pet(name="Mochi", species="dog", age=3)
    suggestions = suggest_tasks_for_pet(pet, retriever=_r())
    assert suggestions
    for s in suggestions:
        assert "Mochi" in s.rationale


def test_suggest_with_query_picks_relevant_top_chunk():
    """top_k=1 + a topic query should land on the matching chunk."""
    pet = Pet(name="Buddy", species="dog", age=0)
    feeding = suggest_tasks_for_pet(
        pet, query="feeding meals nutrition", top_k=1, retriever=_r(),
    )
    exercise = suggest_tasks_for_pet(
        pet, query="walks exercise activity", top_k=1, retriever=_r(),
    )
    # All emitted suggestions trace to the chosen top chunk.
    assert {s.source_id for s in feeding} == {"dog_puppy_feeding"}
    assert {s.source_id for s in exercise} == {"dog_puppy_exercise"}


# ──────────────────────────────────────────────────────────────────
# suggestion_to_task conversion
# ──────────────────────────────────────────────────────────────────
def _sample_suggestion(frequency: str = "daily") -> Suggestion:
    """Build a Suggestion with realistic field values for unit tests."""
    return Suggestion(
        description="Morning walk",
        duration_minutes=30,
        frequency=frequency,
        priority="high",
        suggested_time="07:30",
        source_id="dog_adult_exercise",
        source="AKC adult dog exercise (illustrative)",
        source_url="https://www.akc.org/expert-advice/",
        retrieval_score=0.4,
        rationale=(
            "Suggested for Mochi: 'Morning walk' "
            "(grounded in dog_adult_exercise; AKC adult dog exercise (illustrative))"
        ),
    )


def test_conversion_copies_basic_fields():
    s = _sample_suggestion()
    t = suggestion_to_task(s, due_date="2026-04-26")
    assert t.description == "Morning walk"
    assert t.duration_minutes == 30
    assert t.priority == "high"
    assert t.due_time == "07:30"
    assert t.due_date == "2026-04-26"


def test_conversion_translates_one_time_frequency_to_once():
    """KB schema uses 'one-time' but PawPal+ Task uses 'once'."""
    s = _sample_suggestion(frequency="one-time")
    t = suggestion_to_task(s, due_date="2026-04-26")
    assert t.frequency == "once"


def test_conversion_passes_daily_and_weekly_through_unchanged():
    daily_t = suggestion_to_task(
        _sample_suggestion(frequency="daily"), due_date="2026-04-26",
    )
    weekly_t = suggestion_to_task(
        _sample_suggestion(frequency="weekly"), due_date="2026-04-26",
    )
    assert daily_t.frequency == "daily"
    assert weekly_t.frequency == "weekly"


def test_conversion_default_due_date_is_today():
    t = suggestion_to_task(_sample_suggestion())
    assert t.due_date == date.today().isoformat()


def test_conversion_does_not_pre_stamp_pet_name():
    """pet_name is set by Pet.add_task, not by the conversion."""
    t = suggestion_to_task(_sample_suggestion(), due_date="2026-04-26")
    assert t.pet_name == ""


# ──────────────────────────────────────────────────────────────────
# apply_suggestions_to_pet — full path Pet ← Suggestion → Task
# ──────────────────────────────────────────────────────────────────
def test_apply_suggestions_adds_correct_number_of_tasks():
    pet = Pet(name="Mochi", species="dog", age=3)
    initial = len(pet.get_tasks())

    suggestions = suggest_tasks_for_pet(pet, retriever=_r(), top_k=1)
    added = apply_suggestions_to_pet(pet, suggestions)

    assert len(added) == len(suggestions)
    assert len(pet.get_tasks()) == initial + len(suggestions)


def test_added_tasks_are_stamped_with_pet_name():
    pet = Pet(name="Mochi", species="dog", age=3)
    suggestions = suggest_tasks_for_pet(pet, retriever=_r(), top_k=1)
    apply_suggestions_to_pet(pet, suggestions)
    for t in pet.get_tasks():
        assert t.pet_name == "Mochi"


def test_full_pipeline_does_not_overwrite_existing_user_tasks():
    """RAG suggestions are additive — user-entered tasks survive."""
    pet = Pet(name="Mochi", species="dog", age=3)
    user_task = Task(
        description="Vet appointment", due_date="2026-04-26",
        due_time="14:00", duration_minutes=30, priority="high",
    )
    pet.add_task(user_task)

    suggestions = suggest_tasks_for_pet(pet, retriever=_r(), top_k=1)
    apply_suggestions_to_pet(pet, suggestions)

    descriptions = [t.description for t in pet.get_tasks()]
    assert "Vet appointment" in descriptions


def test_apply_to_empty_suggestions_is_noop():
    pet = Pet(name="Mochi", species="dog", age=3)
    initial = len(pet.get_tasks())
    apply_suggestions_to_pet(pet, [])
    assert len(pet.get_tasks()) == initial
