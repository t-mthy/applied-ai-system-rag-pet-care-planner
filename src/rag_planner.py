"""
PawPal+ AI — RAG Planner
────────────────────────
Single seam between the new RAG layer (Retriever + Generator) and the
existing PawPal+ Core (Owner / Pet / Task / Scheduler).

This module is the only place where a Pet object meets a Suggestion
object. Keeping the seam narrow means:
  - The Core's 21 existing tests stay green (Pet / Task aren't modified).
  - Future swaps to a different Generator (e.g. local LLM) only touch
    src/generator.py, not anything below this module.

The planner is offline-first like every other module: it builds on the
Retriever's TF-IDF index and the Generator's deterministic templating,
nothing else.
"""

from __future__ import annotations

import logging
from datetime import date

from src.generator import Suggestion, generate_suggestions
from src.guardrails import (
    apply_output_guardrail,
    log_event,
    validate_pet_profile,
    validate_query,
)
from src.pawpal_system import Pet, Task
from src.retriever import Retriever

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Life-stage derivation
# ──────────────────────────────────────────────────────────────────
# Pet.age is an int (years), so the puppy/kitten cutoff is "age 0".
# A 6-month-old puppy registers as age=0; a 1-year-old is treated as
# adult. These thresholds are conservative defaults — the KB only
# carries documents under these life_stage labels.
#
# Format: list of (min_age_inclusive, life_stage) bands sorted ascending.
_LIFE_STAGE_BANDS: dict[str, list[tuple[int, str]]] = {
    "dog":    [(0, "puppy"),  (1, "adult"),  (7,  "senior")],
    "cat":    [(0, "kitten"), (1, "adult"),  (10, "senior")],
    "rabbit": [(0, "adult")],  # KB only covers adult rabbits
}


def derive_life_stage(species: str, age: int) -> str | None:
    """Map (species, age in years) → life_stage label.

    Returns ``None`` when species is not present in the KB. Thresholds
    err on the conservative side — a 1-year-old dog is classified
    'adult' even though large breeds may still be juvenile, because
    we'd rather show adult-appropriate suggestions than misclassify a
    working dog as a puppy.
    """
    bands = _LIFE_STAGE_BANDS.get(species.lower().strip())
    if bands is None:
        return None

    # Walk ascending bands; the highest threshold whose min_age <= age
    # gives us the label. The loop is deterministic because the band
    # list is constant and sorted.
    label = bands[0][1]
    for threshold, stage in bands:
        if age >= threshold:
            label = stage
    return label


# ──────────────────────────────────────────────────────────────────
# Frequency translation: KB schema ↔ PawPal+ Task vocabulary
# ──────────────────────────────────────────────────────────────────
# The KB schema (Phase 2) uses "one-time"; the existing Task class
# uses "once". Both are valid in their own context. Translating once
# at this seam keeps each side natural in its own files.
_KB_TO_TASK_FREQUENCY = {
    "one-time": "once",
    "daily":    "daily",
    "weekly":   "weekly",
}


# ──────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ──────────────────────────────────────────────────────────────────
def suggest_tasks_for_pet(
    pet: Pet,
    query: str | None = None,
    top_k: int | None = None,
    retriever: Retriever | None = None,
) -> list[Suggestion]:
    """Run the full RAG pipeline for one Pet.

    Steps:
      1. Derive ``life_stage`` from ``pet.species`` + ``pet.age``.
      2. If derivation returns ``None`` (unsupported species), return [].
      3. Retrieve candidate chunks (metadata filter + optional TF-IDF).
      4. Expand chunks into Suggestions via the Generator.

    A ``retriever`` can be passed in for performance (e.g. a Streamlit
    app caches one in ``session_state``). When omitted, a fresh
    Retriever is built — fine for CLI and tests, but rebuilds the
    TF-IDF index on every call, so don't do this in a request loop.

    The pipeline runs in this order, with structured log events at
    each step (see `src/guardrails.py` for the JSON-lines format):

      1. validate_pet_profile  — reject unsupported species / bad ages
      2. validate_query        — reject oversized / non-string queries
      3. derive_life_stage     — map age to KB life-stage label
      4. retriever.retrieve    — metadata filter + TF-IDF cosine
      5. generate_suggestions  — template-based synthesis
      6. apply_output_guardrail— citation-validation backstop
    """
    # ── 1 & 2. Input validation (reject early, log the reason) ──
    pv = validate_pet_profile(pet.species, pet.age)
    if not pv.valid:
        log_event(
            "input.rejected",
            stage="pet_profile",
            pet=pet.name,
            species=pet.species,
            age=pet.age,
            errors=list(pv.errors),
        )
        return []

    qv = validate_query(query)
    if not qv.valid:
        log_event(
            "input.rejected",
            stage="query",
            pet=pet.name,
            errors=list(qv.errors),
        )
        return []

    # ── 3. Derive life stage ──
    life_stage = derive_life_stage(pet.species, pet.age)
    if life_stage is None:
        # Defense in depth — validate_pet_profile should already
        # have caught this, but if a new species is added to the KB
        # without updating the bands, return clean empty.
        log_event(
            "input.rejected",
            stage="life_stage_derivation",
            pet=pet.name,
            species=pet.species,
            age=pet.age,
        )
        return []

    if retriever is None:
        retriever = Retriever()

    # ── 4. Retrieve ──
    results = retriever.retrieve(
        species=pet.species.lower().strip(),
        life_stage=life_stage,
        query=query,
        top_k=top_k,
    )
    log_event(
        "retrieve",
        pet=pet.name,
        species=pet.species,
        life_stage=life_stage,
        query=query,
        top_k=top_k,
        n_results=len(results),
        retrieved_ids=[r.chunk.id for r in results],
    )

    # ── 5. Generate ──
    raw_suggestions = generate_suggestions(results, pet_name=pet.name)
    log_event(
        "generate",
        pet=pet.name,
        n_suggestions=len(raw_suggestions),
    )

    # ── 6. Output guardrail (citation-validation backstop) ──
    guard = apply_output_guardrail(raw_suggestions, results)
    log_event(
        "guard",
        pet=pet.name,
        n_accepted=len(guard.accepted),
        n_rejected=len(guard.rejected),
        rejected_reasons=[reason for _, reason in guard.rejected],
    )

    logger.info(
        "RAG planner: pet=%r species=%r life_stage=%r → %d accepted, %d rejected",
        pet.name, pet.species, life_stage,
        len(guard.accepted), len(guard.rejected),
    )
    return guard.accepted


# ──────────────────────────────────────────────────────────────────
# Suggestion → Task conversion
# ──────────────────────────────────────────────────────────────────
def suggestion_to_task(
    suggestion: Suggestion,
    due_date: str | None = None,
) -> Task:
    """Convert one Suggestion into a PawPal+ Task.

    ``due_date`` defaults to today (ISO format). The suggestion's
    ``suggested_time`` becomes the Task's ``due_time``. Frequency is
    translated through ``_KB_TO_TASK_FREQUENCY`` so the resulting
    Task uses the existing PawPal+ vocabulary. ``pet_name`` is left
    blank here — ``Pet.add_task`` stamps it on insertion, matching
    the existing PawPal+ flow exactly.
    """
    if due_date is None:
        due_date = date.today().isoformat()

    return Task(
        description=suggestion.description,
        due_date=due_date,
        due_time=suggestion.suggested_time,
        duration_minutes=suggestion.duration_minutes,
        priority=suggestion.priority,
        frequency=_KB_TO_TASK_FREQUENCY[suggestion.frequency],
    )


def apply_suggestions_to_pet(
    pet: Pet,
    suggestions: list[Suggestion],
    due_date: str | None = None,
) -> list[Task]:
    """Convert a list of Suggestions into Tasks and add them to a Pet.

    Returns the list of Tasks that were added. This is the
    convenience helper the UI calls when the user clicks
    "Add selected to plan" on a list of suggestions. Existing tasks
    on the Pet are preserved — RAG suggestions are additive.
    """
    added: list[Task] = []
    for s in suggestions:
        t = suggestion_to_task(s, due_date=due_date)
        pet.add_task(t)
        added.append(t)
    logger.info(
        "RAG planner: added %d task(s) to pet=%r",
        len(added), pet.name,
    )
    return added
