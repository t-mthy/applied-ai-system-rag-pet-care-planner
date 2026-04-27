"""
PawPal+ AI — Generator
──────────────────────
Template-based structured synthesis. Takes the chunks a Retriever
returned for a pet profile and expands each chunk's `task_templates`
into concrete Suggestion objects, with citations attached.

This is Tier C — fully offline, fully deterministic, and mechanically
incapable of producing an ungrounded claim. Every Suggestion's
description, duration, frequency, priority, and time come straight
from the chunk's structured fields, and every Suggestion carries the
chunk's id as its source_id citation.

The Phase 6 guardrail will wrap each Suggestion with refined
confidence + grounding checks; this module only handles the raw
materialization step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.retriever import RetrievalResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Data class
# ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Suggestion:
    """One generated task suggestion, fully grounded in a KB chunk.

    Every field except `rationale` is copied directly from a chunk's
    `task_templates` entry or from the chunk's frontmatter — they
    are never invented or paraphrased. `rationale` is composed by
    `_build_rationale` from those same fields, so it cannot reference
    anything outside the source chunk either.
    """
    # Task fields — these will become PawPal+ Task attributes in Phase 5.
    description: str
    duration_minutes: int
    frequency: str           # "one-time" | "daily" | "weekly"
    priority: str            # "low" | "medium" | "high"
    suggested_time: str      # "HH:MM"

    # Citation (mandatory by architecture spec).
    source_id: str           # = KBChunk.id; the citation handle the guardrail checks.
    source: str              # human-readable organization attribution
    source_url: str

    # Retrieval-aware signal. Phase 6's guardrail will turn this into a
    # refined `confidence` value; the Generator just forwards what the
    # Retriever returned.
    retrieval_score: float

    # One-line human-readable explanation, shown in the UI alongside the task.
    rationale: str


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────
def _build_rationale(
    template_description: str,
    chunk_id: str,
    chunk_source: str,
    pet_name: str | None,
) -> str:
    """Compose a short human-readable rationale string for the UI.

    Every rationale includes the source attribution and the chunk id,
    so any suggestion the user sees in the schedule can be traced back
    to its grounding document. Including `pet_name` makes the line
    read naturally next to the pet's task list.
    """
    pet_clause = f" for {pet_name}" if pet_name else ""
    return (
        f"Suggested{pet_clause}: '{template_description}' "
        f"(grounded in {chunk_id}; {chunk_source})"
    )


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────
def generate_suggestions(
    retrieval_results: list[RetrievalResult],
    pet_name: str | None = None,
) -> list[Suggestion]:
    """Expand every retrieved chunk's task templates into Suggestions.

    Parameters
    ----------
    retrieval_results
        The list returned by `Retriever.retrieve(...)`. May be empty.
    pet_name
        Optional pet name used in rationale strings (UI affordance).

    Returns
    -------
    list[Suggestion]
        Flat list. Order follows retrieval order; within each chunk,
        order follows the order `task_templates` were declared in the
        KB doc. Callers that want a chronological day plan should sort
        the result by `suggested_time`.

    Notes
    -----
    Empty input → empty output (no error). Every Suggestion carries
    its chunk's `id` as `source_id`, which the Phase 6 guardrail
    cross-checks against the retrieved set as a backstop.
    """
    if not retrieval_results:
        logger.info("Generator: empty retrieval, returning 0 suggestions")
        return []

    suggestions: list[Suggestion] = []

    for result in retrieval_results:
        chunk = result.chunk
        # Each chunk's task_templates expands into one Suggestion each.
        for template in chunk.task_templates:
            suggestions.append(Suggestion(
                description=template["description"],
                duration_minutes=template["duration_minutes"],
                frequency=template["frequency"],
                priority=template["priority"],
                suggested_time=template["suggested_time"],
                source_id=chunk.id,
                source=chunk.source,
                source_url=chunk.source_url,
                retrieval_score=result.score,
                rationale=_build_rationale(
                    template["description"],
                    chunk.id,
                    chunk.source,
                    pet_name,
                ),
            ))

    logger.info(
        "Generator produced %d suggestion(s) from %d chunk(s) (pet_name=%r)",
        len(suggestions),
        len(retrieval_results),
        pet_name,
    )
    return suggestions
