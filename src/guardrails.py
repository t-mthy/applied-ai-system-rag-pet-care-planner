"""
PawPal+ AI — Guardrails & Reliability
─────────────────────────────────────
Three components, per `assets/architecture.md` (sections 3 and 4):

  1. Input Validator — pre-flight checks on (species, age, query)
     before retrieval. Rejects unsupported species and nonsensical
     ages without ever touching the KB or the TF-IDF index.

  2. Output Guardrail — drops any suggestion whose `source_id` isn't
     in the retrieved chunk set. With Tier C this is mostly a
     backstop, since the template-based generator copies source_id
     straight from each chunk's frontmatter — but it runs
     unconditionally so a future swap to an LLM-based Generator
     inherits the same protection automatically.

  3. Structured Logger — JSON-lines event log written to
     `logs/pawpal.jsonl`. Every retrieve / generate / guard / reject /
     input-rejected event is captured for offline analysis.

Plus `compute_confidence(retrieval_score)`, a small utility the UI
and CLI use at display time to turn raw cosine similarity into a
user-friendly confidence value.

All four are pure-Python and offline. No external dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.generator import Suggestion
from src.retriever import RetrievalResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Allowed values (locked in Phase 2 KB schema)
# ──────────────────────────────────────────────────────────────────
# Mirrored here so input validation doesn't need to parse the KB.
ALLOWED_SPECIES = frozenset({"dog", "cat", "rabbit"})

# Universal age band — any sane terrestrial pet age fits inside.
# Per-species nuance is handled by the life_stage labels in Phase 5.
MIN_AGE = 0
MAX_AGE = 40

# Maximum query length. We have no LLM, so this isn't prompt-injection
# defense — it's a sanity cap so the TF-IDF vectorizer doesn't get
# handed gigabytes of text and the structured log doesn't fill up
# with stray content.
MAX_QUERY_LEN = 500


# ──────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a pre-flight input check.

    `valid=True` means the call may proceed. `errors` are blockers;
    `warnings` are non-blocking notices logged but not returned to
    the user. Both populated for full transparency.
    """
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuardResult:
    """Output of running `apply_output_guardrail`.

    `accepted` is the suggestions that passed the citation check.
    `rejected` is the suggestions that were dropped, paired with a
    one-line reason — useful for the structured log and for any
    future "rejected suggestions" UI affordance.
    """
    accepted: list[Suggestion]
    rejected: list[tuple[Suggestion, str]]


# ──────────────────────────────────────────────────────────────────
# Input validation
# ──────────────────────────────────────────────────────────────────
def validate_pet_profile(species: str, age: int) -> ValidationResult:
    """Pre-flight check on a Pet's species and age.

    Rejects:
      - unsupported species (anything outside the KB's coverage)
      - non-integer ages
      - negative ages
      - implausibly large ages (e.g. a 250-year-old cat)
      - empty / None species

    Validation runs before any retrieval, so bad inputs never reach
    the TF-IDF vectorizer or the file system.
    """
    errors: list[str] = []

    s = (species or "").lower().strip()
    if not s:
        errors.append("species is empty")
    elif s not in ALLOWED_SPECIES:
        errors.append(
            f"species {species!r} is not in the supported KB "
            f"({sorted(ALLOWED_SPECIES)})"
        )

    # `bool` is a subclass of `int` in Python; reject it explicitly so
    # validate_pet_profile("dog", True) doesn't slip through as age=1.
    if isinstance(age, bool) or not isinstance(age, int):
        errors.append(f"age must be an integer, got {type(age).__name__}")
    elif age < MIN_AGE:
        errors.append(f"age must be >= {MIN_AGE}, got {age}")
    elif age > MAX_AGE:
        errors.append(f"age must be <= {MAX_AGE}, got {age}")

    return ValidationResult(valid=not errors, errors=tuple(errors))


def validate_query(query: str | None) -> ValidationResult:
    """Sanity-check the optional free-text query string.

    `None` and the empty string are valid (they mean "no focus query").
    Long queries are capped here even though the retriever's
    vectorizer would also drop most of the content via stop-word
    removal — the cap keeps log lines compact.
    """
    if query is None:
        return ValidationResult(valid=True)

    if not isinstance(query, str):
        return ValidationResult(
            valid=False,
            errors=(
                f"query must be a string, got {type(query).__name__}",
            ),
        )

    if len(query) > MAX_QUERY_LEN:
        return ValidationResult(
            valid=False,
            errors=(
                f"query is {len(query)} chars, max {MAX_QUERY_LEN}",
            ),
        )

    return ValidationResult(valid=True)


# ──────────────────────────────────────────────────────────────────
# Confidence
# ──────────────────────────────────────────────────────────────────
def compute_confidence(retrieval_score: float) -> float:
    """Map a raw retrieval score to a per-suggestion confidence.

    Formula:
        0.5 (baseline: chunk passed metadata filter for species + life_stage)
      + 0.5 × clamp(retrieval_score, 0, 1)

    Range: [0.5, 1.0] for any chunk we'd actually emit (the metadata
    floor is exact, since unmatched chunks never make it out of the
    retriever in the first place). `retrieval_score=1.0` corresponds
    to the no-query branch in the Retriever, where every metadata-
    matching chunk is treated as equally relevant.
    """
    score = max(0.0, min(1.0, retrieval_score))
    return 0.5 + 0.5 * score


# ──────────────────────────────────────────────────────────────────
# Output guardrail
# ──────────────────────────────────────────────────────────────────
def apply_output_guardrail(
    suggestions: list[Suggestion],
    retrieval_results: list[RetrievalResult],
) -> GuardResult:
    """Drop any suggestion whose `source_id` isn't in the retrieved set.

    With Tier C the template-based Generator copies `source_id`
    straight from each chunk's frontmatter, so there's nothing to
    drop in practice. The guardrail still runs every time — it's the
    backstop that future-proofs against an LLM-based Generator that
    could in principle hallucinate a citation handle.

    Suggestions are *not* mutated. Confidence is computed at display
    time via `compute_confidence` so each rendering layer (UI / CLI /
    eval harness) can present it however makes sense.
    """
    valid_ids = {res.chunk.id for res in retrieval_results}

    accepted: list[Suggestion] = []
    rejected: list[tuple[Suggestion, str]] = []

    for s in suggestions:
        if s.source_id not in valid_ids:
            reason = (
                f"source_id {s.source_id!r} not in retrieved set "
                f"(retrieved: {sorted(valid_ids)})"
            )
            rejected.append((s, reason))
            continue
        accepted.append(s)

    return GuardResult(accepted=accepted, rejected=rejected)


# ──────────────────────────────────────────────────────────────────
# Structured logger (JSON lines)
# ──────────────────────────────────────────────────────────────────
# Default log location, resolved relative to the project root (parent
# of src/). Override with the PAWPAL_LOG_PATH env var, or by calling
# `configure_logger(...)` directly — used by tests to redirect output
# to a temp file.
_DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parent.parent / "logs" / "pawpal.jsonl"
)
_LOG_LOCK = threading.Lock()
_log_path: Path | None = None
_log_enabled: bool = True


def configure_logger(
    path: Path | str | None = None,
    enabled: bool = True,
) -> None:
    """Configure (or reset) the structured logger.

    Default path is `logs/pawpal.jsonl` under the project root. The
    PAWPAL_LOG_PATH env var overrides it when no explicit path is
    passed. Set `enabled=False` to silence the logger entirely
    (events become no-ops). The directory is created lazily on first
    write — calling `configure_logger` does not create files.
    """
    global _log_path, _log_enabled
    if path is None:
        env_path = os.environ.get("PAWPAL_LOG_PATH")
        path = Path(env_path) if env_path else _DEFAULT_LOG_PATH
    _log_path = Path(path)
    _log_enabled = enabled


def log_event(event: str, **fields) -> None:
    """Append one JSON line for one event.

    Always adds an ISO-8601 UTC `ts` field and the event name.
    Keyword arguments are merged in as additional fields. Writes are
    serialized through a thread lock so concurrent calls don't
    interleave bytes mid-line.

    The function never raises — logging failures degrade silently,
    because a broken log line should not break a user-facing call.
    """
    if _log_path is None:
        configure_logger()  # lazy default
    if not _log_enabled:
        return

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }
    record.update(fields)

    try:
        with _LOG_LOCK:
            _log_path.parent.mkdir(parents=True, exist_ok=True)
            with _log_path.open("a", encoding="utf-8") as f:
                # default=str handles datetimes / Paths / etc. so we
                # never blow up on a value we forgot to serialize.
                f.write(json.dumps(record, default=str) + "\n")
    except OSError as exc:
        # Fall back to stdlib logging so the message isn't lost
        # entirely, but never propagate.
        logger.warning("structured log write failed: %s", exc)
