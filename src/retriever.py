"""
PawPal+ AI — Retriever
──────────────────────
Loads the knowledge base from `kb/*.md` once at startup, builds a
scikit-learn TF-IDF index over each document's searchable text, and
answers retrieval queries using metadata pre-filtering followed by
cosine-similarity ranking.

This module is fully offline — no network, no API keys, no model
downloads. The TF-IDF vocabulary is built directly from the small
curated KB at startup. See `assets/architecture.md` (section 5,
"Conventions and Constraints") for the design rationale.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Default KB directory, resolved from this file's location.
# `Path(__file__)` is absolute, so this works regardless of cwd.
_DEFAULT_KB_DIR = Path(__file__).resolve().parent.parent / "kb"


# ──────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class KBChunk:
    """One parsed knowledge-base document.

    `id` matches the filename stem and is what later phases pass
    through the system as the citation `source_id`.
    """
    id: str
    species: str
    life_stage: str
    topic: str
    tags: tuple[str, ...]
    source: str
    source_url: str
    retrieved_on: str
    task_templates: tuple[dict, ...]
    body: str

    @property
    def searchable_text(self) -> str:
        """Topic + tags + body, concatenated. This is the string the
        TF-IDF vectorizer indexes for keyword matching. Tags are
        included so synonyms (e.g. 'young dog' for puppy docs) boost
        recall on user vocabulary that doesn't appear in the body."""
        return f"{self.topic} {' '.join(self.tags)} {self.body}"


@dataclass(frozen=True)
class RetrievalResult:
    """A single retrieval hit. `score` is cosine similarity in
    [0.0, 1.0] when a query was given, or 1.0 when no query was given
    (every metadata-matching chunk is treated as equally relevant)."""
    chunk: KBChunk
    score: float


# ──────────────────────────────────────────────────────────────────
# Document parsing
# ──────────────────────────────────────────────────────────────────
def _parse_doc(path: Path) -> KBChunk:
    """Read and validate a single `kb/*.md` document.

    Frontmatter is TOML between `+++` delimiters; the body is
    everything after the closing delimiter. Schema validation is
    light here — the Phase 2 KB validator catches structural errors
    at curation time. We re-check just enough fields to fail loudly
    if a doc has been hand-edited into a broken state.
    """
    text = path.read_text(encoding="utf-8")
    if not text.startswith("+++\n"):
        raise ValueError(f"{path.name}: missing opening +++ delimiter")
    end = text.find("\n+++", 4)
    if end == -1:
        raise ValueError(f"{path.name}: missing closing +++ delimiter")

    frontmatter_text = text[4:end]
    body = text[end + 4:].strip()

    fm = tomllib.loads(frontmatter_text)

    required = {"id", "species", "life_stage", "topic", "tags",
                "source", "source_url", "retrieved_on", "task_templates"}
    missing = required - fm.keys()
    if missing:
        raise ValueError(f"{path.name}: missing frontmatter keys {missing}")

    if fm["id"] != path.stem:
        raise ValueError(
            f"{path.name}: id={fm['id']!r} does not match filename stem"
        )

    return KBChunk(
        id=fm["id"],
        species=fm["species"],
        life_stage=fm["life_stage"],
        topic=fm["topic"],
        tags=tuple(fm["tags"]),
        source=fm["source"],
        source_url=fm["source_url"],
        retrieved_on=fm["retrieved_on"],
        task_templates=tuple(fm["task_templates"]),
        body=body,
    )


# ──────────────────────────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────────────────────────
class Retriever:
    """Loads `kb/*.md`, builds a TF-IDF index, and answers queries.

    Typical usage::

        r = Retriever()                         # loads kb/, builds index
        results = r.retrieve(species="dog",
                             life_stage="puppy",
                             query="exercise",
                             top_k=3)

    The metadata pre-filter (species + life_stage) runs *before*
    TF-IDF ranking. With a small structured KB, this is what makes
    classical IR competitive with neural retrieval — the candidate
    set drops from ~16 chunks to typically 4–6 before any scoring
    happens.
    """

    def __init__(self, kb_dir: Path | str | None = None):
        # Resolve the KB directory and load every doc into memory.
        self.kb_dir = Path(kb_dir) if kb_dir is not None else _DEFAULT_KB_DIR
        self.chunks: list[KBChunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None  # sparse TF-IDF matrix; rows align with self.chunks

        self._load()
        self._build_index()

        logger.info(
            "Retriever ready: %d chunks, %d TF-IDF features",
            len(self.chunks),
            len(self._vectorizer.vocabulary_),
        )

    # ── Loading ──
    def _load(self) -> None:
        """Read and parse every `kb/*.md` file (excluding README.md)."""
        if not self.kb_dir.is_dir():
            raise FileNotFoundError(f"KB directory not found: {self.kb_dir}")

        for path in sorted(self.kb_dir.glob("*.md")):
            if path.name.lower() == "readme.md":
                continue
            self.chunks.append(_parse_doc(path))

        if not self.chunks:
            raise ValueError(f"No KB documents found in {self.kb_dir}")

    # ── Index construction ──
    def _build_index(self) -> None:
        """Build the TF-IDF index over every chunk's searchable text.

        Bigrams are enabled so synonym phrases like 'young dog' index
        alongside the unigram 'puppy'. English stop-words are removed
        because filler words ('the', 'a', 'is') would otherwise
        dominate the cosine similarity on these short documents.
        """
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        texts = [c.searchable_text for c in self.chunks]
        self._matrix = self._vectorizer.fit_transform(texts)

    # ── Query ──
    def retrieve(
        self,
        species: str,
        life_stage: str,
        query: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return the chunks most relevant to a pet profile.

        Parameters
        ----------
        species
            Exact-match metadata filter. Chunks not matching are excluded.
        life_stage
            Exact-match metadata filter. Chunks not matching are excluded.
        query
            Optional free-text query. When given, the metadata-filtered
            candidates are ranked by TF-IDF cosine similarity. When
            omitted (or empty/whitespace), all candidates are returned
            with score 1.0, ordered by topic then id for determinism.
        top_k
            If set, truncate to this many results. ``None`` = no limit.
        """
        # ── 1. Metadata pre-filter ──
        # We keep both the chunk and its row index so we can slice the
        # TF-IDF matrix by row when a query is provided.
        candidates: list[tuple[int, KBChunk]] = [
            (i, c) for i, c in enumerate(self.chunks)
            if c.species == species and c.life_stage == life_stage
        ]
        if not candidates:
            logger.info(
                "No KB chunks for species=%r life_stage=%r",
                species, life_stage,
            )
            return []

        # ── 2. Score (TF-IDF cosine if query is given, else uniform) ──
        if query and query.strip():
            query_vec = self._vectorizer.transform([query])
            row_indices = [i for i, _ in candidates]
            candidate_matrix = self._matrix[row_indices]
            # cosine_similarity returns shape (1, n_candidates); take row 0.
            sims = cosine_similarity(query_vec, candidate_matrix)[0]
            scored = [
                RetrievalResult(chunk=c, score=float(s))
                for (_, c), s in zip(candidates, sims)
            ]
            # Sort by descending score; tie-break on id so the order
            # is fully deterministic when scores match exactly.
            scored.sort(key=lambda r: (-r.score, r.chunk.id))
        else:
            scored = [
                RetrievalResult(chunk=c, score=1.0)
                for _, c in candidates
            ]
            # Stable, human-readable order when no query is provided.
            scored.sort(key=lambda r: (r.chunk.topic, r.chunk.id))

        # ── 3. Apply top_k limit if asked ──
        if top_k is not None:
            scored = scored[:top_k]

        logger.info(
            "Retrieved %d chunks for species=%r life_stage=%r query=%r",
            len(scored), species, life_stage, query,
        )
        return scored
