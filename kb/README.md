# Knowledge Base — `kb/`

This folder is the curated pet-care knowledge base that the Retriever
searches over. Every recommendation PawPal+ AI ever surfaces to the
user must come from one of these documents.

## Scope of this KB

- **Species covered:** dog, cat, rabbit.
- **Life stages covered:** dog and cat have puppy/kitten, adult, senior.
  Rabbit has adult only.
- **Topics covered:** feeding, exercise / play, grooming, enrichment.
- **Size:** 16 documents. Small on purpose — this is a class project,
  not a clinical knowledge source.

## Important disclaimer

The content in these documents is **paraphrased and summarized** from
publicly available pet-care guidance from established organizations
(ASPCA, AVMA, AKC, Cornell Feline Health Center, the Humane Society,
House Rabbit Society). Every document attributes the organization it
draws from and links to that organization's public pet-care section.

**This is a class project. Nothing in this knowledge base is
veterinary advice.** For real care decisions about a real animal,
consult a licensed veterinarian.

## Document schema

Every `kb/*.md` file uses the same shape:

````markdown
+++
id = "<species>_<life_stage>_<topic>"
species = "dog"            # one of: dog, cat, rabbit
life_stage = "puppy"       # one of: puppy, kitten, adult, senior
topic = "exercise"         # one of: feeding, exercise, play, grooming, enrichment
tags = ["puppy", "young dog", "walks", "play"]
source = "AKC puppy exercise guidance (illustrative summary)"
source_url = "https://www.akc.org/expert-advice/"
retrieved_on = "2026-04-26"

[[task_templates]]
description = "Short structured walk"
duration_minutes = 15
frequency = "daily"
priority = "high"
suggested_time = "08:00"

[[task_templates]]
description = "Short structured walk"
duration_minutes = 15
frequency = "daily"
priority = "medium"
suggested_time = "17:00"
+++

# <human-readable title>

Body paragraph(s) — 3–5 sentences explaining the recommendation.
This is the text the TF-IDF retriever ranks against. Keep it
keyword-rich (synonyms are good — "puppy", "young dog", "juvenile")
and grounded in the cited source.
````

### Why TOML (not YAML)

Python 3.11+ has `tomllib` in the standard library. Using TOML lets the
Retriever parse frontmatter without adding a `pyyaml` dependency, which
keeps the project's offline-first / zero-extra-deps promise. The `+++`
delimiter (Hugo / Zola convention) signals "this is TOML, not YAML."

### Required fields

| Field | Meaning |
|---|---|
| `id` | Stable identifier used as the citation `source_id`. Must match the filename stem. |
| `species` | Used by the metadata pre-filter in the Retriever. |
| `life_stage` | Used by the metadata pre-filter in the Retriever. |
| `topic` | Coarse category for grouping in eval and UI. |
| `tags` | Synonym list — boosts TF-IDF recall (e.g. "puppy" ↔ "young dog"). |
| `source` | Human-readable attribution. |
| `source_url` | Stable landing page for the cited organization. |
| `retrieved_on` | ISO date the summary was written / the source was last consulted. |

### `[[task_templates]]` array

Each entry becomes one suggested `Task` when this chunk is selected.
A doc may have one or many. Fields:

| Field | Type | Required | Notes |
|---|---|---|---|
| `description` | string | yes | Becomes `Task.description`. |
| `duration_minutes` | int | yes | Becomes `Task.duration_minutes`. |
| `frequency` | "one-time" / "daily" / "weekly" | yes | Becomes `Task.frequency`. |
| `priority` | "low" / "medium" / "high" | yes | Becomes `Task.priority`. |
| `suggested_time` | "HH:MM" | yes | Default time slot; user can move it in the UI. |

## Adding a new document

1. Create `kb/<species>_<life_stage>_<topic>.md`.
2. Fill in the frontmatter block following the schema above.
3. Write a short body paragraph paraphrased from a real, attributed
   source. Include synonyms for likely user vocabulary.
4. Run `pytest tests/test_retriever.py` (Phase 3+) — the retriever's
   parser will reject malformed frontmatter at load time.

The retriever rebuilds its TF-IDF index on every startup, so dropping
a new file in `kb/` is the only step needed for it to become
searchable.
