# Model Card — PawPal+ AI

This card explains what PawPal+ AI is, what it's good at, what it's *not*
good at, and the honest story of how it was built. It's written in plain
English so anyone — not just an ML person — can follow along.

For the project overview and setup steps, see the
[README](README.md). For the technical architecture, see
[`assets/architecture.md`](assets/architecture.md).

---

## At a Glance

| Field | Value |
|---|---|
| **System name** | PawPal+ AI — A Retrieval-Augmented Pet Care Planner |
| **What it does** | Suggests a daily pet care plan grounded in a curated knowledge base |
| **Base project** | PawPal+ (Module 2 show-and-tell) — a manual pet task scheduler |
| **AI feature added** | Retrieval-Augmented Generation (RAG) |
| **Generation style** | Template-based, deterministic — no LLM, no API calls |
| **Retrieval style** | scikit-learn TF-IDF with metadata pre-filtering |
| **Knowledge base size** | 16 attributed pet-care documents in `kb/` |
| **Languages supported** | English |
| **Species covered** | Dog, cat, rabbit |
| **Life stages covered** | Puppy / kitten, adult, senior |
| **Runs offline?** | Yes. No API keys, no model downloads, no network at runtime |
| **Determinism** | Same pet profile + same KB → same plan, every run |

---

## What This System Is Meant For

**Intended use:** PawPal+ AI is a portfolio / class-project demonstration of
how to wire a retrieval-augmented planning layer onto an existing
deterministic system. It shows the moving parts of a small but realistic
applied-AI pipeline: a curated knowledge base, a retriever, a generator,
guardrails, structured logging, and an evaluation harness.

It's also a working pet care planner you can actually use to organize a
day for a dog, cat, or rabbit, as long as you treat the suggestions as
helpful starting points and not medical advice.

### What this system is **not** for

- **Not a substitute for a veterinarian.** Real pets have real medical
  histories, breed-specific needs, allergies, prescriptions, and quirks
  this system has no way of knowing about.
- **Not a knowledge source for species outside dog/cat/rabbit.** The
  guardrail will refuse other species rather than make something up,
  but the system has nothing useful to say about birds, fish, reptiles,
  ferrets, hamsters, or anything else.
- **Not for production deployment.** The KB is small (16 docs), curated
  by one person, and never peer-reviewed. The right way to scale this
  pattern in production would be a clinical knowledge source maintained
  by veterinary professionals, plus a real review process for content
  changes.
- **Not a chat system.** The output is a structured task list with
  citations. If you want a conversational AI assistant, this isn't that.

---

## How The System Works In One Paragraph

You enter your pet's species and age. The system maps that to a life
stage (e.g. age 0 dog → "puppy"). It then filters the knowledge base by
species + life stage — usually 2 to 6 documents survive that filter.
If you typed an optional focus query like "exercise walks", it ranks
those documents by keyword match. For each surviving document, the
system reads structured fields out of the document's "task templates"
(description, duration, frequency, time of day) and turns each one
into a suggested task carrying a citation back to the document. A
guardrail double-checks every citation against the documents that
were actually retrieved. You see the suggestions in the UI, click the
ones you like, and they get added to your schedule as ordinary tasks.

For the diagram and full data flow, see
[`assets/architecture.md`](assets/architecture.md).

---

## The Knowledge Base

The KB is the **single source of truth** for everything PawPal+ AI ever
suggests. It lives in `kb/` as 16 short markdown documents, each
covering one (species, life stage, topic) combination — for example
`dog_puppy_exercise.md` or `rabbit_adult_feeding.md`.

### What's in each document

- **Frontmatter (TOML)** with: id, species, life stage, topic, tags
  (synonyms for keyword matching), source organization, source URL,
  retrieval date, and one or more `task_templates` (each with
  description, duration, frequency, priority, suggested time of day).
- **Body paragraph** of 3–5 sentences in plain English, paraphrased
  from the cited source.

### Where the content comes from

Every document paraphrases publicly available pet-care guidance from
established organizations:

- ASPCA — Pet Care
- AVMA — Resources for Pet Owners
- AKC — Expert Advice
- Cornell Feline Health Center
- The Humane Society of the United States
- House Rabbit Society

Each document attributes the organization it draws from and links to
that organization's stable pet-care landing page. Wording is
summarized for educational use, not quoted verbatim.

### How the KB was built

One person (the project author) curated the KB over the course of
Phase 2 of the project. There was no clinical review, no editorial
process, and no automated validation beyond schema checks. **This is
the single biggest limitation of the system** — and the most honest
thing to say about it.

---

## How We Evaluated The System

Three layers of evidence:

### 1. Unit tests (111 passing)

Every module has its own test file. The split:

- **PawPal+ Core (the original code)**: 21 tests — unchanged from before
  this project started. Proof that the AI extension didn't break what
  was already working.
- **Retriever**: 18 tests covering loading, metadata filtering, query
  ranking, synonym recall, top-k limits, and edge cases.
- **Generator**: 14 tests covering fan-out, citation correctness,
  rationale formatting, determinism, and end-to-end integration.
- **RAG Planner**: 24 tests covering life-stage derivation, the
  orchestrator, conversion to PawPal+ Tasks, and the full pipeline.
- **Guardrails**: 34 tests covering input validation, the citation
  check, the confidence formula, and the structured logger — including
  edge cases like the `bool`-as-age trap and lazy log-directory
  creation.

Run with `pytest tests/ -v`. Total runtime: under one second.

### 2. Evaluation harness (35/35 checks across 8 cases)

The harness in `eval/run_eval.py` runs 8 sample pet profiles end-to-end
and checks each one against expected behaviors that encode real domain
knowledge:

- A puppy plan must include short walks (≤20 minutes — the standard
  "5 minutes per month of age" rule).
- A senior cat must get 3+ feeding tasks (the "smaller, more frequent
  meals" pattern from the cited sources).
- A rabbit plan must mention hay somewhere in the descriptions.
- An unsupported species (`hamster`) must produce zero suggestions —
  proving the guardrail actually rejects it.
- An invalid age (`-1`) must produce zero suggestions — same proof.

All 8 cases passed all their checks the first time the harness ran, and
the results are deterministic across runs.

### 3. Structured logging (`logs/pawpal.jsonl`)

Every retrieve, generate, guard, and reject event gets one JSON line.
Useful for auditing what happened on any past run, and for reproducing
problems if a user reports something unexpected.

---

## What Surprised Me During Testing

A few things were interesting enough to call out:

1. **TF-IDF was good enough.** Going in, I assumed the small KB would
   need neural embeddings (sentence-transformers) for the retriever to
   feel "smart." Once I added bigrams and synonym tags to the KB, plain
   TF-IDF turned out to be plenty — the synonym test (`young dog`
   query matching `dog_puppy_exercise`) passed without any model
   download. That saved about 1 GB of pip dependencies and many
   minutes of clone-and-run setup.

2. **Determinism made the eval harness much easier to write.** Because
   the same pet profile always produces the same plan, the eval can
   write equality assertions without flakiness. I expected to need
   "fuzzy" checks (something like *"at least one walk task should be
   short-ish"*) but I could write *"all exercise tasks for this puppy
   ≤20 minutes"* directly.

3. **The senior-cat case was the most interesting one.** The KB only
   covers feeding for senior cats — no play, no grooming. I worried
   the system would either pad the response with irrelevant chunks or
   feel "broken." Instead it returned three feeding tasks and stopped,
   which is the honest answer. I added that case to the eval harness
   *because* it surprised me, not in spite of it.

4. **The output guardrail had nothing to drop.** With Tier C, the
   template-based generator copies fields straight from the chunk,
   so it can't physically write an ungrounded citation. The guardrail
   ran on every call and rejected zero suggestions across all tests
   and eval cases. That's the right behavior — but it means the
   guardrail is essentially a *future-proofing* device for if/when a
   later version swaps in an LLM-based generator.

---

## Biases And Limitations

This is the part where I have to be most honest.

### KB-level biases

- **Small.** 16 documents is a tiny knowledge base. Real care decisions
  for a real pet need much more nuance than this can carry.
- **One curator, no peer review.** Every document was written by one
  person (me) over the course of one project phase. Real pet-care
  knowledge sources have editorial oversight; this one doesn't.
- **US-centric.** The cited organizations are mostly North American.
  Other regions have equally valid (and sometimes different)
  guidelines, especially around topics like pet population control,
  rabbit housing, and kitten weaning.
- **English-only.** No non-English content, no multilingual user
  testing.
- **No breed-specific nuance.** A 1-year-old chihuahua and a 1-year-old
  Great Dane both register as "adult dog" and get the same suggestions.
  In practice, those two animals have very different exercise and
  feeding needs.
- **Single-source-per-doc.** Each document credits one or two
  organizations. Real clinical knowledge sources synthesize across
  many studies and many groups.
- **Heuristics, not science.** Some of the rules baked in (the
  5-minutes-per-month-of-age rule for puppies, the 3-meals-a-day
  pattern for senior cats) are widely-cited rules of thumb, not
  controlled-trial findings. They're better than nothing, but they
  aren't medicine.

### System-level biases

- **Whole-year ages.** A 6-month-old puppy and an 11-month-old puppy
  both register as `age=0`. The KB's puppy advice is calibrated for
  both ends of that range, but a more granular age model (months,
  not years) would clearly help.
- **Conservative life-stage thresholds.** A 1-year-old dog is
  classified "adult" even though many large breeds are still
  juvenile. I chose to err on the side of adult-appropriate
  suggestions because most real-world large-breed owners already
  know to be cautious; the riskier failure mode would be calling a
  small-breed adult a puppy and under-feeding it.
- **Time-of-day suggestions are fixed.** Every "morning walk" lands
  at 07:00, every "evening walk" at 18:00. The user can move them in
  the UI, but the system has no awareness of the user's actual
  schedule.

---

## Could This Be Misused?

Yes, but I tried to make misuse difficult.

### Realistic misuse scenarios

- **Treating it as veterinary advice.** Someone might use the
  suggestions in place of a vet visit. This is the biggest risk, and
  it's why the disclaimer appears in the README, in `kb/README.md`,
  and in this card.
- **Mis-specifying age to get a different plan.** Entering `age=0` for
  an actual 5-year-old dog would produce puppy-style short walks,
  which under-exercises the dog. Mostly self-correcting (the user
  sees the plan), but possible.
- **Trusting it for an unsupported species.** Mitigated — the input
  validator refuses anything outside dog/cat/rabbit.

### What protects against misuse

- **Mandatory citations.** Every suggestion shows the source
  organization and a URL. Users can (and should) check the source.
- **Disclaimer everywhere.** Plain language, in three places: top of
  the README, top of the KB, and in this model card.
- **Human-in-the-loop UI.** No suggestion is silently committed to
  the schedule. The user reviews and clicks Accept.
- **Refuse rather than invent.** Unsupported species and invalid ages
  are rejected, not glossed over.
- **Auditable logs.** Every step is recorded in `logs/pawpal.jsonl`,
  so unexpected behavior can be traced.

---

## AI Collaboration During Development

This project was built in close collaboration with an AI coding
assistant (Claude). I'd estimate 70–80% of the actual code, tests, and
docs were written via that collaboration, with me directing the
high-level decisions, sequencing, and review.

The collaboration produced both very good moments and clearly flawed
moments. Two specific examples:

### A helpful AI suggestion

When I asked about offline simplicity, the AI proposed three tiers
(LLM via Ollama, sentence-transformers embeddings + templates, and
classical TF-IDF + templates). It then walked through the actual
download size and setup friction of each one — pointing out that
sentence-transformers would pull in roughly 1 GB of PyTorch on
`pip install`, on top of the 80 MB embedding model. That detail
was the deciding factor in choosing the lightest tier (TF-IDF) and
made the project's "clone and run in seconds" promise possible.

What was helpful wasn't just the recommendation — it was that the
AI **broke down the actual costs of each option in concrete units**
(megabytes, seconds, retries on flaky networks) instead of giving
a vague "this one is better." That kind of grounded comparison is
what made the trade-off easy to choose.

### A flawed AI suggestion

The first version of the architecture diagram (Phase 1) had me
specifying **Anthropic's hosted API** as the generator. The AI laid
out a nice plan with prompt templates, citation enforcement, and a
guardrail — but it didn't proactively flag the fact that this design
would require an API key, would cost money per call, and could
**not** run on a clone of the repo without setup.

I had to push back twice — first asking *"this won't use API keys
right?"* and then asking *"what gets downloaded with each tier?"* —
before we revised the architecture to the offline-first design that
ended up shipping. The AI was perfectly capable of doing the
offline design from the start; it just didn't *volunteer* the
trade-off when the constraint hadn't been stated yet.

The lesson I took from this: AI collaborators are great at executing
within constraints once the constraints are explicit, but they don't
always notice **missing** constraints unless prompted. The
human partner has to do the work of stating goals up front
(*"this needs to run offline"*) rather than assuming the assistant
will infer them from context. That's something I want to do more
deliberately on future projects.

---

## What I'd Do Next

If I were continuing this project beyond a class scope, the most
valuable improvements would be (in rough priority order):

1. **Expand and review the KB.** The single biggest lever. More
   species (birds, fish, common reptiles, ferrets), breed-specific
   variations for dogs and cats, multilingual versions, and crucially
   a real editorial review process.
2. **Fractional ages.** Replace the integer `age` field with months
   or a (years, months) pair so the puppy/adult cutoff isn't a
   1-year cliff.
3. **User feedback loop.** Add a "this suggestion isn't right for my
   pet" button that writes a structured event to the log. Over time,
   that data would surface KB gaps directly.
4. **Add an LLM behind the Generator interface as an optional path.**
   Keep TF-IDF retrieval and the citation guardrail; let users who
   have Ollama installed get natural-language explanations on top.
   The interface boundary in `src/generator.py` is already designed
   for this swap.
5. **Test KB content correctness, not just shape.** Right now the
   Phase 2 validator confirms the schema; it doesn't sanity-check
   the medical guidance itself. A small set of "expected ground
   truth" tests against authoritative sources would help.
6. **Internationalize the cited sources.** Add WSAVA, RSPCA, and
   regional veterinary associations so users outside North America
   see locally-relevant guidance.

---

## Final Caveat

PawPal+ AI is a class project. Every piece of advice it surfaces was
paraphrased by a non-expert from public guidance written for a
general audience. **It is not veterinary advice, and no part of this
repository should be treated as a substitute for talking to a
veterinarian about your actual pet.**

If you're building on this codebase and you want any of it to inform
real care decisions for real animals, the absolute prerequisite is
replacing the KB with a clinically-reviewed knowledge source under
the supervision of licensed veterinary professionals.
