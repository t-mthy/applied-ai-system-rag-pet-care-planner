"""
PawPal+ CLI Demo Script
────────────────────────
This script is a "playground" that creates sample data and
prints a readable daily schedule to the terminal.
It verifies that our backend logic works before we touch the UI.
"""

from src.pawpal_system import Task, Pet, Owner, Scheduler
from src.rag_planner import (
    apply_suggestions_to_pet,
    derive_life_stage,
    suggest_tasks_for_pet,
)
from src.retriever import Retriever
from datetime import date


def main():
    # ── Today's date as a string (used for all tasks below) ──
    today = date.today().isoformat()  # e.g. "2026-03-27"

    # ──────────────────────────────────────────
    # Step 1: Create an Owner
    # ──────────────────────────────────────────
    owner = Owner(name="Jordan")
    print(f"Owner created: {owner.name}\n")

    # ──────────────────────────────────────────
    # Step 2: Create two Pets and register them
    # ──────────────────────────────────────────
    mochi = Pet(name="Mochi", species="dog", age=3)
    whiskers = Pet(name="Whiskers", species="cat", age=5)

    owner.add_pet(mochi)
    owner.add_pet(whiskers)
    print(f"Pets registered: {mochi.name} ({mochi.species}), "
          f"{whiskers.name} ({whiskers.species})\n")

    # ──────────────────────────────────────────
    # Step 3: Add Tasks (intentionally out of order)
    #         Two tasks overlap to test conflict detection
    # ──────────────────────────────────────────
    # Mochi's tasks
    mochi.add_task(Task(
        description="Evening walk",
        due_date=today, due_time="17:30",
        duration_minutes=30, priority="medium"
    ))
    mochi.add_task(Task(
        description="Morning walk",
        due_date=today, due_time="07:00",
        duration_minutes=30, priority="high", frequency="daily"
    ))
    mochi.add_task(Task(
        description="Flea medication",
        due_date=today, due_time="09:00",
        duration_minutes=5, priority="high"
    ))

    # Whiskers' tasks — Breakfast at 07:15 overlaps with Mochi's 07:00-07:30
    whiskers.add_task(Task(
        description="Breakfast feeding",
        due_date=today, due_time="07:15",
        duration_minutes=10, priority="high", frequency="daily"
    ))
    whiskers.add_task(Task(
        description="Litter box cleaning",
        due_date=today, due_time="12:00",
        duration_minutes=10, priority="medium"
    ))

    # ──────────────────────────────────────────
    # Step 4: Build and display the daily schedule
    # ──────────────────────────────────────────
    scheduler = Scheduler(owner)
    schedule = scheduler.get_daily_schedule(today)

    print("=" * 58)
    print(f"  PawPal+ Daily Schedule — {today}")
    print("=" * 58)
    print(f"  {'Time':<8} {'Task':<22} {'Pet':<10} {'Priority':<8}")
    print("-" * 58)

    for task in schedule:
        print(f"  {task.due_time:<8} {task.description:<22} "
              f"{task.pet_name:<10} {task.priority:<8}")

    print("=" * 58)
    print(f"  Total tasks: {len(schedule)}")
    print()

    # ──────────────────────────────────────────
    # Step 5: Conflict Detection
    # ──────────────────────────────────────────
    conflicts = scheduler.detect_conflicts(schedule)

    if conflicts:
        print("!! Schedule Conflicts Detected:")
        for warning in conflicts:
            print(f"   -> {warning}")
        print()
    else:
        print("No scheduling conflicts found.\n")

    # ──────────────────────────────────────────
    # Step 6: Mark a recurring task complete
    #         and show the auto-generated next occurrence
    # ──────────────────────────────────────────
    morning_walk = schedule[0]  # "Morning walk" at 07:00 (daily)
    print(f"Completing recurring task: \"{morning_walk.description}\"")
    new_task = scheduler.mark_task_complete(morning_walk)

    if new_task:
        print(f"   -> Completed for {morning_walk.due_date}")
        print(f"   -> Next occurrence auto-created for {new_task.due_date}")
    print()

    # Show Mochi's updated tasks (should now include tomorrow's walk)
    print(f"Mochi's tasks after recurrence:")
    for t in mochi.get_tasks():
        print(f"   {t}")
    print()

    # ──────────────────────────────────────────
    # Step 7: Filter and sort demos
    # ──────────────────────────────────────────
    all_tasks = owner.get_all_tasks()

    # Filter: only pending tasks
    pending = scheduler.filter_by_status(all_tasks, completed=False)
    print(f"Pending tasks across all pets: {len(pending)}")

    # Filter: only Whiskers' tasks
    whiskers_tasks = scheduler.filter_by_pet(all_tasks, "Whiskers")
    print(f"Whiskers' tasks: {len(whiskers_tasks)}")

    # Sort by priority
    by_priority = scheduler.sort_by_priority(pending)
    print(f"\nPending tasks sorted by priority:")
    for t in by_priority:
        print(f"   {t}")

    # ──────────────────────────────────────────
    # Step 8: Find next available slot
    # ──────────────────────────────────────────
    print("\n" + "=" * 58)
    print("  Next Available Slot Finder")
    print("=" * 58)

    all_today = scheduler.get_daily_schedule(today)

    # Try to find a 45-minute slot
    slot_45 = scheduler.find_next_available_slot(all_today, 45, today)
    if slot_45:
        print(f"  45-min slot available at: {slot_45}")
    else:
        print("  No 45-min slot available today.")

    # Try to find a 2-hour slot
    slot_120 = scheduler.find_next_available_slot(all_today, 120, today)
    if slot_120:
        print(f"  2-hour slot available at:  {slot_120}")
    else:
        print("  No 2-hour slot available today.")

    print("=" * 58)
    print()

    # ──────────────────────────────────────────
    # Step 9: RAG demo — grounded AI suggestions
    # ──────────────────────────────────────────
    # Add a third pet (a puppy) so the demo exercises a different
    # life_stage band and shows the planner adapting to it.
    print("=" * 58)
    print("  PawPal+ AI — RAG Suggestion Demo")
    print("=" * 58)

    # Build the Retriever once and reuse for both pets (the TF-IDF
    # index isn't free to construct, even if it's cheap).
    retriever = Retriever()

    biscuit = Pet(name="Biscuit", species="dog", age=0)  # puppy
    owner.add_pet(biscuit)
    print(f"  Added puppy {biscuit.name} ({biscuit.species}, age {biscuit.age}).")
    print()

    for demo_pet in [biscuit, whiskers]:
        stage = derive_life_stage(demo_pet.species, demo_pet.age)
        print("-" * 58)
        print(f"  Pet: {demo_pet.name} — {demo_pet.species}, "
              f"age {demo_pet.age} → life_stage={stage!r}")
        print("-" * 58)

        # Default call: no query, all matching chunks.
        suggestions = suggest_tasks_for_pet(demo_pet, retriever=retriever)

        if not suggestions:
            print("  (no KB coverage for this pet)")
            print()
            continue

        # Show the suggested day, sorted by time, with citations.
        print(f"  {len(suggestions)} grounded suggestion(s):")
        for s in sorted(suggestions, key=lambda x: x.suggested_time):
            print(
                f"    {s.suggested_time}  "
                f"{s.description:<35}  "
                f"prio={s.priority:<6}  "
                f"[{s.source_id}]"
            )
        print()

        # Show one full rationale to demonstrate UI-style citation.
        first = sorted(suggestions, key=lambda x: x.suggested_time)[0]
        print(f"  Sample rationale (first suggestion):")
        print(f"    {first.rationale}")
        print(f"    (See: {first.source_url})")
        print()

    # ──────────────────────────────────────────
    # Step 10: Convert suggestions → real Tasks for one pet
    # ──────────────────────────────────────────
    print("-" * 58)
    print("  Converting Biscuit's suggestions into real Tasks")
    print("-" * 58)

    biscuit_suggestions = suggest_tasks_for_pet(biscuit, retriever=retriever)
    added = apply_suggestions_to_pet(biscuit, biscuit_suggestions)
    print(f"  Added {len(added)} AI-suggested task(s) to {biscuit.name}.")

    # Re-run the schedule to show RAG-added tasks coexisting with
    # the user-entered ones from earlier in the demo.
    print()
    print("  Updated full-day schedule (all pets, including Biscuit):")
    full_schedule = scheduler.get_daily_schedule(today)
    for t in full_schedule:
        print(f"    {t.due_time}  {t.description:<35}  ({t.pet_name})")
    print()

    print("=" * 58)


if __name__ == "__main__":
    main()
