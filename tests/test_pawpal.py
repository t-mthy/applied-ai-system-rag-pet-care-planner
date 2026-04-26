"""
PawPal+ Test Suite
──────────────────
Tests to verify core system behavior, algorithmic features, and edge cases.
Run with:  pytest tests/ -v
"""

import sys
import os

# Add the project root to the path so we can import the src/ package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pawpal_system import Task, Pet, Owner, Scheduler


# ──────────────────────────────────────────────
# Helper: quickly build a small test world
# ──────────────────────────────────────────────
def _make_test_setup():
    """Create an Owner with two Pets and a few Tasks for reuse in tests."""
    owner = Owner(name="Jordan")
    mochi = Pet(name="Mochi", species="dog", age=3)
    whiskers = Pet(name="Whiskers", species="cat", age=5)
    owner.add_pet(mochi)
    owner.add_pet(whiskers)
    return owner, mochi, whiskers


# ══════════════════════════════════════════════
# CORE CLASS TESTS
# ══════════════════════════════════════════════

def test_task_mark_complete():
    """Calling mark_complete() should flip completed from False to True."""
    task = Task(
        description="Morning walk",
        due_date="2026-03-29", due_time="07:00",
        duration_minutes=30, priority="high"
    )
    # Task starts as not completed
    assert task.completed is False
    # After marking complete, it should be True
    task.mark_complete()
    assert task.completed is True


def test_task_is_recurring():
    """is_recurring() should return True for daily/weekly, False for once."""
    daily = Task(
        description="Walk", due_date="2026-03-29", due_time="07:00",
        duration_minutes=30, priority="high", frequency="daily"
    )
    weekly = Task(
        description="Grooming", due_date="2026-03-29", due_time="10:00",
        duration_minutes=60, priority="low", frequency="weekly"
    )
    once = Task(
        description="Vet visit", due_date="2026-03-29", due_time="14:00",
        duration_minutes=45, priority="high", frequency="once"
    )
    assert daily.is_recurring() is True
    assert weekly.is_recurring() is True
    assert once.is_recurring() is False


def test_pet_add_task_increases_count():
    """Adding a task to a Pet should increase the pet's task count."""
    pet = Pet(name="Mochi", species="dog", age=3)
    # Pet starts with zero tasks
    assert len(pet.get_tasks()) == 0

    # Add one task — count should become 1
    pet.add_task(Task(
        description="Walk", due_date="2026-03-29", due_time="07:00",
        duration_minutes=30, priority="high"
    ))
    assert len(pet.get_tasks()) == 1

    # Add another — count should become 2
    pet.add_task(Task(
        description="Feeding", due_date="2026-03-29", due_time="08:00",
        duration_minutes=10, priority="medium"
    ))
    assert len(pet.get_tasks()) == 2


def test_pet_add_task_stamps_pet_name():
    """Adding a task to a pet should auto-set the task's pet_name."""
    pet = Pet(name="Whiskers", species="cat", age=5)
    task = Task(
        description="Feeding", due_date="2026-03-29", due_time="08:00",
        duration_minutes=10, priority="high"
    )
    # Before adding, pet_name is empty
    assert task.pet_name == ""
    pet.add_task(task)
    # After adding, pet_name should match the pet
    assert task.pet_name == "Whiskers"


def test_pet_get_pending_tasks():
    """get_pending_tasks() should only return tasks that are not completed."""
    pet = Pet(name="Mochi", species="dog", age=3)
    task_a = Task(
        description="Walk", due_date="2026-03-29", due_time="07:00",
        duration_minutes=30, priority="high"
    )
    task_b = Task(
        description="Feeding", due_date="2026-03-29", due_time="08:00",
        duration_minutes=10, priority="medium"
    )
    pet.add_task(task_a)
    pet.add_task(task_b)

    # Both are pending at first
    assert len(pet.get_pending_tasks()) == 2

    # Complete one — only one should remain pending
    task_a.mark_complete()
    assert len(pet.get_pending_tasks()) == 1
    assert pet.get_pending_tasks()[0].description == "Feeding"


def test_owner_get_all_tasks_across_pets():
    """get_all_tasks() should collect tasks from every pet."""
    owner, mochi, whiskers = _make_test_setup()

    mochi.add_task(Task(
        description="Walk", due_date="2026-03-29", due_time="07:00",
        duration_minutes=30, priority="high"
    ))
    whiskers.add_task(Task(
        description="Feeding", due_date="2026-03-29", due_time="08:00",
        duration_minutes=10, priority="high"
    ))
    whiskers.add_task(Task(
        description="Litter box", due_date="2026-03-29", due_time="12:00",
        duration_minutes=10, priority="medium"
    ))

    # Owner should see all 3 tasks combined
    assert len(owner.get_all_tasks()) == 3


# ══════════════════════════════════════════════
# SORTING TESTS
# ══════════════════════════════════════════════

def test_sort_by_time_returns_chronological_order():
    """Tasks should come back sorted earliest to latest by due_time."""
    owner, mochi, whiskers = _make_test_setup()
    scheduler = Scheduler(owner)

    # Create tasks intentionally out of order
    tasks = [
        Task(description="Evening walk", due_date="2026-03-29",
             due_time="17:00", duration_minutes=30, priority="medium"),
        Task(description="Morning walk", due_date="2026-03-29",
             due_time="07:00", duration_minutes=30, priority="high"),
        Task(description="Lunch feeding", due_date="2026-03-29",
             due_time="12:00", duration_minutes=10, priority="medium"),
    ]

    sorted_tasks = scheduler.sort_by_time(tasks)

    # Verify the order is 07:00 -> 12:00 -> 17:00
    assert sorted_tasks[0].due_time == "07:00"
    assert sorted_tasks[1].due_time == "12:00"
    assert sorted_tasks[2].due_time == "17:00"


def test_sort_by_priority_returns_high_first():
    """Tasks should come back ordered high -> medium -> low."""
    owner, mochi, whiskers = _make_test_setup()
    scheduler = Scheduler(owner)

    tasks = [
        Task(description="Enrichment toy", due_date="2026-03-29",
             due_time="15:00", duration_minutes=20, priority="low"),
        Task(description="Medication", due_date="2026-03-29",
             due_time="09:00", duration_minutes=5, priority="high"),
        Task(description="Grooming", due_date="2026-03-29",
             due_time="11:00", duration_minutes=30, priority="medium"),
    ]

    sorted_tasks = scheduler.sort_by_priority(tasks)

    assert sorted_tasks[0].priority == "high"
    assert sorted_tasks[1].priority == "medium"
    assert sorted_tasks[2].priority == "low"


# ══════════════════════════════════════════════
# RECURRENCE TESTS
# ══════════════════════════════════════════════

def test_daily_recurrence_creates_next_day_task():
    """Completing a daily task should create a new task for tomorrow."""
    owner, mochi, _ = _make_test_setup()
    scheduler = Scheduler(owner)

    daily_walk = Task(
        description="Morning walk", due_date="2026-03-29", due_time="07:00",
        duration_minutes=30, priority="high", frequency="daily"
    )
    mochi.add_task(daily_walk)

    # Complete it through the scheduler (which handles recurrence)
    new_task = scheduler.mark_task_complete(daily_walk)

    # Original should be completed
    assert daily_walk.completed is True
    # New task should exist with tomorrow's date
    assert new_task is not None
    assert new_task.due_date == "2026-03-30"
    assert new_task.completed is False
    # Mochi should now have 2 tasks (original + new)
    assert len(mochi.get_tasks()) == 2


def test_weekly_recurrence_creates_next_week_task():
    """Completing a weekly task should create a new task 7 days later."""
    owner, _, whiskers = _make_test_setup()
    scheduler = Scheduler(owner)

    grooming = Task(
        description="Grooming", due_date="2026-03-29", due_time="10:00",
        duration_minutes=60, priority="low", frequency="weekly"
    )
    whiskers.add_task(grooming)

    new_task = scheduler.mark_task_complete(grooming)

    assert new_task is not None
    # 2026-03-29 + 7 days = 2026-04-05
    assert new_task.due_date == "2026-04-05"
    assert new_task.completed is False


def test_non_recurring_task_does_not_create_new():
    """Completing a one-time task should NOT create a follow-up."""
    owner, mochi, _ = _make_test_setup()
    scheduler = Scheduler(owner)

    vet_visit = Task(
        description="Vet visit", due_date="2026-03-29", due_time="14:00",
        duration_minutes=45, priority="high", frequency="once"
    )
    mochi.add_task(vet_visit)

    new_task = scheduler.mark_task_complete(vet_visit)

    # Original is completed, but no new task was created
    assert vet_visit.completed is True
    assert new_task is None
    # Mochi should still have only the 1 original task
    assert len(mochi.get_tasks()) == 1


# ══════════════════════════════════════════════
# CONFLICT DETECTION TESTS
# ══════════════════════════════════════════════

def test_detect_conflicts_flags_overlapping_tasks():
    """Two tasks that overlap in time should produce a conflict warning."""
    owner, mochi, whiskers = _make_test_setup()
    scheduler = Scheduler(owner)

    # Task A: 07:00 for 30 min -> ends at 07:30
    # Task B: 07:15 for 10 min -> starts before A ends
    tasks = [
        Task(description="Morning walk", due_date="2026-03-29",
             due_time="07:00", duration_minutes=30, priority="high",
             pet_name="Mochi"),
        Task(description="Breakfast feeding", due_date="2026-03-29",
             due_time="07:15", duration_minutes=10, priority="high",
             pet_name="Whiskers"),
    ]

    warnings = scheduler.detect_conflicts(tasks)

    # Should find exactly 1 conflict
    assert len(warnings) == 1
    assert "Morning walk" in warnings[0]
    assert "Breakfast feeding" in warnings[0]


def test_detect_conflicts_no_overlap():
    """Tasks that don't overlap should produce zero warnings."""
    owner, mochi, whiskers = _make_test_setup()
    scheduler = Scheduler(owner)

    # Task A: 07:00 for 30 min -> ends at 07:30
    # Task B: 08:00 -> starts well after A ends
    tasks = [
        Task(description="Morning walk", due_date="2026-03-29",
             due_time="07:00", duration_minutes=30, priority="high",
             pet_name="Mochi"),
        Task(description="Feeding", due_date="2026-03-29",
             due_time="08:00", duration_minutes=10, priority="high",
             pet_name="Whiskers"),
    ]

    warnings = scheduler.detect_conflicts(tasks)
    assert len(warnings) == 0


# ══════════════════════════════════════════════
# EDGE CASE TESTS
# ══════════════════════════════════════════════

def test_pet_with_no_tasks():
    """A pet with no tasks should return empty lists, not errors."""
    pet = Pet(name="Buddy", species="dog", age=1)

    assert len(pet.get_tasks()) == 0
    assert len(pet.get_pending_tasks()) == 0


def test_sort_empty_list():
    """Sorting an empty task list should return an empty list, not crash."""
    owner = Owner(name="Jordan")
    scheduler = Scheduler(owner)

    assert scheduler.sort_by_time([]) == []
    assert scheduler.sort_by_priority([]) == []


def test_detect_conflicts_empty_list():
    """Conflict detection on an empty list should return no warnings."""
    owner = Owner(name="Jordan")
    scheduler = Scheduler(owner)

    assert scheduler.detect_conflicts([]) == []


# ══════════════════════════════════════════════
# NEXT AVAILABLE SLOT TESTS
# ══════════════════════════════════════════════

def test_find_slot_in_morning_gap():
    """Should find a slot in the gap before the first task of the day."""
    owner = Owner(name="Jordan")
    scheduler = Scheduler(owner)

    # One task at 08:00 — there's a gap from 06:00 to 08:00 (120 min)
    tasks = [
        Task(description="Walk", due_date="2026-03-29",
             due_time="08:00", duration_minutes=30, priority="high"),
    ]

    # A 60-min slot should fit at 06:00 (start of schedulable window)
    slot = scheduler.find_next_available_slot(tasks, 60, "2026-03-29")
    assert slot == "06:00"


def test_find_slot_between_tasks():
    """Should find a slot in a gap between two tasks."""
    owner = Owner(name="Jordan")
    scheduler = Scheduler(owner)

    # Task A: 06:00-06:30, Task B: 09:00-09:30
    # Gap between them: 06:30 to 09:00 = 150 min
    tasks = [
        Task(description="Feeding", due_date="2026-03-29",
             due_time="06:00", duration_minutes=30, priority="high"),
        Task(description="Grooming", due_date="2026-03-29",
             due_time="09:00", duration_minutes=30, priority="medium"),
    ]

    # A 60-min slot should fit at 06:30 (right after the first task)
    slot = scheduler.find_next_available_slot(tasks, 60, "2026-03-29")
    assert slot == "06:30"


def test_find_slot_after_last_task():
    """Should find a slot after all tasks if that's where the gap is."""
    owner = Owner(name="Jordan")
    scheduler = Scheduler(owner)

    # Tasks packed from 06:00 to 08:00, nothing after
    tasks = [
        Task(description="Walk", due_date="2026-03-29",
             due_time="06:00", duration_minutes=60, priority="high"),
        Task(description="Feeding", due_date="2026-03-29",
             due_time="07:00", duration_minutes=60, priority="high"),
    ]

    # A 60-min slot should fit at 08:00 (after the last task ends)
    slot = scheduler.find_next_available_slot(tasks, 60, "2026-03-29")
    assert slot == "08:00"


def test_find_slot_returns_none_when_day_is_full():
    """Should return None if no gap is large enough."""
    owner = Owner(name="Jordan")
    scheduler = Scheduler(owner)

    # One giant task that fills the entire schedulable window (06:00-22:00)
    tasks = [
        Task(description="All-day event", due_date="2026-03-29",
             due_time="06:00", duration_minutes=960, priority="high"),
    ]

    # No room for even a 30-min slot
    slot = scheduler.find_next_available_slot(tasks, 30, "2026-03-29")
    assert slot is None


def test_find_slot_empty_day():
    """On an empty day, the slot should start at the beginning of the window."""
    owner = Owner(name="Jordan")
    scheduler = Scheduler(owner)

    # No tasks at all — entire day is open
    slot = scheduler.find_next_available_slot([], 45, "2026-03-29")
    assert slot == "06:00"
