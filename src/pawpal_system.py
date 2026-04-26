"""
PawPal+ System — Core Logic Layer
Classes: Task, Pet, Owner, Scheduler
"""

from dataclasses import dataclass, field
from datetime import date, timedelta


# ──────────────────────────────────────────────
# Task — represents a single pet care activity
# ──────────────────────────────────────────────
@dataclass
class Task:
    description: str            # what the task is (e.g. "Morning walk")
    due_date: str               # date string in "YYYY-MM-DD" format
    due_time: str               # time string in "HH:MM" format
    duration_minutes: int       # how long the task takes
    priority: str               # "low", "medium", or "high"
    frequency: str = "once"     # "once", "daily", or "weekly"
    completed: bool = False     # whether the task is done
    pet_name: str = ""          # which pet this task belongs to

    def mark_complete(self):
        """Mark this task as completed."""
        self.completed = True

    def is_recurring(self):
        """Check if this task repeats (daily or weekly)."""
        return self.frequency in ("daily", "weekly")

    def __repr__(self):
        """Show a short, readable summary of this task."""
        # Build a status tag like "[x]" for done or "[ ]" for pending
        status = "[x]" if self.completed else "[ ]"
        return (
            f"{status} {self.due_time} | {self.description} "
            f"({self.pet_name}, {self.priority}, {self.duration_minutes}min)"
        )


# ──────────────────────────────────────────────
# Pet — stores pet info and its list of tasks
# ──────────────────────────────────────────────
@dataclass
class Pet:
    name: str                   # pet's name
    species: str                # "dog", "cat", etc.
    age: int                    # pet's age in years
    tasks: list = field(default_factory=list)  # list of Task objects

    def add_task(self, task):
        """Add a new task to this pet's task list."""
        # Stamp the task with this pet's name so we know who it belongs to
        task.pet_name = self.name
        self.tasks.append(task)

    def get_tasks(self):
        """Return all tasks for this pet."""
        return list(self.tasks)

    def get_pending_tasks(self):
        """Return only tasks that are not yet completed."""
        return [task for task in self.tasks if not task.completed]

    def remove_task(self, description):
        """Remove a task by its description. Returns True if found."""
        for task in self.tasks:
            if task.description == description:
                self.tasks.remove(task)
                return True
        return False


# ──────────────────────────────────────────────
# Owner — manages one or more pets
# ──────────────────────────────────────────────
@dataclass
class Owner:
    name: str                   # owner's name
    pets: list = field(default_factory=list)  # list of Pet objects

    def add_pet(self, pet):
        """Add a pet to the owner's list."""
        self.pets.append(pet)

    def get_pet(self, name):
        """Find and return a pet by name, or None if not found."""
        for pet in self.pets:
            if pet.name == name:
                return pet
        return None

    def get_all_tasks(self):
        """Gather all tasks from every pet into one list."""
        all_tasks = []
        for pet in self.pets:
            all_tasks.extend(pet.get_tasks())
        return all_tasks


# ──────────────────────────────────────────────
# Scheduler — the "brain" that organizes tasks
# ──────────────────────────────────────────────
class Scheduler:
    def __init__(self, owner):
        """Create a scheduler linked to an owner."""
        self.owner = owner      # the Owner whose pets we manage

    def get_daily_schedule(self, target_date):
        """Get all tasks for a specific date, sorted by time."""
        # Grab every task across all pets
        all_tasks = self.owner.get_all_tasks()
        # Keep only tasks that match the target date
        daily = [t for t in all_tasks if t.due_date == target_date]
        # Return them sorted earliest-first
        return self.sort_by_time(daily)

    def sort_by_time(self, tasks):
        """Sort a list of tasks by their due_time (earliest first)."""
        # "HH:MM" strings sort correctly in alphabetical order
        return sorted(tasks, key=lambda t: t.due_time)

    def sort_by_priority(self, tasks):
        """Sort tasks by priority (high -> medium -> low)."""
        # Map each priority level to a number so "high" comes first
        priority_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(tasks, key=lambda t: priority_order.get(t.priority, 3))

    def filter_by_status(self, tasks, completed):
        """Filter tasks by completion status (True or False)."""
        return [t for t in tasks if t.completed == completed]

    def filter_by_pet(self, tasks, pet_name):
        """Filter tasks to only show ones for a specific pet."""
        return [t for t in tasks if t.pet_name == pet_name]

    def detect_conflicts(self, tasks):
        """Find tasks that overlap in time and return warning messages."""
        warnings = []

        # Sort tasks by start time so we can compare neighbors
        sorted_tasks = self.sort_by_time(tasks)

        # Walk through each pair of consecutive tasks
        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]

            # Calculate when the current task ends
            # Convert "HH:MM" to total minutes, add duration, compare
            current_start = self._time_to_minutes(current.due_time)
            current_end = current_start + current.duration_minutes
            next_start = self._time_to_minutes(next_task.due_time)

            # If the current task hasn't finished before the next one starts,
            # that's a conflict (overlap)
            if current_end > next_start:
                warnings.append(
                    f"Conflict: \"{current.description}\" ({current.pet_name}) "
                    f"ends at {self._minutes_to_time(current_end)}, but "
                    f"\"{next_task.description}\" ({next_task.pet_name}) "
                    f"starts at {next_task.due_time}"
                )

        return warnings

    def handle_recurrence(self, task):
        """If a task is recurring, create the next occurrence."""
        # Only process tasks that actually repeat
        if not task.is_recurring():
            return None

        # Parse the current due date string into a date object
        current_date = date.fromisoformat(task.due_date)

        # Calculate the next due date based on frequency
        if task.frequency == "daily":
            next_date = current_date + timedelta(days=1)
        elif task.frequency == "weekly":
            next_date = current_date + timedelta(weeks=1)
        else:
            return None

        # Create a fresh copy of the task with the new date
        new_task = Task(
            description=task.description,
            due_date=next_date.isoformat(),
            due_time=task.due_time,
            duration_minutes=task.duration_minutes,
            priority=task.priority,
            frequency=task.frequency,
            completed=False,           # new occurrence starts incomplete
            pet_name=task.pet_name,
        )
        return new_task

    def mark_task_complete(self, task):
        """Mark a task done and auto-schedule the next one if it recurs."""
        # Mark the original task as completed
        task.mark_complete()

        # If it's recurring, create the next occurrence
        new_task = self.handle_recurrence(task)
        if new_task:
            # Find which pet owns this task and add the new one to them
            pet = self.owner.get_pet(new_task.pet_name)
            if pet:
                pet.add_task(new_task)
        return new_task

    # ── Helper methods for time math ──

    def _time_to_minutes(self, time_str):
        """Convert 'HH:MM' string to total minutes since midnight."""
        hours, minutes = time_str.split(":")
        return int(hours) * 60 + int(minutes)

    def _minutes_to_time(self, total_minutes):
        """Convert total minutes since midnight back to 'HH:MM' string."""
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}"

    def find_next_available_slot(self, tasks, duration, target_date):
        """Find the next open time slot of a given duration.

        Scans the day from 06:00 to 22:00 and looks for a gap between
        existing tasks that is large enough to fit the requested duration.
        Returns the start time as a "HH:MM" string, or None if no slot fits.
        """
        # Define the window we consider "schedulable" (6 AM to 10 PM)
        day_start = self._time_to_minutes("06:00")   # 360 minutes
        day_end = self._time_to_minutes("22:00")      # 1320 minutes

        # Filter to only tasks on the target date, then sort by time
        day_tasks = [t for t in tasks if t.due_date == target_date]
        sorted_tasks = self.sort_by_time(day_tasks)

        # Build a list of "busy blocks" — (start, end) in minutes
        busy_blocks = []
        for t in sorted_tasks:
            start = self._time_to_minutes(t.due_time)
            end = start + t.duration_minutes
            busy_blocks.append((start, end))

        # Walk through the day looking for a gap big enough
        # Start checking from the beginning of the schedulable window
        current_time = day_start

        for block_start, block_end in busy_blocks:
            # If the block starts after our current position,
            # there's a gap between current_time and block_start
            if block_start > current_time:
                gap = block_start - current_time
                # If the gap is big enough, we found our slot
                if gap >= duration:
                    return self._minutes_to_time(current_time)

            # Move past this busy block (take the later of the two
            # in case blocks overlap)
            if block_end > current_time:
                current_time = block_end

        # Check the gap between the last task and end of day
        if day_end - current_time >= duration:
            return self._minutes_to_time(current_time)

        # No slot found — the day is too full
        return None
