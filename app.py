"""
PawPal+ Streamlit UI
────────────────────
This file connects the Streamlit front-end to the backend logic
in pawpal_system.py. It uses st.session_state to keep data alive
between page refreshes.
"""

import streamlit as st
from datetime import date, time

# Import our backend classes from the logic layer
from src.pawpal_system import Task, Pet, Owner, Scheduler


# ──────────────────────────────────────────────
# Page config (must be the first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

st.title("🐾 PawPal+")
st.caption("Smart pet care management — keep your furry friends happy and healthy.")


# ──────────────────────────────────────────────
# Session state setup
# ──────────────────────────────────────────────
# Streamlit reruns the whole script on every interaction.
# We store our Owner object in session_state so it persists.
if "owner" not in st.session_state:
    st.session_state.owner = None
if "scheduler" not in st.session_state:
    st.session_state.scheduler = None


# ══════════════════════════════════════════════
# SECTION 1: Owner Setup
# ══════════════════════════════════════════════
st.header("1. Owner Setup")

owner_name = st.text_input("Owner name", value="Jordan")

if st.button("Create / Update Owner"):
    # Create a fresh Owner (resets pets if the name changes)
    st.session_state.owner = Owner(name=owner_name)
    st.session_state.scheduler = Scheduler(st.session_state.owner)
    st.success(f"Owner \"{owner_name}\" is ready!")

# Stop here if no owner exists yet
if st.session_state.owner is None:
    st.info("Enter your name above and click the button to get started.")
    st.stop()

# Shortcuts for cleaner code below
owner = st.session_state.owner
scheduler = st.session_state.scheduler

st.divider()


# ══════════════════════════════════════════════
# SECTION 2: Manage Pets
# ══════════════════════════════════════════════
st.header("2. Manage Pets")

col_pet1, col_pet2, col_pet3 = st.columns(3)
with col_pet1:
    new_pet_name = st.text_input("Pet name", value="Mochi")
with col_pet2:
    new_pet_species = st.selectbox("Species", ["dog", "cat", "bird", "rabbit", "other"])
with col_pet3:
    new_pet_age = st.number_input("Age (years)", min_value=0, max_value=30, value=3)

if st.button("Add Pet"):
    # Check if a pet with this name already exists
    if owner.get_pet(new_pet_name):
        st.warning(f"A pet named \"{new_pet_name}\" already exists.")
    else:
        pet = Pet(name=new_pet_name, species=new_pet_species, age=new_pet_age)
        owner.add_pet(pet)
        st.success(f"Added {new_pet_name} the {new_pet_species}!")

# Show registered pets
if owner.pets:
    st.markdown("**Registered pets:**")
    pet_data = [
        {"Name": p.name, "Species": p.species, "Age": p.age,
         "Tasks": len(p.get_tasks())}
        for p in owner.pets
    ]
    st.table(pet_data)
else:
    st.info("No pets yet. Add one above.")
    st.stop()

st.divider()


# ══════════════════════════════════════════════
# SECTION 3: Add Tasks
# ══════════════════════════════════════════════
st.header("3. Add Tasks")

# Build a list of pet names for the dropdown
pet_names = [p.name for p in owner.pets]

col_t1, col_t2 = st.columns(2)
with col_t1:
    task_pet = st.selectbox("Assign to pet", pet_names)
    task_desc = st.text_input("Task description", value="Morning walk")
    task_priority = st.selectbox("Priority", ["high", "medium", "low"])
with col_t2:
    task_date = st.date_input("Due date", value=date.today())
    task_time = st.time_input("Due time", value=time(7, 0))
    task_duration = st.number_input("Duration (min)", min_value=1, max_value=240, value=30)

task_frequency = st.selectbox("Frequency", ["once", "daily", "weekly"])

if st.button("Add Task"):
    # Build a Task object from the form inputs
    new_task = Task(
        description=task_desc,
        due_date=task_date.isoformat(),          # convert date to "YYYY-MM-DD"
        due_time=task_time.strftime("%H:%M"),     # convert time to "HH:MM"
        duration_minutes=int(task_duration),
        priority=task_priority,
        frequency=task_frequency,
    )
    # Find the selected pet and add the task to them
    pet = owner.get_pet(task_pet)
    if pet:
        pet.add_task(new_task)
        st.success(f"Added \"{task_desc}\" for {task_pet}!")

st.divider()


# ══════════════════════════════════════════════
# SECTION 4: Daily Schedule
# ══════════════════════════════════════════════
st.header("4. Daily Schedule")

schedule_date = st.date_input(
    "View schedule for", value=date.today(), key="schedule_date"
)

# Store the generated schedule in session_state so it persists
# across reruns (e.g., when the user adjusts the slot duration input)
if "schedule_data" not in st.session_state:
    st.session_state.schedule_data = None
    st.session_state.schedule_target = None

if st.button("Generate Schedule"):
    target = schedule_date.isoformat()
    schedule = scheduler.get_daily_schedule(target)
    # Save the results so they survive future reruns
    st.session_state.schedule_data = schedule
    st.session_state.schedule_target = target

# Display the schedule if we have one saved
if st.session_state.schedule_data is not None:
    schedule = st.session_state.schedule_data
    target = st.session_state.schedule_target

    if not schedule:
        st.info(f"No tasks scheduled for {target}.")
    else:
        # ── Show the sorted schedule as a table ──
        st.subheader(f"Schedule for {target}")

        schedule_rows = []
        for t in schedule:
            schedule_rows.append({
                "Time": t.due_time,
                "Task": t.description,
                "Pet": t.pet_name,
                "Priority": t.priority,
                "Duration": f"{t.duration_minutes} min",
                "Frequency": t.frequency,
                "Status": "Done" if t.completed else "Pending",
            })
        st.table(schedule_rows)

        # ── Explain the schedule reasoning ──
        with st.expander("Why this order?"):
            st.markdown(
                "Tasks are sorted **by time** (earliest first) so you can follow "
                "them in order throughout the day. High-priority tasks appear at "
                "the time they're due — check the **Priority** column to see what "
                "matters most."
            )

        # ── Conflict warnings ──
        conflicts = scheduler.detect_conflicts(schedule)
        if conflicts:
            st.subheader("Schedule Conflicts")
            for warning in conflicts:
                st.warning(warning)
        else:
            st.success("No scheduling conflicts detected!")

        # ── Next available slot finder ──
        st.subheader("Find the Next Available Slot")
        slot_duration = st.number_input(
            "How many minutes do you need?",
            min_value=5, max_value=480, value=30, key="slot_duration"
        )
        slot = scheduler.find_next_available_slot(schedule, slot_duration, target)
        if slot:
            st.success(
                f"A {slot_duration}-minute slot is available starting at **{slot}** "
                f"on {target}."
            )
        else:
            st.error(
                f"No {slot_duration}-minute slot available on {target}. "
                f"Try a shorter duration or a different day."
            )

st.divider()


# ══════════════════════════════════════════════
# SECTION 5: Task Management
# ══════════════════════════════════════════════
st.header("5. Task Management")

all_tasks = owner.get_all_tasks()

if not all_tasks:
    st.info("No tasks to manage yet. Add some in Section 3.")
else:
    # ── Filtering controls ──
    st.subheader("Filter & Sort")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        filter_pet = st.selectbox(
            "Filter by pet", ["All"] + pet_names, key="filter_pet"
        )
    with col_f2:
        filter_status = st.selectbox(
            "Filter by status", ["All", "Pending", "Done"], key="filter_status"
        )
    with col_f3:
        sort_option = st.selectbox(
            "Sort by", ["Time", "Priority"], key="sort_option"
        )

    # Start with all tasks, then apply filters
    filtered = list(all_tasks)

    # Apply pet filter
    if filter_pet != "All":
        filtered = scheduler.filter_by_pet(filtered, filter_pet)

    # Apply status filter
    if filter_status == "Pending":
        filtered = scheduler.filter_by_status(filtered, completed=False)
    elif filter_status == "Done":
        filtered = scheduler.filter_by_status(filtered, completed=True)

    # Apply sort
    if sort_option == "Time":
        filtered = scheduler.sort_by_time(filtered)
    else:
        filtered = scheduler.sort_by_priority(filtered)

    # Display the filtered/sorted results
    if not filtered:
        st.info("No tasks match your filters.")
    else:
        filtered_rows = []
        for t in filtered:
            filtered_rows.append({
                "Time": t.due_time,
                "Date": t.due_date,
                "Task": t.description,
                "Pet": t.pet_name,
                "Priority": t.priority,
                "Frequency": t.frequency,
                "Status": "Done" if t.completed else "Pending",
            })
        st.table(filtered_rows)

    # ── Mark a task complete ──
    st.subheader("Complete a Task")

    # Build descriptions for pending tasks only
    pending = scheduler.filter_by_status(all_tasks, completed=False)
    if not pending:
        st.success("All tasks are done!")
    else:
        pending_labels = [
            f"{t.due_time} — {t.description} ({t.pet_name})" for t in pending
        ]
        selected_label = st.selectbox("Select a task to complete", pending_labels)

        if st.button("Mark Complete"):
            # Find which pending task matches the selected label
            idx = pending_labels.index(selected_label)
            task_to_complete = pending[idx]

            # Use the scheduler so recurrence is handled automatically
            new_task = scheduler.mark_task_complete(task_to_complete)

            st.success(f"Completed: \"{task_to_complete.description}\"")
            if new_task:
                st.info(
                    f"Recurring task — next occurrence auto-scheduled "
                    f"for {new_task.due_date} at {new_task.due_time}."
                )
            # Rerun so the UI reflects the change
            st.rerun()
