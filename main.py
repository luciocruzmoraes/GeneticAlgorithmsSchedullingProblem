import os
import pandas as pd
import random
from genetic_algorithm import execute_genAlgorithm, analyze_solution

random.seed(42)


def load_inputs():
    # load CSVs
    df_classes = pd.read_csv("data/classes_data.csv")
    df_rooms = pd.read_csv("data/rooms_data.csv")
    df_teachers = pd.read_csv("data/teachers_data.csv")

    # teacher assignments (if exists) - expected columns: teacher_id, class_id or class assignment per subject
    if os.path.exists("data/teacher_assignments.csv"):
        df_assign = pd.read_csv("data/teacher_assignments.csv")
    else:
        df_assign = None

    return df_classes, df_rooms, df_teachers, df_assign


def expand_lessons(df_classes):
    """
    Expand classes entries into lesson instances according to lessons_per_week.
    We'll create lesson ids as: CLASSID::SUBJECT::n
    Returns:
      - lesson_instances: list of lesson ids (order important)
      - lesson_meta: list/dict with info about each lesson (class_id, subject)
    """
    lesson_instances = []
    lesson_meta = []
    for _, row in df_classes.iterrows():
        class_id = row["class_id"]
        subject = row["subject"]
        lessons = int(row.get("lessons_per_week", 1))
        for i in range(lessons):
            lid = f"{class_id}::{subject}::{i+1}"
            lesson_instances.append(lid)
            lesson_meta.append({
                "lesson_id": lid,
                "class_id": class_id,
                "subject": subject
            })
    return lesson_instances, lesson_meta


def build_timeslots(df_schedules):
    """
    Create timeslot ids and mapping to human labels.
    We will use timeslot_id = schedule_id from df_schedules.
    Also create timeslot_to_shift mapping ('M','A','E') for availability checks.
    """
    timeslots = df_schedules["schedule_id"].tolist()
    timeslot_to_shift = {}
    schedule_meta = {}
    for _, r in df_schedules.iterrows():
        tid = r["schedule_id"]
        shift = r.get("shift", "")
        shift_code = "M" if shift.lower().startswith("m") else ("A" if shift.lower().startswith("a") else ("E" if shift.lower().startswith("e") else "?"))
        timeslot_to_shift[tid] = shift_code
        # human label: e.g. "M1 Monday 07:30-08:20"
        label = f"{r.get('period','')}"
        schedule_meta[tid] = {
            "weekday": r.get("weekday", ""),
            "shift": shift,
            "period": r.get("period", ""),
            "label": f"{shift[0] if shift else ''}{r.get('period','')}".replace(" ", "")
        }
    return timeslots, timeslot_to_shift, schedule_meta


def build_rooms_capacity(df_rooms):
    return dict(zip(df_rooms["room_id"], df_rooms["capacity"]))


def build_class_students(df_classes):
    # class_id -> num_students (assuming class rows duplicated per subject, keep first)
    class_students = {}
    for _, row in df_classes.iterrows():
        cid = row["class_id"]
        if cid not in class_students:
            class_students[cid] = int(row["num_students"])
    return class_students


def build_teacher_assignment_map(lesson_instances, df_assignments, df_teachers, df_classes):
    """
    Build mapping lesson_index -> teacher_id.
    If teacher_assignments.csv exists, try to map by class_id. Otherwise
    create a heuristic mapping: assign each lesson's class to a random teacher whose main subject matches if possible.
    """
    teacher_of_lesson = {}
    teachers_info = {}

    # build teacher info dict
    for _, r in df_teachers.iterrows():
        teachers_info[r["teacher_id"]] = {
            "name": r.get("name"),
            "available_morning": bool(r.get("available_morning")),
            "available_afternoon": bool(r.get("available_afternoon")),
            "available_evening": bool(r.get("available_evening")),
            "max_workload": int(r.get("max_workload", 40)),
            "main_subject": r.get("main_subject")
        }

    # if explicit assignments exist, map class -> teacher (prefer class-level mapping)
    class_to_teacher = {}
    if df_assignments is not None:
        # try to find column class_id or class
        if "class_id" in df_assignments.columns and "teacher_id" in df_assignments.columns:
            for _, r in df_assignments.iterrows():
                class_to_teacher[str(r["class_id"])] = r["teacher_id"]

    # fallback: try to assign by subject match or randomly
    teacher_by_subject = {}
    for tid, info in teachers_info.items():
        subj = info.get("main_subject")
        if subj:
            teacher_by_subject.setdefault(subj, []).append(tid)

    # Now map lessons
    for idx, lesson in enumerate(lesson_instances):
        class_id, subject, inst = lesson.split("::")
        # priority: class_to_teacher
        tid = class_to_teacher.get(class_id, None)
        if tid is None:
            # try subject match
            possible = teacher_by_subject.get(subject, [])
            if possible:
                tid = random.choice(possible)
            else:
                tid = random.choice(list(teachers_info.keys()))
        teacher_of_lesson[idx] = tid

    return teacher_of_lesson, teachers_info


def main():
    print("=" * 70)
    print("STEP 1: LOAD INPUTS")
    print("=" * 70)
    df_classes, df_rooms, df_teachers, df_assign = load_inputs()

    # ensure schedules exist
    df_schedules = pd.read_csv("data/schedules_data.csv")

    print("\nExpanding lessons (lessons_per_week)...")
    lesson_instances, lesson_meta = expand_lessons(df_classes)
    print(f"Total lesson instances to allocate: {len(lesson_instances)}")

    # build timeslots and meta
    timeslots, timeslot_to_shift, schedule_meta = build_timeslots(df_schedules)

    # rooms and capacities
    rooms = df_rooms["room_id"].tolist()
    rooms_capacity = build_rooms_capacity(df_rooms)

    # class students
    class_students = build_class_students(df_classes)

    # teacher mapping per lesson and teacher info
    teacher_of_lesson, teacher_info = build_teacher_assignment_map(lesson_instances, df_assign, df_teachers, df_classes)

    # Execute GA
    print("\nRunning scheduling GA (this may take a while)...")
    best, fitness, reasons, mapping = execute_genAlgorithm(
        rooms=rooms,
        timeslots=timeslots,
        lesson_instances=lesson_instances,
        rooms_capacity=rooms_capacity,
        class_students=class_students,
        teacher_of_lesson=teacher_of_lesson,
        teacher_info=teacher_info,
        timeslot_to_shift=timeslot_to_shift,
        ngen=80,
        npop=150
    )

    # analyze and print report
    analyze_solution(best, fitness, reasons, mapping, lesson_instances, df_classes, rooms_capacity, 
                     { (lesson): teacher_of_lesson[idx] for idx, lesson in enumerate(lesson_instances) }, teacher_info, schedule_meta)


if __name__ == "__main__":
    main()
