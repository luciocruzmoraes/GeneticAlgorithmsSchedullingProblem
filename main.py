import os
import pandas as pd
import random
from genetic_algorithm import execute_genAlgorithm, analyze_solution

random.seed(42)


def load_inputs():
    # Load CSVs
    df_classes = pd.read_csv("data/classes_data.csv")
    df_rooms = pd.read_csv("data/rooms_data.csv")
    df_teachers = pd.read_csv("data/teachers_data.csv")

    # Teacher assignments (if exists)
    if os.path.exists("data/teacher_assignments.csv"):
        df_assign = pd.read_csv("data/teacher_assignments.csv")
    else:
        df_assign = None

    return df_classes, df_rooms, df_teachers, df_assign


def expand_lessons(df_classes):
    """
    Expande as turmas em instâncias de aulas conforme 'lessons_per_week'.
    Gera IDs no formato: CLASSID::SUBJECT::n
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
    Cria IDs de timeslot e metadados legíveis.
    """
    timeslots = df_schedules["schedule_id"].tolist()
    schedule_meta = {}

    for _, r in df_schedules.iterrows():
        tid = r["schedule_id"]
        shift = r.get("shift", "")
        label = f"{shift[0] if shift else ''}{r.get('period', '')}".replace(" ", "")
        schedule_meta[tid] = {
            "weekday": r.get("weekday", ""),
            "shift": shift,
            "period": r.get("period", ""),
            "label": label
        }

    return timeslots, schedule_meta


def build_rooms_capacity(df_rooms):
    return dict(zip(df_rooms["room_id"], df_rooms["capacity"]))


def build_class_students(df_classes):
    """Mapeia cada turma para o número de alunos."""
    class_students = {}
    for _, row in df_classes.iterrows():
        cid = row["class_id"]
        if cid not in class_students:
            class_students[cid] = int(row["num_students"])
    return class_students


def build_teacher_assignment_map(lesson_instances, df_assignments, df_teachers, df_classes):
    """
    Mapeia cada aula (lesson_index) para um professor.
    Caso não haja CSV de atribuições, cria um mapeamento heurístico.
    """
    teacher_of_lesson = {}
    teachers_info = {}

    # Dicionário de informações dos professores
    for _, r in df_teachers.iterrows():
        teachers_info[r["teacher_id"]] = {
            "name": r.get("name"),
            "available_morning": bool(r.get("available_morning")),
            "available_afternoon": bool(r.get("available_afternoon")),
            "available_evening": bool(r.get("available_evening")),
            "max_workload": int(r.get("max_workload", 40)),
            "main_subject": r.get("main_subject")
        }

    # Se houver atribuições explícitas (CSV)
    class_to_teacher = {}
    if df_assignments is not None:
        if "class_id" in df_assignments.columns and "teacher_id" in df_assignments.columns:
            for _, r in df_assignments.iterrows():
                class_to_teacher[str(r["class_id"])] = r["teacher_id"]

    # Atribui professores por disciplina ou aleatoriamente
    teacher_by_subject = {}
    for tid, info in teachers_info.items():
        subj = info.get("main_subject")
        if subj:
            teacher_by_subject.setdefault(subj, []).append(tid)

    for idx, lesson in enumerate(lesson_instances):
        class_id, subject, _ = lesson.split("::")
        tid = class_to_teacher.get(class_id, None)
        if tid is None:
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
    df_schedules = pd.read_csv("data/schedules_data.csv")

    print("\nExpanding lessons (lessons_per_week)...")
    lesson_instances, lesson_meta = expand_lessons(df_classes)
    print(f"Total lesson instances to allocate: {len(lesson_instances)}")

    # Build timeslots and metadata
    timeslots, schedule_meta = build_timeslots(df_schedules)

    # Rooms and capacities
    rooms = df_rooms["room_id"].tolist()
    rooms_capacity = build_rooms_capacity(df_rooms)

    # Class students
    class_students = build_class_students(df_classes)

    # Teacher mapping
    teacher_of_lesson, teacher_info = build_teacher_assignment_map(
        lesson_instances, df_assign, df_teachers, df_classes
    )

    # Executa o algoritmo genético
    print("\nRunning scheduling GA (this may take a while)...")
    best, fitness, reasons, mapping = execute_genAlgorithm(
        rooms=rooms,
        timeslots=timeslots,
        lesson_instances=lesson_instances,
        rooms_capacity=rooms_capacity,
        class_students=class_students,
        teacher_of_lesson=teacher_of_lesson,
        teacher_info=teacher_info,
        class_grade_map={},  # compatível com a função do genetic_algorithm
        ngen=80,
        npop=150
    )

    # Análise e relatório final
    analyze_solution(
        best, fitness, reasons, mapping,
        lesson_instances, df_classes,
        rooms_capacity,
        {lesson: teacher_of_lesson[idx] for idx, lesson in enumerate(lesson_instances)},
        teacher_info,
        schedule_meta
    )


if __name__ == "__main__":
    main()
