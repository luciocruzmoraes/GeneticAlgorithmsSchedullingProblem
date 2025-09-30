import pandas as pd
import random
import os

random.seed(42)


# -------------------------
# 1) Schedules (Horários)
# -------------------------
def generate_schedules():
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    base_times = [
        ("07:30-08:20", "M1"),
        ("08:20-09:10", "M2"),
        ("09:10-10:00", "M3"),
        ("10:20-11:10", "M4"),
        ("11:10-12:00", "M5"),
        ("12:00-12:40", "M6"),
    ]
    rows = []
    for day in weekdays:
        for _, (period, label) in enumerate(base_times, start=1):
            schedule_id = f"{day}_{label}"
            rows.append({
                "schedule_id": schedule_id,
                "weekday": day,
                "shift": "Morning",
                "period": period,
                "label": f"{label} {period}"
            })
    return pd.DataFrame(rows)


# -------------------------
# 2) Classes (Turmas + Disciplinas)
# -------------------------
def generate_classes():
    # Disciplinas por série (simplificado para estadual)
    grades = {
        "6º Ano": ["Matemática", "Português", "História", "Geografia", "Inglês", "Ciências", "Educação Física"],
        "7º Ano": ["Matemática", "Português", "História", "Geografia", "Inglês", "Ciências", "Educação Física"],
        "8º Ano": ["Matemática", "Português", "História", "Geografia", "Inglês", "Física", "Química", "Educação Física"],
        "9º Ano": ["Matemática", "Português", "História", "Geografia", "Inglês", "Física", "Química", "Educação Física"],
        "1ª Série": ["Matemática", "Português", "História", "Geografia", "Inglês", "Física", "Química", "Biologia"],
        "2ª Série": ["Matemática", "Português", "História", "Geografia", "Inglês", "Física", "Química", "Biologia"],
        "3ª Série": ["Matemática", "Português", "História", "Geografia", "Inglês", "Física", "Química", "Biologia", "Sociologia"]
    }

    rows = []
    cid = 1
    for grade, subjects in grades.items():
        for turma in ["A", "B"]:  # duas turmas por série
            class_id = f"C{cid:03d}"
            num_students = random.randint(25, 40)
            for subject in subjects:
                lessons_per_week = random.randint(2, 5)  # carga horária por disciplina
                rows.append({
                    "class_id": class_id,
                    "class_name": f"{grade} {turma}",
                    "grade": grade,
                    "num_students": num_students,
                    "subject": subject,
                    "lessons_per_week": lessons_per_week
                })
            cid += 1
    return pd.DataFrame(rows)


# -------------------------
# 3) Teachers
# -------------------------
def generate_teachers():
    teacher_names = [
        "Ana Souza", "Carlos Silva", "Fernanda Lima", "João Pereira",
        "Mariana Costa", "Paulo Oliveira", "Julia Rodrigues",
        "Rafael Mendes", "Cláudia Nunes", "Bruno Rocha"
    ]
    specialties = [
        "Matemática", "Português", "História", "Geografia",
        "Inglês", "Física", "Química", "Biologia", "Ciências", "Educação Física", "Sociologia"
    ]
    rows = []
    for tid, name in enumerate(teacher_names, start=1):
        rows.append({
            "teacher_id": f"T{tid:03d}",
            "name": name,
            "specialty": random.choice(specialties),
            "available_morning": True,
            "available_afternoon": False,
            "available_evening": False,
            "max_workload": random.randint(12, 20)
        })
    return pd.DataFrame(rows)


# -------------------------
# 4) Rooms
# -------------------------
def generate_rooms():
    rows = []
    for i in range(1, 11):
        rows.append({
            "room_id": f"ROOM_{i:02d}",
            "capacity": random.randint(30, 40)
        })
    rows.append({"room_id": "LAB_CIENCIAS", "capacity": 35})
    rows.append({"room_id": "LAB_INFO", "capacity": 30})
    rows.append({"room_id": "AUDITORIO", "capacity": 80})
    return pd.DataFrame(rows)


# -------------------------
# Save all in /data
# -------------------------
def main():
    # cria pasta data se não existir
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    schedules = generate_schedules()
    classes = generate_classes()
    teachers = generate_teachers()
    rooms = generate_rooms()

    schedules.to_csv(os.path.join(output_dir, "schedules_data.csv"), index=False)
    classes.to_csv(os.path.join(output_dir, "classes_data.csv"), index=False)
    teachers.to_csv(os.path.join(output_dir, "teachers_data.csv"), index=False)
    rooms.to_csv(os.path.join(output_dir, "rooms_data.csv"), index=False)

    print(f"Arquivos simulados gerados com sucesso em: {output_dir}/")


if __name__ == "__main__":
    main()
