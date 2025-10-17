import pandas as pd
import random
import os

random.seed(42)


# -------------------------
# 1) Generate Schedules
# -------------------------
def generate_schedules():
    """Generate the base weekly schedule grid (Monday–Friday, morning shift)."""
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
        for period, label in base_times:
            schedule_id = f"{day}_{label}"
            rows.append({
                "schedule_id": schedule_id,
                "weekday": day,
                "shift": "Manhã",
                "period": period,
                "label": f"{label} {period}"
            })
    return pd.DataFrame(rows)


# -------------------------
# 2) Generate Teachers
# -------------------------
def generate_teachers():
    """Generate synthetic teacher profiles with subject specialties and workload."""
    teacher_names = [
        "Ana Souza", "Carlos Pereira", "Juliana Costa", "João Almeida",
        "Mariana Silva", "Paulo Rocha", "Lucas Martins", "Beatriz Santos",
        "Rafael Oliveira", "Clara Mendes"
    ]

    subjects = [
        "Português", "Matemática", "História", "Geografia",
        "Ciências", "Inglês", "Artes", "Educação Física",
        "Literatura", "Física", "Química", "Biologia",
        "Sociologia", "Filosofia"
    ]

    all_periods = ["M1", "M2", "M3", "M4", "M5", "M6"]
    rows = []

    for tid, name in enumerate(teacher_names, start=1):
        specialist_subject = random.choice(subjects)
        favorite_subjects = random.sample(subjects, k=random.randint(1, 3))
        preferred_periods = ", ".join(sorted(random.sample(all_periods, k=random.randint(2, 5))))
        max_workload = random.randint(12, 20)

        rows.append({
            "teacher_id": f"T{tid:03d}",
            "name": name,
            "main_subject": specialist_subject,
            "favorite_subjects": ", ".join(favorite_subjects),
            "available_morning": bool(random.getrandbits(1)),
            "available_afternoon": bool(random.getrandbits(1)),
            "available_evening": bool(random.getrandbits(1)),
            "max_workload": max_workload
        })
    return pd.DataFrame(rows)


# -------------------------
# 3) Generate Rooms
# -------------------------
def generate_rooms():
    """Generate a list of available rooms with different capacities."""
    rows = []
    for i in range(1, 11):
        rows.append({
            "room_id": f"SALA_{i:02d}",
            "capacity": random.randint(30, 40)
        })
    rows.append({"room_id": "LAB_CIENCIAS", "capacity": 35})
    rows.append({"room_id": "LAB_INFORMATICA", "capacity": 30})
    rows.append({"room_id": "AUDITORIO", "capacity": 80})
    return pd.DataFrame(rows)


# -------------------------
# 4) Generate Historical Enrollment Data
# -------------------------
def generate_historical_enrollment_data():
    """Generate synthetic historical enrollment, transfer, and dropout data."""
    years = [2020, 2021, 2022, 2023, 2024]
    grades = [
        "6º Ano", "7º Ano", "8º Ano", "9º Ano",
        "1ª Série EM", "2ª Série EM", "3ª Série EM"
    ]
    bimestres = ["B1", "B2", "B3", "B4"]
    data = []

    for year in years:
        for grade in grades:
            regional_gdp = random.uniform(50000, 90000)
            unemployment_rate = random.uniform(5.0, 12.0)
            education_investment = random.uniform(800000, 1500000)

            # Base number of students per grade
            base_students = {
                "6º Ano": random.randint(200, 280),
                "7º Ano": random.randint(190, 270),
                "8º Ano": random.randint(180, 260),
                "9º Ano": random.randint(170, 250),
                "1ª Série EM": random.randint(160, 240),
                "2ª Série EM": random.randint(150, 230),
                "3ª Série EM": random.randint(140, 220),
            }
            total_enrollments = base_students[grade]

            # Average rates (%)
            transfer_in_rate = random.uniform(1.0, 6.0)
            transfer_out_rate = random.uniform(1.0, 6.0)
            dropout_rate = random.uniform(0.5, 4.0)

            total_transfer_in = int(total_enrollments * transfer_in_rate / 100)
            total_transfer_out = int(total_enrollments * transfer_out_rate / 100)
            total_dropouts = int(total_enrollments * dropout_rate / 100)

            # Bimonthly distribution (may include zeros)
            transfers_in_bi = {b: random.randint(0, total_transfer_in // 4) for b in bimestres}
            transfers_out_bi = {b: random.randint(0, total_transfer_out // 4) for b in bimestres}
            dropouts_bi = {b: random.randint(0, total_dropouts // 4) for b in bimestres}

            data.append({
                "year": year,
                "grade": grade,
                "regional_gdp": regional_gdp,
                "unemployment_rate": unemployment_rate,
                "education_investment": education_investment,
                "total_enrollments": total_enrollments,
                "transfer_in_rate": round(transfer_in_rate, 2),
                "transfer_out_rate": round(transfer_out_rate, 2),
                "dropout_rate": round(dropout_rate, 2),
                "total_transfer_in": total_transfer_in,
                "total_transfer_out": total_transfer_out,
                "total_dropouts": total_dropouts,
                **{f"transfers_in_{b}": transfers_in_bi[b] for b in bimestres},
                **{f"transfers_out_{b}": transfers_out_bi[b] for b in bimestres},
                **{f"dropouts_{b}": dropouts_bi[b] for b in bimestres}
            })

    return pd.DataFrame(data)


# -------------------------
# 5) Main - Generate All except classes
# -------------------------
def main():
    """Generate and export base datasets: schedules, teachers, rooms, and enrollment history."""
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    schedules = generate_schedules()
    teachers = generate_teachers()
    rooms = generate_rooms()
    historical = generate_historical_enrollment_data()

    schedules.to_csv(os.path.join(output_dir, "schedules_data.csv"), index=False)
    teachers.to_csv(os.path.join(output_dir, "teachers_data.csv"), index=False)
    rooms.to_csv(os.path.join(output_dir, "rooms_data.csv"), index=False)
    historical.to_csv(os.path.join(output_dir, "historical_enrollment_data.csv"), index=False)

    return schedules, teachers, rooms, historical


if __name__ == "__main__":
    schedules, teachers, rooms, historical = main()

    print(f"   - Schedules: {len(schedules)}")
    print(f"   - Teachers:  {len(teachers)}")
    print(f"   - Rooms:     {len(rooms)}")
    print(f"   - Historical records: {len(historical)}")
