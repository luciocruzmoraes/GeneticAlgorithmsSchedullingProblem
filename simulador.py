import pandas as pd
import random
import os

random.seed(42)

# -------------------------
# 1) Generate Schedules
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
# 2) Generate Classes
# -------------------------
def generate_classes():
    fundamental_subjects = ["Portuguese", "Mathematics", "History", "Geography",
                            "Science", "English", "Arts", "Physical Education", "Religious Education"]

    highschool_subjects = ["Portuguese", "Literature", "Mathematics", "Physics", "Chemistry",
                           "Biology", "History", "Geography", "Sociology", "Philosophy", "English"]

    grades = {
        "6th Grade": fundamental_subjects,
        "7th Grade": fundamental_subjects,
        "8th Grade": fundamental_subjects,
        "9th Grade": fundamental_subjects,
        "1st Year": highschool_subjects,
        "2nd Year": highschool_subjects,
        "3rd Year": highschool_subjects
    }

    rows = []
    cid = 1
    for grade, subjects in grades.items():
        for group in ["A", "B"]:
            class_id = f"C{cid:03d}"
            num_students = random.randint(25, 40)
            for subject in subjects:
                lessons_per_week = random.randint(2, 5)
                level = "fundamental" if "Grade" in grade else "highschool"
                rows.append({
                    "class_id": class_id,
                    "class_name": f"{grade} {group}",
                    "education_level": level,
                    "num_students": num_students,
                    "subject": subject,
                    "lessons_per_week": lessons_per_week
                })
            cid += 1
    return pd.DataFrame(rows)


# -------------------------
# 3) Generate Teachers (with timeslot preferences)
# -------------------------
def generate_teachers():
    teacher_names = [
        "Anna Smith", "Charles Johnson", "Laura Brown", "John Davis",
        "Maria Clark", "Paul Lewis", "Julia Walker",
        "Robert Hall", "Clara Young", "Brian Adams"
    ]

    fundamental_subjects = ["Portuguese", "Mathematics", "History", "Geography",
                            "Science", "English", "Arts", "Physical Education", "Religious Education"]

    highschool_subjects = ["Portuguese", "Literature", "Mathematics", "Physics", "Chemistry",
                           "Biology", "History", "Geography", "Sociology", "Philosophy", "English"]

    all_subjects = list(set(fundamental_subjects + highschool_subjects))

    compatible_subjects = {
        "Portuguese": ["Literature", "English"],
        "Literature": ["Portuguese", "English"],
        "Mathematics": ["Physics", "Chemistry"],
        "Physics": ["Mathematics", "Chemistry"],
        "Chemistry": ["Physics", "Biology"],
        "Biology": ["Chemistry", "Environmental Studies"],
        "History": ["Geography", "Sociology"],
        "Geography": ["History", "Philosophy"],
        "Science": ["Biology", "Chemistry"],
        "English": ["Portuguese", "Literature"],
        "Arts": ["Physical Education", "Technology & Innovation"],
        "Physical Education": ["Biology", "Health"],
        "Religious Education": ["Philosophy", "Sociology"],
        "Sociology": ["Philosophy", "History"],
        "Philosophy": ["Sociology", "History"]
    }

    grade_ranges = [
        "6th–7th Grade",
        "8th–9th Grade",
        "1st–2nd Year",
        "2nd–3rd Year",
        "8th Grade–3rd Year"
    ]

    # Generate all possible timeslots for the week
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    periods = ["M1", "M2", "M3", "M4", "M5", "M6"]
    all_timeslots = [f"{d}_{p}" for d in weekdays for p in periods]

    rows = []
    for tid, name in enumerate(teacher_names, start=1):
        specialty = random.choice(all_subjects)

        # Education levels
        can_teach_both = random.random() < 0.6
        education_levels = "fundamental, highschool" if can_teach_both else random.choice(["fundamental", "highschool"])

        # Preferred subjects
        compat = compatible_subjects.get(specialty, [])
        additional_prefs = random.sample(compat, k=min(random.randint(0, 2), len(compat)))
        preferred_subjects = [specialty] + additional_prefs

        # Preferred grade range
        preferred_grades_range = random.choice(grade_ranges)

        # Preferred and available time slots
        preferred_timeslots = random.sample(all_timeslots, k=random.randint(8, 15))
        # Available slots include preferred + extras (broader range)
        available_timeslots = list(set(preferred_timeslots + random.sample(all_timeslots, k=random.randint(10, 20))))

        rows.append({
            "teacher_id": f"T{tid:03d}",
            "name": name,
            "education_levels": education_levels,
            "main_subject": specialty,
            "preferred_subjects": ", ".join(preferred_subjects),
            "preferred_grades_range": preferred_grades_range,
            "preferred_timeslots": ", ".join(preferred_timeslots),
            "available_timeslots": ", ".join(available_timeslots),
            "preferred_max_classes": random.randint(2, 5),
            "max_workload": random.randint(12, 20)
        })

    return pd.DataFrame(rows)


# -------------------------
# 4) Generate Rooms
# -------------------------
def generate_rooms():
    rows = []
    for i in range(1, 11):
        rows.append({
            "room_id": f"ROOM_{i:02d}",
            "capacity": random.randint(30, 40)
        })
    rows.append({"room_id": "SCIENCE_LAB", "capacity": 35})
    rows.append({"room_id": "COMPUTER_LAB", "capacity": 30})
    rows.append({"room_id": "AUDITORIUM", "capacity": 80})
    return pd.DataFrame(rows)


# -------------------------
# 5) Main Script
# -------------------------
def main():
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

    print(f"Simulated data successfully generated in: {output_dir}/")
    print("-> Teachers now have detailed preferences for specific timeslots (e.g., Monday_M1, Wednesday_M3).")


if __name__ == "__main__":
    main()
