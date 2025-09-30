import pandas as pd
from genetic_algorithm import config_genAlgorithm, execute_genAlgorithm, analyze_problem_and_suggest

def load_teacher_assignments(filename='teacher_assignments.csv'):
    """
    Load teacher-class assignments from CSV.
    Expected columns: teacher_id, teacher_name, class_id, class_name,
    subject, num_students, lessons_per_week, assigned
    """
    try:
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()

        # If the file has "assigned" column, filter only assigned classes
        if 'assigned' in df.columns:
            df = df[df['assigned'].str.upper().str.strip() == 'Y']

        print(f"✓ Teacher assignments loaded from {filename}")
        print(f"  Total assignments: {len(df)}")
        return df
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        print("Please run the teacher choice simulation first.")
        return None
    except Exception as e:
        print(f"Error loading assignments: {e}")
        return None


def load_scheduling_data(df_assignments):
    """Load rooms, schedules, and teacher availability for schedule optimization."""
    try:
        # Load rooms
        df_rooms = pd.read_csv('rooms_data.csv')
        df_rooms.columns = df_rooms.columns.str.strip()

        # Load schedules
        df_schedules = pd.read_csv('schedules_data.csv')
        df_schedules.columns = df_schedules.columns.str.strip()

        # Load teacher info
        df_teachers = pd.read_csv('teachers_data.csv')
        df_teachers.columns = df_teachers.columns.str.strip()

        # Build classes list from assignments
        classes = df_assignments['class_id'].tolist()

        # Map class → student count
        class_students = dict(zip(df_assignments['class_id'], df_assignments['num_students']))

        # Rooms
        rooms = df_rooms['room_id'].tolist()
        rooms_capacity = dict(zip(df_rooms['room_id'], df_rooms['capacity']))

        # Schedules
        schedules = df_schedules['schedule_id'].tolist()

        # Teachers with their assigned classes
        teachers = {}
        for teacher_id in df_assignments['teacher_id'].unique():
            teacher_classes = df_assignments[df_assignments['teacher_id'] == teacher_id]['class_id'].tolist()
            teacher_info = df_teachers[df_teachers['teacher_id'] == teacher_id].iloc[0]

            teachers[teacher_id] = {
                "classes": teacher_classes,
                "available_morning": teacher_info['available_morning'],
                "available_afternoon": teacher_info['available_afternoon'],
                "available_evening": teacher_info['available_evening']
            }

        data = {
            "classes": classes,
            "class_students": class_students,
            "rooms": rooms,
            "rooms_capacity": rooms_capacity,
            "schedules": schedules,
            "teachers": teachers
        }

        print("\n✓ Scheduling data prepared successfully")
        print(f"  - {len(classes)} classes to schedule")
        print(f"  - {len(rooms)} rooms available")
        print(f"  - {len(schedules)} schedule slots available")
        print(f"  - {len(teachers)} teachers involved")

        return data

    except Exception as e:
        print(f"Error loading scheduling data: {e}")
        return None


def check_viability(classes, class_students, classes_capacity):
    """Check if all classes can be accommodated in available rooms."""
    inviable_classes = []
    for group in classes:
        students = class_students[group]
        if not any(cap >= students for cap in classes_capacity.values()):
            inviable_classes.append(group)
    if inviable_classes:
        print("\n⚠️  The following classes are NOT viable:")
        for t in inviable_classes:
            print(f"  - {t} ({class_students[t]} students)")
        return False
    return True


def check_distinct_class(solution):
    """Check if all rooms are distinct in a given solution."""
    rooms = [room for room, schedule in solution]
    return len(rooms) == len(set(rooms))


def show_teacher_schedule(solution, classes, teachers):
    """Print teacher schedules for allocated classes."""
    group_schedule = {}
    for i, (room, schedule) in enumerate(solution):
        group = classes[i]
        group_schedule[group] = schedule

    print("\nTeacher Schedules:")
    for teacher, data in teachers.items():
        if data["classes"]:
            print(f"Teacher {teacher}:")
            for group in data["classes"]:
                schedule = group_schedule.get(group, None)
                if schedule:
                    print(f"  - Class {group}: Schedule {schedule}")
                else:
                    print(f"  - Class {group}: Not allocated")


def save_schedule_results(best_solution, data, filename='final_schedule.csv'):
    """Save the final schedule solution to CSV."""
    schedule_data = []

    for group, room, schedule in best_solution:
        teacher_id = None
        for prof, prof_data in data["teachers"].items():
            if group in prof_data["classes"]:
                teacher_id = prof
                break

        schedule_data.append({
            'class_id': group,
            'room': room,
            'schedule': schedule,
            'teacher_id': teacher_id,
            'num_students': data["class_students"][group],
            'room_capacity': data["rooms_capacity"][room]
        })

    df_schedule = pd.DataFrame(schedule_data)
    df_schedule.to_csv(filename, index=False, encoding="utf-8")
    print(f"\n✓ Final schedule saved to: {filename}")


def main():
    print("="*70)
    print("STEP 4: TEACHER ASSIGNMENT AND SCHEDULE OPTIMIZATION")
    print("="*70)

    # Load teacher assignments
    df_assignments = load_teacher_assignments()
    if df_assignments is None:
        return

    # Load scheduling data
    data = load_scheduling_data(df_assignments)
    if data is None:
        print("Unable to load scheduling data. Exiting.")
        return

    # Check viability
    print("\nChecking viability...")
    if not check_viability(data["classes"], data["class_students"], data["rooms_capacity"]):
        print("Please adjust the data to continue.")
        return

    print("✓ All classes are viable")

    # Configure genetic algorithm
    print("\nConfiguring genetic algorithm...")
    toolbox = config_genAlgorithm(
        data["classes"], data["rooms"], data["schedules"],
        data["rooms_capacity"], data["class_students"], data["teachers"]
    )

    # Execute genetic algorithm
    print("\nStarting schedule optimization...")
    solution_found = False
    tries = 0
    rejected_solutions = 0
    max_tries_before_suggestion = 100

    while not solution_found:
        tries += 1
        print(f"\n Attempt #{tries} - Evolving population...")

        hof, log = execute_genAlgorithm(toolbox, ngen=50, npop=100)

        if tries % max_tries_before_suggestion == 0:
            analyze_problem_and_suggest(data, tries)
            user_input = input("\nDo you want to continue searching? (y/n): ").strip().lower()
            if user_input != 'y':
                print("\nSearch interrupted by user.")
                return

        for ind in hof:
            if ind.fitness.values[0] == 0.0:
                actual_solution = [(room, schedule) for room, schedule in ind]

                if check_distinct_class(actual_solution):
                    solution_found = True
                    best_result = [
                        (data["classes"][i], room, schedule) for i, (room, schedule) in enumerate(actual_solution)
                    ]
                    fitness = ind.fitness.values

                    print("\n" + "="*70)
                    print("✓ OPTIMAL SCHEDULE FOUND!")
                    print("="*70)

                    print("\nSchedule Details:")
                    for group, room, schedule in best_result:
                        print(f"  Class {group} → Room {room} - Schedule {schedule}")

                    print(f"\nFitness (violations, occupancy rate): ({fitness[0]}, {fitness[1]:.2f}%)")

                    print("\nRoom occupancy:")
                    students_per_room = {}
                    for i, (room, _) in enumerate(actual_solution):
                        group = data["classes"][i]
                        students = data["class_students"][group]
                        students_per_room[room] = students_per_room.get(room, 0) + students

                    for room, students in students_per_room.items():
                        capacity = data["rooms_capacity"].get(room, 0)
                        occupancy = (students / capacity) * 100 if capacity > 0 else 0.0
                        print(f"  Room {room}: {students} students / {capacity} capacity → {occupancy:.2f}%")

                    # Show teacher schedules
                    show_teacher_schedule(actual_solution, data["classes"], data["teachers"])

                    # Save results
                    save_schedule_results(best_result, data)

                    if rejected_solutions > 0:
                        print(f"\n Total rejected solutions: {rejected_solutions}")

                    print("\n" + "="*70)
                    print("✓ SCHEDULE OPTIMIZATION COMPLETE!")
                    print("="*70)
                    return
                else:
                    rejected_solutions += 1
                    print(f"  ✗ Solution #{rejected_solutions} REJECTED: Duplicate room assignments")
            else:
                rejected_solutions += 1


if __name__ == "__main__":
    main()
