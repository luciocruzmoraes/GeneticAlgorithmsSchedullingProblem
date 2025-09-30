import pandas as pd

def load_assignments(filename='data/teacher_assignments.csv'):
    """Load teacher assignments from CSV into a dictionary {teacher_id: [classes]}"""
    try:
        df = pd.read_csv(filename)
        df.columns = df.columns.str.strip()

        assignments = {}
        for teacher_id in df['teacher_id'].unique():
            teacher_classes = df[df['teacher_id'] == teacher_id]['class_id'].tolist()
            assignments[teacher_id] = teacher_classes

        return assignments
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except Exception as e:
        print(f"Error loading teacher assignments: {e}")
        return None


def load_scheduling_data(assignments_file='teacher_assignments.csv'):
    """Load all data needed for schedule optimization"""
    try:
        # Load assignments
        assignments = load_assignments(assignments_file)
        if assignments is None:
            return None

        # Load CSV data
        df_classes = pd.read_csv('data/classes_data.csv')
        df_classes.columns = df_classes.columns.str.strip()

        df_rooms = pd.read_csv('data/rooms_data.csv')
        df_rooms.columns = df_rooms.columns.str.strip()

        df_schedules = pd.read_csv('data/schedules_data.csv')
        df_schedules.columns = df_schedules.columns.str.strip()

        df_teachers = pd.read_csv('teachers_data.csv')
        df_teachers.columns = df_teachers.columns.str.strip()

        # Classes assigned to teachers
        assigned_class_ids = []
        for teacher_classes in assignments.values():
            assigned_class_ids.extend(teacher_classes)

        classes = assigned_class_ids

        # Get student counts
        class_students = {}
        for class_id in classes:
            class_data = df_classes[df_classes['class_id'] == class_id].iloc[0]
            class_students[class_id] = class_data['num_students']

        # Rooms
        rooms = df_rooms['room_id'].tolist()
        rooms_capacity = dict(zip(df_rooms['room_id'], df_rooms['capacity']))

        # Schedules
        schedules = df_schedules['schedule_id'].tolist()

        # Teachers dictionary with availability
        teachers = {}
        for teacher_id, teacher_classes in assignments.items():
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

        print(f"\n✓ Scheduling data loaded successfully")
        print(f"  - {len(classes)} classes to schedule")
        print(f"  - {len(rooms)} rooms available")
        print(f"  - {len(schedules)} schedule slots available")
        print(f"  - {len(teachers)} teachers involved")

        return data

    except Exception as e:
        print(f"Error loading scheduling data: {e}")
        return None


def check_viability(classes, class_students, classes_capacity):
    """Check if all classes can be accommodated"""
    inviable_classes = []
    for group in classes:
        students = class_students[group]
        if not any(cap >= students for cap in classes_capacity.values()):
            inviable_classes.append(group)
    if inviable_classes:
        print("\n The following classes are NOT viable:")
        for t in inviable_classes:
            print(f"  - {t} ({class_students[t]} students)")
        return False
    return True


def check_distinct_class(solution):
    """Check if all rooms are distinct"""
    rooms = [room for room, schedule in solution]
    return len(rooms) == len(set(rooms))


def show_teacher_schedule(solution, classes, teachers):
    """Display teacher schedules"""
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


def save_schedule_results(best_result, data, filename='final_schedule.csv'):
    """Save the final schedule to CSV"""
    schedule_data = []

    for group, room, schedule in best_result:
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
    print("STEP 5: FINAL SCHEDULE GENERATION")
    print("="*70)

    data = load_scheduling_data()
    if data is None:
        print("Error loading scheduling data.")
        return

    if not check_viability(data["classes"], data["class_students"], data["rooms_capacity"]):
        print("Please adjust the data to continue.")
        return

    # NOTE: This module assumes that teacher_assignmement already found an optimal solution
    # and exported results. Here we just save and display final info.
    print("\n✓ Final schedule generation ready (executed after GA optimization).")

if __name__ == "__main__":
    main()
