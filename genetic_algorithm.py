
import random
from collections import defaultdict
from deap import base, creator, tools, algorithms

random.seed(42)


def create_individual(rooms, timeslots, length):
    """
    Individual: list of length `length`, each gene is (room_id, timeslot_id).
    """
    return [(random.choice(rooms), random.choice(timeslots)) for _ in range(length)]


def evaluate_schedule(individual, lesson_instances, rooms_capacity, class_students,
                      teacher_of_lesson, teacher_info, timeslot_to_shift):
    """
    Evaluate a schedule represented by `individual`.
    Returns (violations,) as DEAP expects a tuple.
    Also attaches reasons list to individual (for reporting).
    """
    violations = 0
    reasons = []

    assigned_room_timeslot = defaultdict(list)  # (room, timeslot) -> lessons
    teacher_load = defaultdict(int)             # teacher_id -> num lessons

    for idx, gene in enumerate(individual):
        room, timeslot = gene
        lesson_id = lesson_instances[idx]
        class_id = lesson_id.split("::")[0]
        students = class_students[class_id]

        # Check if room exists
        if room not in rooms_capacity:
            violations += 5
            reasons.append(f"Room {room} not found in registry.")
            continue

        # Check room capacity
        if rooms_capacity[room] < students:
            diff = students - rooms_capacity[room]
            violations += diff
            reasons.append(
                f"Room {room} (capacity {rooms_capacity[room]}) was assigned to class {class_id} with {students} students."
            )

        # Room-time conflict
        assigned_room_timeslot[(room, timeslot)].append((idx, class_id))

        # Teacher availability and workload
        teacher_id = teacher_of_lesson.get(idx, None)
        if teacher_id:
            teacher_load[teacher_id] += 1
            tinfo = teacher_info.get(teacher_id, {})
            shift_code = timeslot_to_shift.get(timeslot, "?")
            if shift_code == "M" and not tinfo.get("available_morning", False):
                violations += 1
                reasons.append(
                    f"Teacher {tinfo.get('name', teacher_id)} ({teacher_id}) is not available in the morning, "
                    f"but was assigned to class {class_id} at timeslot {timeslot}."
                )
            if shift_code == "A" and not tinfo.get("available_afternoon", False):
                violations += 1
                reasons.append(
                    f"Teacher {tinfo.get('name', teacher_id)} ({teacher_id}) is not available in the afternoon, "
                    f"but was assigned to class {class_id} at timeslot {timeslot}."
                )
            if shift_code == "E" and not tinfo.get("available_evening", False):
                violations += 1
                reasons.append(
                    f"Teacher {tinfo.get('name', teacher_id)} ({teacher_id}) is not available in the evening, "
                    f"but was assigned to class {class_id} at timeslot {timeslot}."
                )

    # Room conflicts
    for (room, timeslot), lst in assigned_room_timeslot.items():
        if len(lst) > 1:
            involved = ", ".join([f"{c}" for (_, c) in lst])
            violations += 5 * (len(lst) - 1)
            reasons.append(
                f"Scheduling conflict: room {room} was assigned simultaneously "
                f"to classes {involved} at timeslot {timeslot}."
            )

    # Teacher overload
    for tid, load in teacher_load.items():
        maxw = teacher_info.get(tid, {}).get("max_workload", None)
        if maxw is not None and load > maxw:
            diff = load - maxw
            violations += diff
            reasons.append(
                f"Teacher {teacher_info.get(tid, {}).get('name', tid)} ({tid}) exceeded workload ({load} > {maxw})."
            )

    individual.reasons = reasons
    return (violations,)


def mutate_individual(individual, rooms, timeslots, indpb=0.05):
    """Mutation: for each gene, with probability indpb, change room or timeslot."""
    for i in range(len(individual)):
        if random.random() < indpb:
            if random.random() < 0.6:  # change timeslot
                individual[i] = (individual[i][0], random.choice(timeslots))
            else:  # change room
                individual[i] = (random.choice(rooms), individual[i][1])
    return (individual,)


def crossover_individual(ind1, ind2):
    """Two-point crossover on list of genes (tuples)."""
    if len(ind1) < 2:
        return ind1, ind2
    a = random.randint(1, len(ind1) - 1)
    b = random.randint(1, len(ind1) - 1)
    if a > b:
        a, b = b, a
    ind1[a:b], ind2[a:b] = ind2[a:b], ind1[a:b]
    return ind1, ind2


def execute_genAlgorithm(rooms, timeslots, lesson_instances, rooms_capacity, class_students,
                         teacher_of_lesson, teacher_info, timeslot_to_shift,
                         ngen=80, npop=150):
    """
    Runs GA. Returns (best_individual, fitness_value, reasons, mapping of lesson->assignment).
    """
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "ScheduleInd"):
        creator.create("ScheduleInd", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.ScheduleInd,
                     lambda: create_individual(rooms, timeslots, len(lesson_instances)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", crossover_individual)
    toolbox.register("mutate", mutate_individual, rooms=rooms, timeslots=timeslots, indpb=0.08)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_schedule,
                     lesson_instances=lesson_instances,
                     rooms_capacity=rooms_capacity,
                     class_students=class_students,
                     teacher_of_lesson=teacher_of_lesson,
                     teacher_info=teacher_info,
                     timeslot_to_shift=timeslot_to_shift)

    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", min)
    stats.register("avg", lambda fits: sum(fits) / len(fits))

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=ngen,
                                       stats=stats, halloffame=hof, verbose=False)

    best = hof[0]
    fitness_value = best.fitness.values[0] if isinstance(best.fitness.values, tuple) else best.fitness.values
    if isinstance(fitness_value, (list, tuple)):
        fitness_value = fitness_value[0]
    reasons = getattr(best, "reasons", [])

    mapping = {lesson: best[idx] for idx, lesson in enumerate(lesson_instances)}

    return best, fitness_value, reasons, mapping


def analyze_solution(best, fitness, reasons, mapping, lesson_instances, df_classes,
                     rooms_capacity, teacher_assignments, teacher_info, schedule_meta):
    """
    Print detailed report:
      - Room issues
      - Scheduling conflicts
      - Teacher issues
      - Weekly allocation summary per class (includes grade/year)
    """
    room_issues = [r for r in reasons if ("Room " in r or "capacity" in r)]
    conflict_issues = [r for r in reasons if "conflict" in r or "Conflict" in r]
    teacher_issues = [r for r in reasons if "Teacher " in r or "teacher" in r]

    print("\n=== ALLOCATION REPORT ===\n")

    print("Room Issues:")
    print(" - None found." if not room_issues else "\n".join(f" - {r}" for r in room_issues))

    print("\nScheduling Conflicts:")
    print(" - None found." if not conflict_issues else "\n".join(f" - {r}" for r in conflict_issues))

    print("\nTeacher Issues:")
    print(" - None found." if not teacher_issues else "\n".join(f" - {r}" for r in teacher_issues))

    print("\nWeekly Allocation Summary per Class:\n")

    schedule_by_class = {}
    for lesson in lesson_instances:
        class_id, subject, inst = lesson.split("::")
        room, timeslot = mapping[lesson]
        meta = schedule_meta.get(timeslot, {"weekday": "?", "shift": "?", "period": timeslot, "label": timeslot})
        teacher = teacher_assignments.get(lesson, None)
        teacher_name = teacher_info.get(teacher, {}).get("name", teacher) if teacher else "N/A"

        schedule_by_class.setdefault(class_id, []).append({
            "subject": subject,
            "room": room,
            "timeslot": timeslot,
            "weekday": meta.get("weekday"),
            "shift": meta.get("shift"),
            "period": meta.get("period"),
            "label": meta.get("label"),
            "teacher": teacher_name
        })

    for class_id, slots in schedule_by_class.items():
        row = df_classes[df_classes["class_id"] == class_id]
        if not row.empty:
            class_name = row.iloc[0].get("class_name", class_id)
            grade = row.iloc[0].get("grade", "?")
            num_students = row.iloc[0].get("num_students", "N/A")
        else:
            class_name = class_id
            grade = "?"
            num_students = "N/A"

        print(f"Class {class_id} ({class_name}) - {grade} - {num_students} students")
        slots_sorted = sorted(slots, key=lambda s: (weekday_order(s["weekday"]), s["label"]))
        for s in slots_sorted:
            print(f"   - {s['weekday']} | {s['label']} | {s['shift']} | Room: {s['room']} | "
                  f"Subject: {s['subject']} | Teacher: {s['teacher']}")
        print("-" * 70)


def weekday_order(weekday):
    order = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4,
        "Segunda": 0, "Terça": 1, "Quarta": 2, "Quinta": 3, "Sexta": 4
    }
    return order.get(weekday, 99)
