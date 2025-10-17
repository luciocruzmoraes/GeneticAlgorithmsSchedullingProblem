import random
from collections import defaultdict
from deap import base, creator, tools, algorithms

random.seed(42)
INDPB = 0.08

# ===========================================================
# STEP 1: CREATE INDIVIDUALS
# ===========================================================
def create_individual(rooms, timeslots, length):
    """Create an individual composed of (room, timeslot) pairs."""
    return [(random.choice(rooms), random.choice(timeslots)) for _ in range(length)]


# ===========================================================
# STEP 2: FITNESS EVALUATION
# ===========================================================
def evaluate_schedule(individual, lesson_instances, rooms_capacity, class_students,
                      teacher_of_lesson, teacher_info, class_grade_map):
    """
    Evaluate a scheduling solution considering hard and soft constraints.

    HARD CONSTRAINTS:
      - Room must exist and have enough capacity
      - Room conflict (same room/time)
      - Teacher conflict (same teacher/time)
      - Workload limit

    SOFT CONSTRAINTS:
      - Teacher preferred periods
      - Teacher favorite subjects
    """
    violations = 0
    reasons = []

    assigned_room_timeslot = defaultdict(list)
    teacher_load = defaultdict(int)
    teacher_schedule = defaultdict(list)

    for idx, gene in enumerate(individual):
        room, timeslot = gene
        lesson_id = lesson_instances[idx]
        class_id, subject, *_ = lesson_id.split("::")
        students = class_students[class_id]

        # HARD: Room existence
        if room not in rooms_capacity:
            violations += 10
            reasons.append(f"[HARD] Room {room} not found in dataset.")
            continue

        # HARD: Room capacity
        if rooms_capacity[room] < students:
            diff = students - rooms_capacity[room]
            violations += diff * 2
            reasons.append(
                f"[HARD] Room {room} (capacity {rooms_capacity[room]}) "
                f"is too small for class {class_id} ({students} students)."
            )

        assigned_room_timeslot[(room, timeslot)].append((idx, class_id))

        # Teacher assignment
        teacher_id = teacher_of_lesson.get(idx, None)
        if teacher_id:
            teacher_load[teacher_id] += 1
            teacher_schedule[teacher_id].append((timeslot, class_id))
            tinfo = teacher_info.get(teacher_id, {})

            # SOFT: Preferred periods
            preferred_periods_str = tinfo.get("teacher_preferred_periods", "")
            preferred_periods = [p.strip() for p in preferred_periods_str.split(",") if p.strip()]
            if preferred_periods and timeslot not in preferred_periods:
                violations += 0.5
                reasons.append(
                    f"[SOFT] {tinfo.get('teacher_name', teacher_id)} prefers {preferred_periods} "
                    f"but was assigned to {timeslot}."
                )

            # SOFT: Favorite subjects
            fav_subjects_str = tinfo.get("teacher_favorites_subject", "")
            fav_subjects = [s.strip() for s in fav_subjects_str.split(",") if s.strip()]
            if fav_subjects and subject not in fav_subjects:
                violations += 0.5
                reasons.append(
                    f"[SOFT] {tinfo.get('teacher_name', teacher_id)} prefers {fav_subjects} "
                    f"but was assigned {subject}."
                )

    # HARD: Room conflicts
    for (room, timeslot), group in assigned_room_timeslot.items():
        if len(group) > 1:
            involved = ", ".join([c for _, c in group])
            violations += 10 * (len(group) - 1)
            reasons.append(f"[HARD] Room conflict: {room} used by {involved} at {timeslot}.")

    # HARD: Teacher conflicts
    for tid, schedule in teacher_schedule.items():
        timeslot_count = defaultdict(list)
        for timeslot, class_id in schedule:
            timeslot_count[timeslot].append(class_id)
        for timeslot, classes in timeslot_count.items():
            if len(classes) > 1:
                violations += 10 * (len(classes) - 1)
                reasons.append(
                    f"[HARD] Teacher {teacher_info.get(tid, {}).get('teacher_name', tid)} "
                    f"assigned to multiple classes ({', '.join(classes)}) in {timeslot}."
                )

    # HARD: Workload exceeded
    for tid, load in teacher_load.items():
        maxw = teacher_info.get(tid, {}).get("teacher_max_workload", None)
        if maxw is not None and load > maxw:
            diff = load - maxw
            violations += diff * 2
            reasons.append(
                f"[HARD] Teacher {teacher_info.get(tid, {}).get('teacher_name', tid)} "
                f"exceeded workload ({load} > {maxw})."
            )

    individual.reasons = reasons
    return (violations,)


# ===========================================================
# STEP 3: MUTATION AND CROSSOVER
# ===========================================================
def mutate_individual(individual, rooms, timeslots, indpb=INDPB):
    """Mutate an individual by altering room or timeslot with probability indpb."""
    for i in range(len(individual)):
        if random.random() < indpb:
            if random.random() < 0.6:
                individual[i] = (individual[i][0], random.choice(timeslots))
            else:
                individual[i] = (random.choice(rooms), individual[i][1])
    return (individual,)


def crossover_individual(ind1, ind2):
    """Apply two-point crossover between individuals."""
    if len(ind1) < 2:
        return ind1, ind2
    a = random.randint(1, len(ind1) - 1)
    b = random.randint(1, len(ind1) - 1)
    if a > b:
        a, b = b, a
    ind1[a:b], ind2[a:b] = ind2[a:b], ind1[a:b]
    return ind1, ind2


# ===========================================================
# STEP 4: RUN GENETIC ALGORITHM
# ===========================================================
def execute_genAlgorithm(rooms, timeslots, lesson_instances, rooms_capacity, class_students,
                         teacher_of_lesson, teacher_info, class_grade_map,
                         ngen=80, npop=150):
    """Run the genetic algorithm to allocate schedules."""
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "ScheduleInd"):
        creator.create("ScheduleInd", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.ScheduleInd,
                     lambda: create_individual(rooms, timeslots, len(lesson_instances)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", crossover_individual)
    toolbox.register("mutate", mutate_individual, rooms=rooms, timeslots=timeslots, indpb=INDPB)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_schedule,
                     lesson_instances=lesson_instances,
                     rooms_capacity=rooms_capacity,
                     class_students=class_students,
                     teacher_of_lesson=teacher_of_lesson,
                     teacher_info=teacher_info,
                     class_grade_map=class_grade_map)

    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", min)
    stats.register("avg", lambda fits: sum(fits) / len(fits))

    pop, logbook = algorithms.eaSimple(
        pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=ngen,
        stats=stats, halloffame=hof, verbose=False
    )

    best = hof[0]
    fitness_value = best.fitness.values[0]
    reasons = getattr(best, "reasons", [])
    mapping = {lesson: best[idx] for idx, lesson in enumerate(lesson_instances)}

    if fitness_value > 0 and ngen >= 50:
        print("\nAlgorithm did not eliminate all violations after 50 generations.")
        print("Remaining top issues:\n")
        for reason in reasons[:30]:
            print(f" - {reason}")

    return best, fitness_value, reasons, mapping


# ===========================================================
# STEP 5: SOLUTION ANALYSIS
# ===========================================================
def analyze_solution(best, fitness, reasons, mapping, lesson_instances, df_classes,
                     rooms_capacity, teacher_assignments, teacher_info, schedule_meta):
    """Generate a readable report for the final schedule."""
    print("\n=== FINAL SCHEDULE REPORT ===\n")

    schedule_by_class = {}
    for lesson in lesson_instances:
        class_id, subject, inst = lesson.split("::")
        room, timeslot = mapping[lesson]
        meta = schedule_meta.get(timeslot, {"weekday": "?", "period": timeslot, "label": timeslot})
        teacher = teacher_assignments.get(lesson, None)
        teacher_name = teacher_info.get(teacher, {}).get("teacher_name", teacher) if teacher else "N/A"

        schedule_by_class.setdefault(class_id, []).append({
            "subject": subject,
            "room": room,
            "timeslot": timeslot,
            "weekday": meta.get("weekday"),
            "period": meta.get("period"),
            "label": meta.get("label"),
            "teacher": teacher_name
        })

    for class_id, slots in schedule_by_class.items():
        row = df_classes[df_classes["class_id"] == class_id]
        if not row.empty:
            class_name = row.iloc[0].get("class_name", class_id)
            num_students = row.iloc[0].get("num_students", "N/A")
        else:
            class_name = class_id
            num_students = "N/A"

        print(f"Class {class_id} ({class_name}) - {num_students} students")
        slots_sorted = sorted(slots, key=lambda s: (weekday_order(s["weekday"]), s["label"]))
        for s in slots_sorted:
            print(f"   - {s['weekday']} | {s['label']} | Room: {s['room']} | "
                  f"Subject: {s['subject']} | Teacher: {s['teacher']}")
        print("-" * 70)


def weekday_order(weekday):
    """Return weekday index for sorting."""
    order = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4,
        "Segunda": 0, "Terça": 1, "Quarta": 2, "Quinta": 3, "Sexta": 4,
        "Saturday": 5, "Sunday": 6
    }
    return order.get(weekday, 99)
