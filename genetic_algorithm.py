import random
from collections import defaultdict
from deap import base, creator, tools, algorithms

random.seed(42)


def create_individual(rooms, timeslots, length):
    """Cria um indivíduo: lista de (room_id, timeslot_id)."""
    return [(random.choice(rooms), random.choice(timeslots)) for _ in range(length)]


# ===========================================================
# FUNÇÃO DE FITNESS — versão refinada com preferências reais
# ===========================================================
def evaluate_schedule(individual, lesson_instances, rooms_capacity, class_students,
                      teacher_of_lesson, teacher_info, class_grade_map):
    """
    Avalia um cronograma (indivíduo) considerando restrições rígidas e preferências.

    HARD CONSTRAINTS:
    - Sala deve existir e ter capacidade suficiente
    - Conflito de sala (mesmo horário/sala)
    - Conflito de professor (mesmo horário)
    - Limite de carga horária

    SOFT CONSTRAINTS:
    - Preferência de horários (teacher_preferred_periods)
    - Preferência de disciplinas (teacher_favorites_subject)
    """

    violations = 0
    reasons = []

    assigned_room_timeslot = defaultdict(list)
    teacher_load = defaultdict(int)
    teacher_schedule = defaultdict(list)

    for idx, gene in enumerate(individual):
        room, timeslot = gene
        lesson_id = lesson_instances[idx]
        class_id = lesson_id.split("::")[0]
        subject = lesson_id.split("::")[1]
        students = class_students[class_id]

        # ---- HARD: Sala existente ----
        if room not in rooms_capacity:
            violations += 10
            reasons.append(f"[HARD] Sala {room} não encontrada no registro.")
            continue

        # ---- HARD: Capacidade da sala ----
        if rooms_capacity[room] < students:
            diff = students - rooms_capacity[room]
            violations += diff * 2
            reasons.append(
                f"[HARD] Sala {room} (capacidade {rooms_capacity[room]}) "
                f"atribuída à turma {class_id} com {students} alunos."
            )

        assigned_room_timeslot[(room, timeslot)].append((idx, class_id))

        # ---- Professor responsável ----
        teacher_id = teacher_of_lesson.get(idx, None)
        if teacher_id:
            teacher_load[teacher_id] += 1
            teacher_schedule[teacher_id].append((timeslot, class_id))
            tinfo = teacher_info.get(teacher_id, {})

            # SOFT: Preferência de horários
            preferred_periods_str = tinfo.get("teacher_preferred_periods", "")
            preferred_periods = [p.strip() for p in preferred_periods_str.split(",") if p.strip()]
            if preferred_periods and timeslot not in preferred_periods:
                violations += 0.5
                reasons.append(
                    f"[SOFT] {tinfo.get('teacher_name', teacher_id)} prefere {preferred_periods} "
                    f"mas foi alocado em {timeslot}."
                )

            # SOFT: Preferência de disciplinas
            fav_subjects_str = tinfo.get("teacher_favorites_subject", "")
            fav_subjects = [s.strip() for s in fav_subjects_str.split(",") if s.strip()]
            if fav_subjects and subject not in fav_subjects:
                violations += 0.5
                reasons.append(
                    f"[SOFT] {tinfo.get('teacher_name', teacher_id)} prefere disciplinas {fav_subjects} "
                    f"mas foi alocado em {subject}."
                )

    # ---- HARD: Conflitos de sala ----
    for (room, timeslot), lst in assigned_room_timeslot.items():
        if len(lst) > 1:
            involved = ", ".join([f"{c}" for (_, c) in lst])
            violations += 10 * (len(lst) - 1)
            reasons.append(f"[HARD] Conflito de sala: {room} usada simultaneamente por {involved} em {timeslot}.")

    # ---- HARD: Conflitos de professor ----
    for tid, schedule in teacher_schedule.items():
        timeslot_count = defaultdict(list)
        for timeslot, class_id in schedule:
            timeslot_count[timeslot].append(class_id)
        for timeslot, classes in timeslot_count.items():
            if len(classes) > 1:
                violations += 10 * (len(classes) - 1)
                reasons.append(
                    f"[HARD] Professor {teacher_info.get(tid, {}).get('teacher_name', tid)} com múltiplas turmas "
                    f"({', '.join(classes)}) no mesmo horário ({timeslot})."
                )

    # ---- HARD: Carga horária excedida ----
    for tid, load in teacher_load.items():
        maxw = teacher_info.get(tid, {}).get("teacher_max_workload", None)
        if maxw is not None and load > maxw:
            diff = load - maxw
            violations += diff * 2
            reasons.append(
                f"[HARD] Professor {teacher_info.get(tid, {}).get('teacher_name', tid)} excedeu carga horária ({load} > {maxw})."
            )

    individual.reasons = reasons
    return (violations,)


# ===========================================================
# Mutação e Crossover
# ===========================================================
def mutate_individual(individual, rooms, timeslots, indpb=0.05):
    """Mutação: altera sala ou horário com probabilidade indpb."""
    for i in range(len(individual)):
        if random.random() < indpb:
            if random.random() < 0.6:
                individual[i] = (individual[i][0], random.choice(timeslots))
            else:
                individual[i] = (random.choice(rooms), individual[i][1])
    return (individual,)


def crossover_individual(ind1, ind2):
    """Crossover de dois pontos."""
    if len(ind1) < 2:
        return ind1, ind2
    a = random.randint(1, len(ind1) - 1)
    b = random.randint(1, len(ind1) - 1)
    if a > b:
        a, b = b, a
    ind1[a:b], ind2[a:b] = ind2[a:b], ind1[a:b]
    return ind1, ind2


# ===========================================================
# Execução do Algoritmo Genético
# ===========================================================
def execute_genAlgorithm(rooms, timeslots, lesson_instances, rooms_capacity, class_students,
                         teacher_of_lesson, teacher_info, class_grade_map,
                         ngen=80, npop=150):
    """Executa o algoritmo genético de alocação."""
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
                     class_grade_map=class_grade_map)

    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("min", min)
    stats.register("avg", lambda fits: sum(fits) / len(fits))

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=ngen,
                                      stats=stats, halloffame=hof, verbose=False)

    best = hof[0]
    fitness_value = best.fitness.values[0]
    reasons = getattr(best, "reasons", [])
    mapping = {lesson: best[idx] for idx, lesson in enumerate(lesson_instances)}

    if fitness_value > 0 and ngen >= 50:
        print("\nO algoritmo não conseguiu eliminar todas as violações em 50 iterações.")
        print("  Principais problemas ainda existentes:\n")
        for reason in reasons[:30]:
            print(f" - {reason}")

    return best, fitness_value, reasons, mapping


# ===========================================================
# Análise de Solução e Ordenação de Dias
# ===========================================================
def analyze_solution(best, fitness, reasons, mapping, lesson_instances, df_classes,
                     rooms_capacity, teacher_assignments, teacher_info, schedule_meta):
    """Gera relatório detalhado sem campo 'grade'."""
    print("\n=== RELATÓRIO DE ALOCAÇÃO ===\n")

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

        print(f"Turma {class_id} ({class_name}) - {num_students} alunos")
        slots_sorted = sorted(slots, key=lambda s: (weekday_order(s["weekday"]), s["label"]))
        for s in slots_sorted:
            print(f"   - {s['weekday']} | {s['label']} | Sala: {s['room']} | "
                  f"Disciplina: {s['subject']} | Prof.: {s['teacher']}")
        print("-" * 70)


def weekday_order(weekday):
    order = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4,
        "Segunda": 0, "Terça": 1, "Quarta": 2, "Quinta": 3, "Sexta": 4
    }
    return order.get(weekday, 99)