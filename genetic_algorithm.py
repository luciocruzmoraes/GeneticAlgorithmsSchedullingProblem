import random
from collections import defaultdict
from deap import base, creator, tools, algorithms

random.seed(42)


def create_individual(rooms, timeslots, length):
    """
    Individual: list of length `length`, each gene is (room_id, timeslot_id)
    """
    return [(random.choice(rooms), random.choice(timeslots)) for _ in range(length)]


def evaluate_schedule(individual, lesson_instances, rooms_capacity, class_students,
                      teacher_of_lesson, teacher_info, timeslot_to_shift):
    """
    Evaluate a schedule represented by `individual`.
    Returns (violations, ) as DEAP expects a tuple.
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

        # Sala existe?
        if room not in rooms_capacity:
            violations += 5
            reasons.append(f"Sala {room} não encontrada no cadastro.")
            continue

        # Capacidade da sala
        if rooms_capacity[room] < students:
            diff = students - rooms_capacity[room]
            violations += diff
            reasons.append(
                f"Sala {room} (capacidade {rooms_capacity[room]}) recebeu turma {class_id} com {students} alunos."
            )

        # Conflito de sala
        assigned_room_timeslot[(room, timeslot)].append((idx, class_id))

        # Disponibilidade e carga do professor
        teacher_id = teacher_of_lesson.get(idx, None)
        if teacher_id:
            teacher_load[teacher_id] += 1
            tinfo = teacher_info.get(teacher_id, {})
            shift_code = timeslot_to_shift.get(timeslot, "?")
            if shift_code == "M" and not tinfo.get("available_morning", False):
                violations += 1
                reasons.append(
                    f"Professor {tinfo.get('name', teacher_id)} ({teacher_id}) não disponível de manhã, "
                    f"mas foi alocado para turma {class_id} no horário {timeslot}."
                )
            if shift_code == "A" and not tinfo.get("available_afternoon", False):
                violations += 1
                reasons.append(
                    f"Professor {tinfo.get('name', teacher_id)} ({teacher_id}) não disponível à tarde, "
                    f"mas foi alocado para turma {class_id} no horário {timeslot}."
                )
            if shift_code == "E" and not tinfo.get("available_evening", False):
                violations += 1
                reasons.append(
                    f"Professor {tinfo.get('name', teacher_id)} ({teacher_id}) não disponível à noite, "
                    f"mas foi alocado para turma {class_id} no horário {timeslot}."
                )

    # Conflitos de sala
    for (room, timeslot), lst in assigned_room_timeslot.items():
        if len(lst) > 1:
            involved = ", ".join([f"{c}" for (_, c) in lst])
            violations += 5 * (len(lst) - 1)
            reasons.append(
                f"Conflito de agendamento: sala {room} foi alocada simultaneamente "
                f"para as turmas {involved} no horário {timeslot}."
            )

    # Sobrecarga de professor
    for tid, load in teacher_load.items():
        maxw = teacher_info.get(tid, {}).get("max_workload", None)
        if maxw is not None and load > maxw:
            diff = load - maxw
            violations += diff
            reasons.append(
                f"Professor {teacher_info.get(tid, {}).get('name', tid)} ({tid}) excedeu carga ({load} > {maxw})."
            )

    individual.reasons = reasons
    return (violations,)


def mutate_individual(individual, rooms, timeslots, indpb=0.05):
    """Mutation: for each gene, with probability indpb, change room or timeslot."""
    for i in range(len(individual)):
        if random.random() < indpb:
            if random.random() < 0.6:  # troca horário
                individual[i] = (individual[i][0], random.choice(timeslots))
            else:  # troca sala
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
    Runs GA. Returns (best_individual, fitness_value, reasons, mapping of lesson->assignment)
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
      - Problemas de Sala
      - Conflitos de agendamento
      - Problemas de Professores
      - Resumo de alocação por turma (inclui série/ano)
    """
    problemas_sala = [r for r in reasons if ("Sala " in r or "capacidade" in r)]
    problemas_conflito = [r for r in reasons if "Conflito" in r or "conflito" in r]
    problemas_professor = [r for r in reasons if "Professor" in r or "professor" in r]

    print("\n=== RELATÓRIO DE ALOCAÇÃO ===\n")

    print("Problemas de Sala:")
    print(" - Nenhum problema encontrado." if not problemas_sala else "\n".join(f" - {r}" for r in problemas_sala))

    print("\nConflitos de Agendamento:")
    print(" - Nenhum conflito encontrado." if not problemas_conflito else "\n".join(f" - {r}" for r in problemas_conflito))

    print("\nProblemas de Professores:")
    print(" - Nenhum problema encontrado." if not problemas_professor else "\n".join(f" - {r}" for r in problemas_professor))

    print("\nResumo de Alocação Semanal por Turma:\n")

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

        print(f"Turma {class_id} ({class_name}) - {grade} - {num_students} alunos")
        slots_sorted = sorted(slots, key=lambda s: (weekday_order(s["weekday"]), s["label"]))
        for s in slots_sorted:
            print(f"   - {s['weekday']} | {s['label']} | {s['shift']} | Sala: {s['room']} | "
                  f"Disciplina: {s['subject']} | Prof.: {s['teacher']}")
        print("-" * 70)


def weekday_order(weekday):
    order = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4,
        "Segunda": 0, "Terça": 1, "Quarta": 2, "Quinta": 3, "Sexta": 4
    }
    return order.get(weekday, 99)
