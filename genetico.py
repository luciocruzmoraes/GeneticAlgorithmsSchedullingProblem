import random
from deap import base, creator, tools, algorithms

def gerar_dados_atuais():
    return {
        "turmas": ["T1", "T2", "T3", "T4", "T5"],
        "salas": ["S1", "S2", "S3", "S4", "S5"],
        "capacidade_salas": {
            "S1": 45,
            "S2": 30,
            "S3": 25,
            "S4": 40,
            "S5": 35
        },
        "alunos_turma": {"T1": 35, "T2": 25, "T3": 20, "T4": 45, "T5": 15},
        "horarios": ["H1", "H2", "H3"],
        "professores": {
            "P1": {"turmas": ["T1", "T4"], "preferencias": ["H1", "H2"], "restricoes": ["H3"]},
            "P2": {"turmas": ["T2", "T3"], "preferencias": ["H2"], "restricoes": []},
            "P3": {"turmas": ["T5"], "preferencias": ["H1", "H3"], "restricoes": ["H2"]},
        }
    }

def avaliar_individuo(ind, turmas, capacidade_salas, alunos_turma, professores):
    conflitos = 0
    excesso = 0
    uso_sala_horario = {}
    uso_salas = set()
    penalidade_pref = 0
    restricoes_violadas = 0
    total_alunos_alocados = 0
    total_capacidade_utilizada = 0

    for i, (sala, horario) in enumerate(ind):
        turma = turmas[i]
        chave = (sala, horario)

        # Penaliza se a mesma sala + horário já estiver em uso
        if chave in uso_sala_horario:
            conflitos += 1
        else:
            uso_sala_horario[chave] = turma

        # Penaliza se a mesma sala for usada por mais de uma turma (independente do horário)
        if sala in uso_salas:
            conflitos += 1
        else:
            uso_salas.add(sala)

        # Verifica excesso de alunos para a capacidade da sala
        if alunos_turma[turma] > capacidade_salas[sala]:
            excesso += 1
        else:
            total_alunos_alocados += alunos_turma[turma]
            total_capacidade_utilizada += capacidade_salas[sala]

        # Penalidade por preferências e restrições dos professores
        for prof, dados in professores.items():
            if turma in dados["turmas"]:
                if horario in dados.get("restricoes", []):
                    restricoes_violadas += 1
                if horario not in dados.get("preferencias", []):
                    penalidade_pref += 1
                break

    violacoes = conflitos + excesso + restricoes_violadas
    taxa_ocup = (total_alunos_alocados / total_capacidade_utilizada) * 100 if total_capacidade_utilizada > 0 and violacoes == 0 else 0.0

    return violacoes + penalidade_pref, taxa_ocup

def mutar_individuo(individual, salas, horarios, prob=0.2):
    for i in range(len(individual)):
        if random.random() < prob:
            individual[i] = (random.choice(salas), random.choice(horarios))
    return individual,

def configurar_algoritmo_genetico(turmas, salas, horarios, capacidade_salas, alunos_turma, professores):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_pair", lambda: (random.choice(salas), random.choice(horarios)))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_pair, n=len(turmas))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_fn(ind):
        return avaliar_individuo(ind, turmas, capacidade_salas, alunos_turma, professores)

    toolbox.register("evaluate", eval_fn)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutar_individuo, salas=salas, horarios=horarios)
    toolbox.register("select", tools.selNSGA2)

    return toolbox

def executar_algoritmo(toolbox, ngen=40, npop=50):
    pop = toolbox.population(n=npop)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", lambda x: tuple(map(min, zip(*x))))
    stats.register("avg", lambda x: tuple(map(lambda y: sum(y)/len(y), zip(*x))))

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2,
                                       ngen=ngen, stats=stats, halloffame=hof, verbose=False)
    return hof, logbook

if __name__ == "__main__":
    dados = gerar_dados_atuais()
    toolbox = configurar_algoritmo_genetico(
        dados["turmas"],
        dados["salas"],
        dados["horarios"],
        dados["capacidade_salas"],
        dados["alunos_turma"],
        dados["professores"]
    )
    hof, log = executar_algoritmo(toolbox)

    print("Melhores soluções:")
    for sol in hof:
        print(sol, sol.fitness.values)
