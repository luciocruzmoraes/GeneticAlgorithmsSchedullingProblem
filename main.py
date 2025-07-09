from regressao import treinar_modelo_regressao, gerar_dados_historicos
from genetico import configurar_algoritmo_genetico, executar_algoritmo, gerar_dados_atuais

def verificar_viabilidade(turmas, alunos_turma, capacidade_salas):
    turmas_inviaveis = []
    for turma in turmas:
        alunos = alunos_turma[turma]
        if not any(cap >= alunos for cap in capacidade_salas.values()):
            turmas_inviaveis.append(turma)
    if turmas_inviaveis:
        print("⚠️ As seguintes turmas NÃO cabem em nenhuma sala disponível:")
        for t in turmas_inviaveis:
            print(f"  - {t} ({alunos_turma[t]} alunos)")
        return False
    return True

def verificar_salas_distintas(solucao):
    salas = [sala for sala, horario in solucao]
    return len(salas) == len(set(salas))

def mostrar_escala_professores(solucao, turmas, professores):
    # Mapeia turma → horário
    turma_horario = {}
    for i, (sala, horario) in enumerate(solucao):
        turma = turmas[i]
        turma_horario[turma] = horario

    print("\n📅 Escala de horários por professor:")
    for prof, dados in professores.items():
        print(f"\nProfessor {prof}:")
        for turma in dados["turmas"]:
            horario = turma_horario.get(turma, None)
            if horario:
                print(f"  - Turma {turma}: Horário {horario}")
            else:
                print(f"  - Turma {turma}: Não alocada")

def main():
    # 1. Treinar modelo de regressão com dados históricos simulados
    df_hist = gerar_dados_historicos()
    modelo_rl = treinar_modelo_regressao(df_hist)

    # 2. Gerar dados atuais simulados
    dados = gerar_dados_atuais()

    # 3. Pré-verificação de viabilidade estrutural
    if not verificar_viabilidade(dados["turmas"], dados["alunos_turma"], dados["capacidade_salas"]):
        print("Por favor, ajuste os dados para continuar.")
        return

    # 4. Configurar o algoritmo genético
    toolbox = configurar_algoritmo_genetico(
        dados["turmas"], dados["salas"], dados["horarios"],
        dados["capacidade_salas"], dados["alunos_turma"], dados["professores"]
    )

    # 5. Executar até encontrar uma solução sem violação
    solucao_encontrada = False
    tentativas = 0

    while not solucao_encontrada:
        tentativas += 1
        print(f"\n🔁 Tentativa #{tentativas} - Evoluindo população...")

        hof, log = executar_algoritmo(toolbox, ngen=50, npop=100)

        for ind in hof:
            if ind.fitness.values[0] == 0.0:  
                solucao_atual = [(sala, horario) for sala, horario in ind]

                if verificar_salas_distintas(solucao_atual):
                    solucao_encontrada = True
                    melhor_resultado = [
                        (dados["turmas"][i], sala, horario) for i, (sala, horario) in enumerate(solucao_atual)
                    ]
                    fitness = ind.fitness.values

                    print("\n✅ Melhor Solução Encontrada:")
                    for turma, sala, horario in melhor_resultado:
                        print(f"  Turma {turma} → Sala {sala} - Horário {horario}")

                    print(f"\nFitness (violações, taxa de ocupação global): ({fitness[0]}, {fitness[1]:.2f}%)")

                    print("\n📊 Ocupação por sala:")
                    alunos_por_sala = {}
                    for i, (sala, _) in enumerate(solucao_atual):
                        turma = dados["turmas"][i]
                        alunos = dados["alunos_turma"][turma]
                        alunos_por_sala[sala] = alunos_por_sala.get(sala, 0) + alunos

                    for sala, alunos in alunos_por_sala.items():
                        capacidade = dados["capacidade_salas"].get(sala, 0)
                        ocupacao = (alunos / capacidade) * 100 if capacidade > 0 else 0.0
                        print(f"  Sala {sala}: {alunos} alunos / {capacidade} capacidade → {ocupacao:.2f}%")

                    # Mostrar escala de horários por professor
                    mostrar_escala_professores(solucao_atual, dados["turmas"], dados["professores"])
                    return

if __name__ == "__main__":
    main()
