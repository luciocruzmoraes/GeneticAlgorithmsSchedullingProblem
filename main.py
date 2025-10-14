import os
import pandas as pd
import random
from genetic_algorithm import execute_genAlgorithm, analyze_solution
from linear_regression import (
    load_historical_data,
    train_models_by_grade,
    predict_by_grade,
    generate_classes_from_predictions
)

random.seed(42)


def ensure_data_directory():
    """Garante que a pasta 'data' exista."""
    if not os.path.isdir("data"):
        os.makedirs("data")
        print("Pasta 'data' criada automaticamente.")


def load_inputs():
    """Carrega os arquivos necessários para o algoritmo genético."""
    if not os.path.exists("data/classes_data.csv"):
        raise FileNotFoundError("O arquivo 'data/classes_data.csv' não foi encontrado. "
                                "Verifique se a regressão linear o gerou corretamente.")

    df_classes = pd.read_csv("data/classes_data.csv")

    required_files = [
        "data/rooms_data.csv",
        "data/teachers_data.csv",
        "data/schedules_data.csv"
    ]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {file}")

    df_rooms = pd.read_csv("data/rooms_data.csv")
    df_teachers = pd.read_csv("data/teachers_data.csv")
    df_schedules = pd.read_csv("data/schedules_data.csv")

    if os.path.exists("data/teacher_assignments.csv"):
        df_assign = pd.read_csv("data/teacher_assignments.csv")
    else:
        df_assign = None

    return df_classes, df_rooms, df_teachers, df_assign, df_schedules


def expand_lessons(df_classes):
    """Expande as turmas conforme o número de aulas por semana."""
    lesson_instances = []
    for _, row in df_classes.iterrows():
        class_id = row["class_id"]
        subject = row.get("subject", "Geral")
        lessons = int(row.get("lessons_per_week", 1))
        for i in range(lessons):
            lesson_instances.append(f"{class_id}::{subject}::{i + 1}")
    return lesson_instances


def build_timeslots(df_schedules):
    """Cria lista de horários disponíveis e metadados."""
    timeslots = df_schedules["schedule_id"].tolist()
    schedule_meta = {}
    for _, r in df_schedules.iterrows():
        tid = r["schedule_id"]
        shift = r.get("shift", "")
        label = f"{shift[0] if shift else ''}{r.get('period', '')}".replace(" ", "")
        schedule_meta[tid] = {
            "weekday": r.get("weekday", ""),
            "shift": shift,
            "period": r.get("period", ""),
            "label": label
        }
    return timeslots, schedule_meta


def build_rooms_capacity(df_rooms):
    """Cria um dicionário com a capacidade de cada sala."""
    return dict(zip(df_rooms["room_id"], df_rooms["capacity"]))


def build_class_students(df_classes):
    """Retorna dicionário de alunos por turma."""
    class_students = {}
    for _, row in df_classes.iterrows():
        cid = row["class_id"]
        class_students[cid] = int(row["num_students"])
    return class_students


def build_teacher_assignment_map(lesson_instances, df_assignments, df_teachers):
    """Associa professores às aulas (aleatoriamente se não houver mapeamento fixo)."""
    teacher_of_lesson = {}
    teachers_info = {}

    for _, r in df_teachers.iterrows():
        teachers_info[r["teacher_id"]] = {
            "name": r.get("name"),
            "available_morning": bool(r.get("available_morning", True)),
            "available_afternoon": bool(r.get("available_afternoon", True)),
            "available_evening": bool(r.get("available_evening", False)),
            "max_workload": int(r.get("max_workload", 40)),
            "main_subject": r.get("main_subject", "Geral")
        }

    teacher_ids = list(teachers_info.keys())
    for idx, lesson in enumerate(lesson_instances):
        teacher_of_lesson[idx] = random.choice(teacher_ids)

    return teacher_of_lesson, teachers_info


def main():
    print("\n" + "=" * 70)
    print("INICIANDO SISTEMA DE ALOCAÇÃO DE HORÁRIOS")
    print("=" * 70 + "\n")
    
    ensure_data_directory()

    # ===========================================================
    # STEP 0: LINEAR REGRESSION - GERAÇÃO DE CLASSES
    # ===========================================================
    print("=" * 70)
    print("STEP 0: LINEAR REGRESSION - Gerando classes_data.csv")
    print("=" * 70)

    classes_file = "data/classes_data.csv"
    historical_file = "data/historical_enrollment_data.csv"
    
    # Verificar se classes_data.csv já existe
    if os.path.isfile(classes_file):
        print(f"  Arquivo {classes_file} já existe.")
        user_input = input("Deseja regerar o arquivo? (s/n): ").strip().lower()
        if user_input != 's':
            print("Usando arquivo existente.\n")
        else:
            print("Regenerando arquivo...")
            if not os.path.isfile(historical_file):
                print(f"ERRO: Arquivo {historical_file} não encontrado.")
                print("Não é possível regenerar sem dados históricos.")
                return
            
            df_hist = load_historical_data(historical_file)
            if df_hist is None or df_hist.empty:
                print("ERRO: DataFrame histórico vazio ou inválido.")
                return
            
            print(f"Dados históricos carregados: {len(df_hist)} registros")
            models = train_models_by_grade(df_hist)
            print(f"Modelos treinados para {len(models)} séries")
            
            predictions = predict_by_grade(models)
            print(f"Previsões geradas para {len(predictions)} séries")
            
            generate_classes_from_predictions(predictions)
            print("classes_data.csv regerado com sucesso.\n")
    else:
        # Arquivo não existe, precisa criar
        print(f"Arquivo {classes_file} não encontrado. Gerando...")
        
        if not os.path.isfile(historical_file):
            print(f"ERRO: Arquivo {historical_file} não encontrado.")
            print("Por favor, crie o arquivo com os dados históricos de matrículas.")
            print(f"\nVerificação: pasta 'data' existe? {os.path.isdir('data')}")
            if os.path.isdir('data'):
                print(f"Conteúdo da pasta 'data': {os.listdir('data')}")
            return
        
        # Executar a regressão linear
        print("Carregando dados históricos...")
        df_hist = load_historical_data(historical_file)
        
        if df_hist is None or df_hist.empty:
            print(" ERRO: DataFrame histórico vazio ou inválido.")
            return
        
        print(f"Dados históricos carregados: {len(df_hist)} registros")
        models = train_models_by_grade(df_hist)
        print(f"Modelos treinados para {len(models)} séries")
        
        predictions = predict_by_grade(models)
        print(f"Previsões geradas para {len(predictions)} séries")
        
        generate_classes_from_predictions(predictions)
        print("classes_data.csv gerado com sucesso.\n")
        
        # Verificar se o arquivo foi realmente criado
        if not os.path.isfile(classes_file):
            print(f" ERRO: {classes_file} não foi criado apesar da execução do STEP 0!")
            if os.path.isdir('data'):
                print(f"Conteúdo da pasta 'data': {os.listdir('data')}")
            return
    
    print(f"Arquivo {classes_file} pronto para uso!\n")

    # ===========================================================
    # STEP 1: LOAD INPUTS
    # ===========================================================
    print("=" * 70)
    print("STEP 1: LOAD INPUTS")
    print("=" * 70)

    try:
        df_classes, df_rooms, df_teachers, df_assign, df_schedules = load_inputs()
        print(f"{len(df_classes)} turmas carregadas de classes_data.csv.")
        print(f"{len(df_rooms)} salas carregadas.")
        print(f" {len(df_teachers)} professores carregados.")
        print(f" {len(df_schedules)} horários carregados.")
    except Exception as e:
        print(f"ERRO ao carregar os arquivos de entrada: {e}")
        return

    # ===========================================================
    # STEP 2: GENETIC ALGORITHM
    # ===========================================================
    print("=" * 70)
    print("STEP 2: RUN GENETIC ALGORITHM")
    print("=" * 70)

    lesson_instances = expand_lessons(df_classes)
    print(f"Total de aulas a serem alocadas: {len(lesson_instances)}")
    
    timeslots, schedule_meta = build_timeslots(df_schedules)
    rooms = df_rooms["room_id"].tolist()
    rooms_capacity = build_rooms_capacity(df_rooms)
    class_students = build_class_students(df_classes)
    teacher_of_lesson, teacher_info = build_teacher_assignment_map(
        lesson_instances, df_assign, df_teachers
    )

    print("Executando algoritmo genético...")
    best, fitness, reasons, mapping = execute_genAlgorithm(
        rooms=rooms,
        timeslots=timeslots,
        lesson_instances=lesson_instances,
        rooms_capacity=rooms_capacity,
        class_students=class_students,
        teacher_of_lesson=teacher_of_lesson,
        teacher_info=teacher_info,
        class_grade_map={},
        ngen=80,
        npop=150
    )

    # ===========================================================
    # STEP 3: REPORT
    # ===========================================================
    print("=" * 70)
    print("STEP 3: REPORT AND ANALYSIS")
    print("=" * 70)

    analyze_solution(
        best,
        fitness,
        reasons,
        mapping,
        lesson_instances,
        df_classes,
        rooms_capacity,
        {lesson: teacher_of_lesson[idx] for idx, lesson in enumerate(lesson_instances)},
        teacher_info,
        schedule_meta
    )
    
    print("=" * 70)
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()