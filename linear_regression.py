import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os
import random
import joblib

# ===========================================================
# 1. Carrega e prepara dados históricos
# ===========================================================
def load_historical_data(file_path="data/historical_enrollment_data.csv"):
    if not os.path.exists(file_path):
        print(f"Arquivo {file_path} não encontrado.")
        return None
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    # Mantém apenas colunas relevantes
    df = df[[
        "year", "grade", "regional_gdp",
        "unemployment_rate", "education_investment", "total_enrollments"
    ]]

    df["normalized_year"] = df["year"] - df["year"].min()
    df["per_capita_investment"] = df["education_investment"] / df["total_enrollments"]
    df["economic_index"] = df["regional_gdp"] / (df["unemployment_rate"] + 1)

    return df


# ===========================================================
# 2. Treina modelos por série
# ===========================================================
def train_models_by_grade(df):
    if df is None or df.empty:
        print("Dados históricos inválidos ou vazios.")
        return {}

    models = {}
    
    grades = df["grade"].unique()

    for grade in grades:
        subset = df[df["grade"] == grade]
        X = subset[[
            "normalized_year",
            "regional_gdp",
            "unemployment_rate",
            "education_investment",
            "per_capita_investment",
            "economic_index"
        ]]
        y = subset["total_enrollments"]

        # É importante criar um novo scaler para cada série
        # para que o ajuste seja específico para a distribuição de dados daquela série.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression().fit(X_scaled, y)
        models[grade] = {"model": model, "scaler": scaler}

    joblib.dump(models, "modelos_por_grade.pkl")
    print(f"{len(models)} modelos treinados com sucesso (1 por série).")
    return models


# ===========================================================
# 3. Faz previsões por série
# ===========================================================
def predict_by_grade(models, future_year=2025, pid_regional=70000,
                     unemployment_rate=7.0, education_investment=1300000):
    results = {}
    # Ano base usado na normalização durante o treinamento
    base_year = 2020 

    for grade, data in models.items():
        model = data["model"]
        scaler = data["scaler"]

        normalized_year = future_year - base_year
        
        # A estimativa de alunos aqui é um "chute" para calcular o investimento per capita.
        # Isso introduz uma circularidade. Uma abordagem melhor seria iterar ou usar uma média.
        # Por simplicidade, mantemos um valor fixo, mas ciente da limitação.
        estimated_students_for_calc = 200 
        per_capita_investment = education_investment / estimated_students_for_calc
        economic_index = pid_regional / (unemployment_rate + 1)

        X_new = np.array([[
            normalized_year,
            pid_regional,
            unemployment_rate,
            education_investment,
            per_capita_investment,
            economic_index
        ]])
        
        X_scaled = scaler.transform(X_new)
        prediction = model.predict(X_scaled)[0]
        results[grade] = max(0, int(prediction))

    print("\nPrevisão de matrículas por série:")
    for g, v in results.items():
        print(f" - {g}: {v} alunos previstos")

    return results


# ===========================================================
# 4. Gera classes com nomes reais (ex: 6ºA, 6ºB, 1ª Série EMA)
# ===========================================================
def generate_classes_from_predictions(predictions):
    subjects = ["Portuguese", "Mathematics", "History", "Geography",
                "Science", "English", "Arts", "Physical Education"]
    rows = []
    class_counter = 1

    for grade, total_students in predictions.items():
        remaining = total_students
        section_letter = ord("A")  # começa em 'A'

        while remaining > 0:
            num_students = min(random.randint(25, 40), remaining)
            
            # Lógica para criar um nome de turma mais realista
            if "Ano" in grade:
                # Ex: "6º Ano" vira "6ºA"
                class_name = f"{grade.split()[0]}{chr(section_letter)}"
            else:
                # Ex: "1ª Série EM" vira "1ª Série EM A"
                class_name = f"{grade} {chr(section_letter)}"

            class_id = f"C{class_counter:03d}"

            for subject in subjects:
                lessons_per_week = random.randint(2, 5)
                rows.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "education_level": grade,
                    "num_students": num_students,
                    "subject": subject,
                    "lessons_per_week": lessons_per_week
                })

            remaining -= num_students
            section_letter += 1
            class_counter += 1

    os.makedirs("data", exist_ok=True)
    df_classes = pd.DataFrame(rows)
    df_classes.to_csv("data/classes_data.csv", index=False)
    print(f"\nArquivo 'data/classes_data.csv' gerado com {len(df_classes['class_id'].unique())} turmas.")


# ===========================================================
# 5. Execução completa
# ===========================================================
def main():
    df_hist = load_historical_data("data/historical_enrollment_data.csv")
    models = train_models_by_grade(df_hist)
    predictions = predict_by_grade(models)
    generate_classes_from_predictions(predictions)


if __name__ == "__main__":
    main()