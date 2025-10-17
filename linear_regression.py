import os
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib


def load_historical_data(file_path="data/historical_enrollment_data.csv"):
    """Load and prepare historical enrollment data."""
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    required_columns = [
        "year", "grade", "regional_gdp",
        "unemployment_rate", "education_investment",
        "total_enrollments"
    ]
    df = df[required_columns]

    df["normalized_year"] = df["year"] - df["year"].min()
    df["per_capita_investment"] = df["education_investment"] / df["total_enrollments"]
    df["economic_index"] = df["regional_gdp"] / (df["unemployment_rate"] + 1)

    return df


def train_models_by_grade(df):
    """Train a linear regression model for each grade."""
    if df is None or df.empty:
        print("Invalid or empty historical data.")
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

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression().fit(X_scaled, y)
        models[grade] = {"model": model, "scaler": scaler}

    joblib.dump(models, "models_by_grade.pkl")
    print(f"{len(models)} models successfully trained.")
    return models


def predict_by_grade(models, future_year=2025, regional_gdp=70000,
                     unemployment_rate=7.0, education_investment=1300000):
    """Predict enrollments for each grade in a given future year."""
    if not models:
        print("No trained models found.")
        return {}

    results = {}
    base_year = 2020

    for grade, data in models.items():
        model = data["model"]
        scaler = data["scaler"]

        normalized_year = future_year - base_year
        estimated_students = 200
        per_capita_investment = education_investment / estimated_students
        economic_index = regional_gdp / (unemployment_rate + 1)

        X_new = np.array([[
            normalized_year,
            regional_gdp,
            unemployment_rate,
            education_investment,
            per_capita_investment,
            economic_index
        ]])

        X_scaled = scaler.transform(X_new)
        prediction = model.predict(X_scaled)[0]
        results[grade] = max(0, int(prediction))

    print("\nPredicted enrollments per grade:")
    for grade, value in results.items():
        print(f" - {grade}: {value} students expected")

    return results


def generate_classes_from_predictions(predictions):
    """Generate class data based on predicted enrollments."""
    if not predictions:
        print("No predictions available to generate classes.")
        return

    subjects = [
        "Portuguese", "Mathematics", "History", "Geography",
        "Science", "English", "Arts", "Physical Education"
    ]
    rows = []
    class_counter = 1

    for grade, total_students in predictions.items():
        remaining = total_students
        section_letter = ord("A")

        while remaining > 0:
            num_students = min(random.randint(25, 40), remaining)

            if "Year" in grade or "Ano" in grade:
                class_name = f"{grade.split()[0]}{chr(section_letter)}"
            else:
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
    print(f"\nFile 'data/classes_data.csv' generated with {len(df_classes['class_id'].unique())} classes.")


def main():
    """Execute the full regression and prediction pipeline."""
    df_hist = load_historical_data("data/historical_enrollment_data.csv")
    models = train_models_by_grade(df_hist)
    predictions = predict_by_grade(models)
    generate_classes_from_predictions(predictions)


if __name__ == "__main__":
    main()
