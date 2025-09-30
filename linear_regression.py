import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_historical_data(file_path="data/historical_enrollment_data.csv"):
    if not os.path.exists(file_path):
        print(f"file_path {file_path} não encontrado.")
        return None
    
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    
    # Create derived features based on simulator data
    df["normalized_year"] = df["year"] - df["year"].min()
    df["per_capita_investiment"] = df["education_investment"] / df["total_enrollments"]
    df["economical_index"] = df["regional_gdp"] / (df["unemployment_rate"] + 1)
    
    return df

def train_linear_regression(df, salvar_modelo=True):
    if df is None:
        return None
    
    # Features used for prediction (all columns from simulator)
    features = [
        "normalized_year", 
        "semester", 
        "regional_gdp", 
        "unemployment_rate", 
        "education_investment", 
        "per_capita_investiment", 
        "economical_index"
    ]
    
    X = df[features]
    y = df["total_enrollments"]
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    modelo = LinearRegression().fit(X_scaled, y)
    
    # Save model and scaler
    if salvar_modelo:
        joblib.dump(modelo, "modelo_regressao.pkl")
        joblib.dump(scaler, "scaler.pkl")
    
    return {
        "modelo": modelo, 
        "scaler": scaler, 
        "features": features
    }

def predict_future_demand(modelo_info, future_year=2025, semester=1, 
                         pid_regional=60000, unemployment_rate=7.0, 
                         education_investiment=1200000):
    if modelo_info is None:
        return None
    
    modelo = modelo_info["modelo"]
    scaler = modelo_info["scaler"]
    
    # Calculate normalized year (base: 2019, as in simulator)
    normalized_year = future_year - 2019
    
    # Calculate derived features
    # Current students estimate for per capita calculation
    estimated_students = 200  # Base value, can be adjusted
    per_capita_investiment = education_investiment / estimated_students
    economical_index = pid_regional / (unemployment_rate + 1)
    
    # Create array with all features in correct order
    X_novo = np.array([[
        normalized_year,
        semester,
        pid_regional,
        unemployment_rate,
        education_investiment,
        per_capita_investiment,
        economical_index
    ]])
    
    # Normalize and predict
    X_scaled = scaler.transform(X_novo)
    previsao = modelo.predict(X_scaled)[0]
    
    return int(previsao)