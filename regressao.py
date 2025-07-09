
# regressao.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def gerar_dados_historicos(n=100):
    np.random.seed(42)
    dados = {
        "num_turmas": np.random.randint(5, 15, n),
        "num_salas": np.random.randint(3, 10, n),
        "num_professores": np.random.randint(5, 20, n),
        "taxa_ocupacao": np.random.uniform(60, 95, n),
    }
    return pd.DataFrame(dados)

def treinar_modelo_regressao(df):
    X = df[["num_turmas", "num_salas", "num_professores"]]
    y = df["taxa_ocupacao"]
    modelo = LinearRegression().fit(X, y)
    return modelo
