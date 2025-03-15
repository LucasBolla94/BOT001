import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import dump, load
from tqdm import tqdm
import time
import os

def load_and_process_data(file_path):
    """Carrega os dados e calcula os indicadores técnicos."""
    df = pd.read_csv(file_path)

    # Calculando a volatilidade
    df['volatility'] = df['close'].rolling(window=10).std()

    # Calculando a razão das médias móveis
    df['ma_ratio'] = df['close'].rolling(window=5).mean() / df['close'].rolling(window=20).mean()

    # Calculando o RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Calculando o MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26

    # Calculando as Bandas de Bollinger
    sma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    df['bollinger_upper'] = sma20 + (std20 * 2)
    df['bollinger_lower'] = sma20 - (std20 * 2)

    # Removendo valores NaN
    df.dropna(inplace=True)

    return df

def initialize_models(model_paths):
    """Inicializa ou carrega os modelos existentes."""
    models = []
    for model_path in model_paths:
        if os.path.exists(model_path):
            model = load(model_path)
            print(f"Modelo carregado de {model_path}")
        else:
            model = None
        models.append(model)
    return models

def train_incremental(models, X, y, scaler=None):
    """Treina os modelos de forma incremental com barra de progresso."""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    for i, model in enumerate(tqdm(models, desc="Treinando modelos", unit="modelo")):
        start_time = time.time()  # Início do cronômetro
        
        if model is None:
            if i == 0:
                models[i] = SGDClassifier(loss='log_loss', max_iter=1000, warm_start=True)
            elif i == 1:
                models[i] = DecisionTreeClassifier()
            elif i == 2:
                models[i] = RandomForestClassifier(n_estimators=5, n_jobs=-1, warm_start=True)  # Reduzindo tempo de treino
        
        models[i].fit(X_scaled, y)
        
        elapsed_time = time.time() - start_time  # Tempo de execução
        print(f"Tempo de treinamento do {type(models[i]).__name__}: {elapsed_time:.2f} segundos")

    return models, scaler

def evaluate_models(models, X, y, scaler):
    """Avalia os modelos individualmente e como ensemble."""
    X_scaled = scaler.transform(X)
    accuracies = []

    for model in tqdm(models, desc="Avaliando modelos", unit="modelo"):
        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        accuracies.append(accuracy)
        print(f"Acurácia do modelo {type(model).__name__}: {accuracy:.2f}")

    ensemble = VotingClassifier(estimators=[
        ('sgd', models[0]),
        ('dt', models[1]),
        ('rf', models[2])
    ], voting='hard')
    ensemble.fit(X_scaled, y)
    y_pred_ensemble = ensemble.predict(X_scaled)
    ensemble_accuracy = accuracy_score(y, y_pred_ensemble)
    print(f"Acurácia do ensemble: {ensemble_accuracy:.2f}")

    return accuracies, ensemble, ensemble_accuracy

def save_models(models, scaler, model_paths, scaler_path):
    """Salva os modelos e o scaler."""
    for model, model_path in zip(models, model_paths):
        dump(model, model_path)
        print(f"Modelo {type(model).__name__} salvo em {model_path}")
    dump(scaler, scaler_path)
    print(f"Scaler salvo em {scaler_path}")

if __name__ == "__main__":
    data_filename = 'data/sol_usdc_data.csv'
    model_filenames = [
        'models/sgd_model.pkl',
        'models/decision_tree_model.pkl',
        'models/random_forest_model.pkl'
    ]
    scaler_filename = 'models/scaler.pkl'

    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    df = load_and_process_data(data_filename)

    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    feature_names = ['volatility', 'ma_ratio', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']
    X = df[feature_names]
    y = df['target']

    models = initialize_models(model_filenames)

    scaler = None
    if os.path.exists(scaler_filename):
        scaler = load(scaler_filename)
        print(f"Scaler carregado de {scaler_filename}")

    if scaler is not None:
        X = X[scaler.feature_names_in_]

    models, scaler = train_incremental(models, X, y, scaler)

    accuracies, ensemble, ensemble_accuracy = evaluate_models(models, X, y, scaler)

    save_models(models, scaler, model_filenames, scaler_filename)
