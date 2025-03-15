import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from joblib import dump, load
import os

def load_and_process_data(file_path):
    """Carrega os dados e aplica processamento para cálculo das features."""
    df = pd.read_csv(file_path)
    
    # Criar as colunas se não existirem
    if 'volatility' not in df.columns:
        df['volatility'] = df['close'].rolling(window=10).std()
    if 'ma_ratio' not in df.columns:
        df['ma_ratio'] = df['close'].rolling(window=5).mean() / df['close'].rolling(window=20).mean()

    # Remover valores NaN que podem surgir no cálculo das médias
    df.dropna(inplace=True)
    
    return df

def initialize_model(model_path=None):
    """Inicializa o modelo ou carrega um existente."""
    if model_path and os.path.exists(model_path):
        model = load(model_path)
        if not isinstance(model, SGDClassifier):
            print(f"Modelo carregado de {model_path} não é um SGDClassifier. Inicializando um novo modelo.")
            model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True)
        else:
            print(f"Modelo SGDClassifier carregado de {model_path}")
    else:
        model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=True)
        print("Novo modelo SGDClassifier inicializado")
    return model

def train_incremental(model, X, y, scaler=None):
    """Treina o modelo de forma incremental, atualizando apenas os novos dados."""
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    model.partial_fit(X_scaled, y, classes=[0, 1])
    return model, scaler

def evaluate_model(model, X, y, scaler):
    """Avalia o modelo utilizando acurácia como métrica."""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)
    print(f"Acurácia do modelo: {accuracy:.2f}")
    return accuracy

def save_model(model, scaler, model_path, scaler_path):
    """Salva o modelo treinado e o scaler."""
    dump(model, model_path)
    dump(scaler, scaler_path)
    print(f"Modelo salvo em {model_path}")
    print(f"Scaler salvo em {scaler_path}")

if __name__ == "__main__":
    data_filename = 'data/sol_usdc_data.csv'
    model_filename = 'models/modelo_scalping.pkl'
    scaler_filename = 'models/scaler.pkl'

    # Garantir que os diretórios existem
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_filename), exist_ok=True)

    # Carrega e processa os dados
    df = load_and_process_data(data_filename)

    # Criar a coluna alvo
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Definir as features
    features = ['volatility', 'ma_ratio']

    # Verificar se as colunas necessárias existem
    if not all(f in df.columns for f in features):
        print(f"Erro: As colunas {features} não foram encontradas no DataFrame!")
        print(f"Colunas disponíveis: {df.columns}")
        exit()

    # Separar os dados
    X = df[features]
    y = df['target']

    # Inicializar ou carregar o modelo existente
    model = initialize_model(model_filename)

    # Inicializar ou carregar o scaler existente
    scaler = None
    if os.path.exists(scaler_filename):
        scaler = load(scaler_filename)
        print(f"Scaler carregado de {scaler_filename}")

    # Treinar o modelo de forma incremental
    model, scaler = train_incremental(model, X, y, scaler)

    # Avaliar o modelo
    evaluate_model(model, X, y, scaler)

    # Salvar o modelo e o scaler atualizados
    save_model(model, scaler, model_filename, scaler_filename)
