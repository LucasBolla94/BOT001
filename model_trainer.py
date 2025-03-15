import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from feature_engineering import load_and_process_data

def train_model(df):
    # Definindo a variável alvo
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    # Selecionando as features para o modelo
    features = ['volatility', 'ma_ratio']
    X = df[features]
    y = df['target']

    # Dividindo os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Inicializando e treinando o modelo de Regressão Logística
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Fazendo previsões no conjunto de teste
    y_pred = model.predict(X_test)
    # Calculando a acurácia do modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo: {accuracy:.2f}')

    return model

def save_model(model, filename):
    # Salvando o modelo treinado no arquivo especificado
    dump(model, filename)
    print(f'Modelo salvo em {filename}')

if __name__ == "__main__":
    # Carregando e processando os dados
    df = load_and_process_data('data/sol_usdc_data.csv')
    # Treinando o modelo
    model = train_model(df)
    # Salvando o modelo treinado
    save_model(model, 'models/modelo_scalping.pkl')
