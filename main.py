from data_fetcher import save_data_to_csv
from feature_engineering import load_and_process_data
from model_trainer import train_model, save_model

def main():
    # Coletar dados
    symbol = 'SOL/USDC'
    data_filename = 'data/sol_usdc_data.csv'
    save_data_to_csv(symbol, data_filename)

    # Processar dados
    df = load_and_process_data(data_filename)

    # Treinar modelo
    model = train_model(df)

    # Salvar modelo
    model_filename = 'models/modelo_scalping.pkl'
    save_model(model, model_filename)

if __name__ == "__main__":
    main()
