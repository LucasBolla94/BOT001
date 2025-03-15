import pandas as pd

def add_features(df):
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(window=10).std()
    df['ma_short'] = df['close'].rolling(window=5).mean()
    df['ma_long'] = df['close'].rolling(window=20).mean()
    df['ma_ratio'] = df['ma_short'] / df['ma_long']
    df.dropna(inplace=True)
    return df

def load_and_process_data(filename):
    df = pd.read_csv(filename, index_col='timestamp', parse_dates=True)
    df = add_features(df)
    return df

if __name__ == "__main__":
    filename = 'data/sol_usdc_data.csv'
    df = load_and_process_data(filename)
    print(df.head())
