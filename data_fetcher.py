import ccxt
import pandas as pd
import time

def fetch_ohlcv(symbol, timeframe, since, limit):
    exchange = ccxt.binance()
    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        since = ohlcv[-1][0] + 1
        all_ohlcv.extend(ohlcv)
        time.sleep(exchange.rateLimit / 1000)
    return all_ohlcv

def save_data_to_csv(symbol, filename):
    timeframe = '1m'  # Intervalo de 1 minuto
    since = ccxt.binance().parse8601('2023-01-01T00:00:00Z')  # Data de início
    limit = 1000  # Número de registros por requisição

    ohlcv = fetch_ohlcv(symbol, timeframe, since, limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.to_csv(filename)
    print(f'Dados salvos em {filename}')

if __name__ == "__main__":
    symbol = 'SOL/USDC'
    filename = 'data/sol_usdc_data.csv'
    save_data_to_csv(symbol, filename)
