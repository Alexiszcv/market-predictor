import yfinance as yf
import pandas as pd

# Utilisation de l'API de yahoofinance pour importer les données relatives aux actifs financiers (ici les indices CAC40, STOXX600 et EUROSTOXX50)
def download_index(symbol, name, start="2018-01-01", end=None): 
    df = yf.download(symbol, start=start, end=end)
    df.columns = df.columns.droplevel(1)
    df = df.rename(columns={col: f"{col}_{name}" for col in df.columns if col != "Date"})
    df.to_csv(f"data/raw/{name}.csv", index=True)
    print(f"{name} saved with {len(df)} rows.")

if __name__ == "__main__":
    indices = {
        "CAC40": "^FCHI",
        "STOXX600": "^STOXX",
        "EUROSTOXX50": "^STOXX50E"
    }

    for name, symbol in indices.items():
        download_index(symbol, name)
