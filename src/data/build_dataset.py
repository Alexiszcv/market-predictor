import pandas as pd
from src.utils.path import RAW_DIR, PROCESSED_DIR
from src.features.add_features import add_lag_and_rolling_features, add_technical_indicators, add_log_returns, add_log_prices


index_list = ["CAC40", "EUROSTOXX50", "STOXX600"]
indicator_list = ["MA_5", "MACD", "EMA_10", "SMA_10", "ADX_10", "APO_10", "CCI_10", "MFI_10", "RSI_10", "ATR_10"]

def load_and_format(file_name):
    df = pd.read_csv(RAW_DIR / file_name, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df

def main():
    # Charger les indices
    cac40 = load_and_format("CAC40.csv")
    stoxx600 = load_and_format("STOXX600.csv")
    eurostoxx50 = load_and_format("EUROSTOXX50.csv")

    # Fusionner toutes les sources sur Date
    df = cac40.join([stoxx600, eurostoxx50], how="inner")

    for index in index_list:    
        df = add_technical_indicators(df, index)

    # Ajout des lags et rolling features
    df = add_lag_and_rolling_features(df, index_list, indicator_list)

    # Ajout des rentabilités logarithmiques 
    df = add_log_returns(df, index_list)

    #Ajout des log-prix de fermeture
    df = add_log_prices(df, index_list)
    
    # Sauvegarde en Parquet
    output_path = PROCESSED_DIR / "dataset_full.parquet"
    df.to_parquet(output_path, compression="snappy")
    print(f"✅ Dataset saved to {output_path} — shape: {df.shape}")



if __name__ == "__main__":
    main()
