import pandas as pd
from src.utils.path import RAW_DIR, PROCESSED_DIR
from src.features.add_features import add_lag_rolling_and_return_features, add_technical_indicators, add_temporal_features


index_list = ["CAC40", "EUROSTOXX50", "STOXX600"]
indicator_list = ["MA_5", "MACD", "EMA_10", "SMA_10", "ADX_10", "APO_10", "CCI_10", "MFI_10", "RSI_10", "ATR_10"]

def load_and_format(file_name):
    df = pd.read_csv(RAW_DIR / file_name, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df

def main():
    # Charger les indices
    cac40 = load_and_format("CAC40_monthly.csv")
    stoxx600 = load_and_format("STOXX600_monthly.csv")
    eurostoxx50 = load_and_format("EUROSTOXX50_monthly.csv")

    # Fusionner toutes les sources sur Date
    df = cac40.join([stoxx600, eurostoxx50], how="inner")

    for index in index_list:    
        df = add_technical_indicators(df, index)

    # Ajout des lags et rolling features (log_Close_lag, log_Open_lag, log_High_lag, log_Low_lag, log_Volume_lag de 1 à 5 à chaque fois)
    df = add_lag_rolling_and_return_features(df, index_list)
    
    # Ajout des features temporelles
    df = add_temporal_features(df)

    
    # Sauvegarde en Parquet
    output_path = PROCESSED_DIR / "dataset_months_full.parquet"
    df.to_parquet(output_path, compression="snappy")
    print(f"✅ Dataset saved to {output_path} — shape: {df.shape}")



if __name__ == "__main__":
    main()
