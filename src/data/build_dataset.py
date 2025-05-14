import pandas as pd
from src.utils.path import RAW_DIR, PROCESSED_DIR
import ta
from ta.trend import ADXIndicator
from src.features.add_features import add_lag_and_rolling_features


index_list = ["CAC40", "EUROSTOXX50", "STOXX600"]
indicator_list = ["MA_5", "MACD", "EMA_10", "SMA_10", "ADX_10", "APO_10", "CCI_10", "MFI_10", "RSI_10", "ATR_10"]

def load_and_format(file_name):
    df = pd.read_csv(RAW_DIR / file_name, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df

def add_technical_indicators(df, index_name):
    close_col, high_col, low_col, volume_col = f"Close_{index_name}", f"High_{index_name}", f"Low_{index_name}", f"Volume_{index_name}"

    # MA (5)
    df[f"MA_5_{index_name}"] = df[close_col].rolling(window=5).mean()

    # DEMA (5)
    #df[f"DEMA_5_{index_name}"] = ta.trend.DEMAIndicator(close=df[close_col], window=5).dema()

    # MACD
    macd = ta.trend.MACD(close=df[close_col])
    df[f"MACD_{index_name}"] = macd.macd()

    # KAMA (5)
    #df[f"KAMA_5_{index_name}"] = ta.trend.KAMAIndicator(close=df[close_col], window=5).kama()

    # EMA, SMA (10)
    df[f"EMA_10_{index_name}"] = ta.trend.EMAIndicator(close=df[close_col], window=10).ema_indicator()
    df[f"SMA_10_{index_name}"] = ta.trend.SMAIndicator(close=df[close_col], window=10).sma_indicator()

    # ADX, DX (10)
    adx_indicator = ta.trend.ADXIndicator(high=df[high_col], low=df[low_col], close=df[close_col], window=10)
    df[f"ADX_10_{index_name}"] = adx_indicator.adx()
    #df[f"DX_10_{index_name}"] = adx_indicator.dx()

    # APO (10)
    df[f"APO_10_{index_name}"] = ta.momentum.AwesomeOscillatorIndicator(high=df[high_col], low=df[low_col]).awesome_oscillator()

    # CCI (10)
    df[f"CCI_10_{index_name}"] = ta.trend.CCIIndicator(high=df[high_col], low=df[low_col], close=df[close_col], window=10).cci()

    # MFI (10)
    df[f"MFI_10_{index_name}"] = ta.volume.MFIIndicator(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], window=10).money_flow_index()

    # RSI (10)
    df[f"RSI_10_{index_name}"] = ta.momentum.RSIIndicator(close=df[close_col], window=10).rsi()

    # ATR, NATR (10)
    atr = ta.volatility.AverageTrueRange(high=df[high_col], low=df[low_col], close=df[close_col], window=10)
    df[f"ATR_10_{index_name}"] = atr.average_true_range()
    #df[f"NATR_10_{index_name}"] = atr.natr()

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
    
    # Sauvegarde en Parquet
    output_path = PROCESSED_DIR / "dataset_full.parquet"
    df.to_parquet(output_path, compression="snappy")
    print(f"✅ Dataset saved to {output_path} — shape: {df.shape}")



if __name__ == "__main__":
    main()
