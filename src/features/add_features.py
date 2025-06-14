import pandas as pd
import ta
from ta.trend import ADXIndicator
import numpy as np

def add_technical_indicators(df, index_name): # IMPORTANT ! On shift(1) tous les indicateurs, permet d'avoir la valeur calculée à partir des données à t-1 pour l'instant t (sinon cela induirait du data leakage)
    close_col, high_col, low_col, volume_col = f"Close_{index_name}", f"High_{index_name}", f"Low_{index_name}", f"Volume_{index_name}"

    # MA (5)
    df[f"MA_5_{index_name}"] = df[close_col].rolling(window=5).mean().shift(1)

    # MACD
    macd = ta.trend.MACD(close=df[close_col])
    df[f"MACD_{index_name}"] = macd.macd().shift(1)

    # EMA, SMA (10)
    df[f"EMA_10_{index_name}"] = ta.trend.EMAIndicator(close=df[close_col], window=10).ema_indicator().shift(1)
    df[f"SMA_10_{index_name}"] = ta.trend.SMAIndicator(close=df[close_col], window=10).sma_indicator().shift(1)

    # ADX (10)
    adx_indicator = ta.trend.ADXIndicator(high=df[high_col], low=df[low_col], close=df[close_col], window=10)
    df[f"ADX_10_{index_name}"] = adx_indicator.adx().shift(1)

    # APO (10)
    df[f"APO_10_{index_name}"] = ta.momentum.AwesomeOscillatorIndicator(high=df[high_col], low=df[low_col]).awesome_oscillator().shift(1)

    # CCI (10)
    df[f"CCI_10_{index_name}"] = ta.trend.CCIIndicator(high=df[high_col], low=df[low_col], close=df[close_col], window=10).cci().shift(1)

    # MFI (10)
    df[f"MFI_10_{index_name}"] = ta.volume.MFIIndicator(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], window=10).money_flow_index().shift(1)

    # RSI (10)
    df[f"RSI_10_{index_name}"] = ta.momentum.RSIIndicator(close=df[close_col], window=10).rsi().shift(1)

    # ATR (10)
    atr = ta.volatility.AverageTrueRange(high=df[high_col], low=df[low_col], close=df[close_col], window=10)
    df[f"ATR_10_{index_name}"] = atr.average_true_range().shift(1)

    return df



def get_lag_feature(df, indicator, period):
    col_name = f"{indicator}_lag{period}"
    return pd.Series(df[indicator].shift(period), name=col_name)

def get_rolling_features(df, indicator, window=5):
    return pd.concat([
        pd.Series(df[indicator].rolling(window=window).mean(), name=f"{indicator}_rollmean{window}"),
        pd.Series(df[indicator].rolling(window=window).std(), name=f"{indicator}_rollstd{window}")
    ], axis=1)

def add_lag_rolling_and_return_features(df, index_list):

    # Rolling mean / std directement sur log-prix 

    for index in index_list: 
        df[f"log_Close_{index}"] = np.log(df[f"Close_{index}"])

        df[f"log_Close_{index}_rolling_mean5"] = df[f"log_Close_{index}"].shift(1).rolling(window=5).mean() # Pareil, on shift pour éviter le data leakage 
        df[f"log_Close_{index}_rolling_std5"] = df[f"log_Close_{index}"].shift(1).rolling(window=5).std()

    # Ajout des lags 

    for index in ["CAC40", "EUROSTOXX50", "STOXX600"]: 
        for i in range(1, 6):
            df[f"log_Close_{index}_lag{i}"] = np.log(df[f"Close_{index}"]).shift(i)
            df[f"log_Open_{index}_lag{i}"] = np.log(df[f"Open_{index}"]).shift(i)
            df[f"log_High_{index}_lag{i}"] = np.log(df[f"High_{index}"]).shift(i)
            df[f"log_Low_{index}_lag{i}"] = np.log(df[f"Low_{index}"]).shift(i)
            df[f"log_Volume_{index}_lag{i}"] = np.log(df[f"Volume_{index}"]).shift(i)
            df[f"log_Return_{index}_lag{i}"] = np.log(df[f"Close_{index}"]).shift(i) - np.log(df[f"Close_{index}"]).shift(i+1)

    # Défragmenter proprement (optionnel mais recommandé)
    df = df.copy()

    return df 


def add_temporal_features(df): # Pour le modèle de prédiction journalier 
    df['day_of_week'] = df.index.dayofweek        # 0=Lundi, ..., 4=Vendredi
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_month_end'] = df.index.is_month_end.astype(int)
    df['is_month_start'] = df.index.is_month_start.astype(int)
    df['is_start_of_week'] = (df['day_of_week'] <= 1).astype(int) # Début de semaine : Lundi (0), Mardi (1)
    df['is_end_of_week'] = (df['day_of_week'] >= 3).astype(int) # Fin de semaine : Jeudi (3), Vendredi (4)

    return df

def add_temporal_features_month(df): # Pour le modèle de prédiction mensuel 
    df['month'] = df.index.month
    df['is_january'] = (df['month'] == 1).astype(int)
    df['is_earnings_season'] = df['month'].isin([4, 7, 10]).astype(int)
    df['is_september_october'] = df['month'].isin([9, 10]).astype(int)
    df['is_summer'] = df['month'].isin([7, 8]).astype(int)
    df['is_end_of_year'] = df['month'] == 12
    df['is_end_of_year'] = (df['is_end_of_year'] & (df.index.day >= 15)).astype(int)
    return df




