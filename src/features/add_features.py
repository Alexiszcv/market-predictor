import pandas as pd
import ta
from ta.trend import ADXIndicator
import numpy as np

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


def get_lag_feature(df, indicator, period):
    col_name = f"{indicator}_lag{period}"
    return pd.Series(df[indicator].shift(period), name=col_name)

def get_rolling_features(df, indicator, window=5):
    return pd.concat([
        pd.Series(df[indicator].rolling(window=window).mean(), name=f"{indicator}_rollmean{window}"),
        pd.Series(df[indicator].rolling(window=window).std(), name=f"{indicator}_rollstd{window}")
    ], axis=1)

def add_lag_and_rolling_features(df, index_list, indicator_list):

    new_features = []
    indicators = []

    for indicator in indicator_list:
        for index in index_list:
            indicators.append(f"{indicator}_{index}")

    for ind in indicators:
        new_features.append(get_lag_feature(df, ind, 1))
        new_features.append(get_rolling_features(df, ind, 5))

    # Concat toutes les nouvelles colonnes en une fois
    df = pd.concat([df] + new_features, axis=1)

    # Ajout des lags directement pour les prix des indices

    for index in ["CAC40", "EUROSTOXX50", "STOXX600"]: 
        df[f"Close_{index}_lag1"] = df[f"Close_{index}"].shift(1)
        df[f"Close_{index}_lag2"] = df[f"Close_{index}"].shift(2)


    # Défragmenter proprement (optionnel mais recommandé)
    df = df.copy()

    return df 

def add_log_returns(df, indices):
    """
    Ajoute les log-returns (rendements logarithmiques) pour chaque indice.

    Paramètres :
    - df : DataFrame avec les colonnes 'Close_{indice}'
    - indices : liste des noms d'indices, ex : ['CAC40', 'STOXX600', 'EUROSTOXX50']

    Retour :
    - DataFrame avec colonnes 'Return_{indice}' ajoutées
    """
    for index in indices:
        close_col = f"Close_{index}"
        return_col = f"Return_{index}"
        df[return_col] = np.log(df[close_col] / df[close_col].shift(1))
        df[f"Return_{index}_t+1"] = df[return_col].shift(-1)
    return df


def add_log_prices(df, indices):
    for index in indices:
        close_col = f"Close_{index}"
        log_close_col = f"Log_close_{index}"
        df[log_close_col] = np.log(df[close_col]) 
    return df    
