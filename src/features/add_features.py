import pandas as pd

def add_lag_feature(df, indicator, period):
    """
    Ajoute une colonne de type lag au DataFrame.

    Paramètres :
    - df : DataFrame pandas
    - indicator : nom de la colonne de l'indicateur (ex: "EMA_5_CAC40")
    - period : nombre de jours de décalage (lag)

    Retour :
    - DataFrame avec une nouvelle colonne : indicator_lag{period}
    """
    col_name = f"{indicator}_lag{period}"
    df[col_name] = df[indicator].shift(period)
    return df

def add_rolling_feature(df, indicator, window=5):
    """
    Ajoute les moyennes et écarts-types glissants d'un indicateur.

    Paramètres :
    - df : DataFrame pandas
    - indicator : nom de la colonne de l'indicateur (ex: "EMA_5_CAC40")
    - window : taille de la fenêtre glissante (par défaut 5)

    Retour :
    - DataFrame avec deux nouvelles colonnes : indicator_rollmean{window}, indicator_rollstd{window}
    """
    mean_col = f"{indicator}_rollmean{window}"
    std_col = f"{indicator}_rollstd{window}"
    
    df[mean_col] = df[indicator].rolling(window=window).mean()
    df[std_col] = df[indicator].rolling(window=window).std()
    return df


# Seconde version mieux optimisée 

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

    # Défragmenter proprement (optionnel mais recommandé)
    df = df.copy()

    return df 


