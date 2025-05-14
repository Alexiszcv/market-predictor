import numpy as np
import pandas as pd

def reconstruct_prices(initial_price, returns_series, name="Reconstructed_Price"):
    """
    Reconstruit une série de prix à partir d'un prix initial et d'une série de rendements logarithmiques.

    Paramètres :
    - initial_price : float, le dernier prix connu (à t)
    - returns_series : pd.Series, rendements log à partir de t+1
    - name : nom de la série de prix retournée

    Retour :
    - pd.Series : série temporelle des prix reconstruits alignée avec returns_series.index
    """
    prices = [initial_price]
    for r in returns_series:
        prices.append(prices[-1] * np.exp(r))
    
    return pd.Series(prices[1:], index=returns_series.index, name=name)

last_prices_40_600_50 = [7543.18017578125, 479.0199890136719, 4521.64990234375]