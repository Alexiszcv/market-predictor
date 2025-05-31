import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from src.utils.path import FULL_DATASET, PRED_DIR
import matplotlib.pyplot as plt

cols = ['Open', 'Close', 'High', 'Low', 'Volume', 'log_Close']
indices = ["CAC40", "STOXX600", "EUROSTOXX50"]

col_to_exclude = []
for col in cols:
    for ind in indices: 
        col_to_exclude.append(f'{col}_{ind}') 

# Chargement des données
X = pd.read_parquet(FULL_DATASET)


# Paramètres
window_size = 500  # Taille de la fenêtre glissante
targets = ['log_Close_CAC40', 'log_Close_STOXX600', 'log_Close_EUROSTOXX50']  # Variables cibles
features = [col for col in X.columns if col not in col_to_exclude]  

y = X[targets]
X = X[features]



model = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=5, verbosity=0))

# Stocker les prédictions et vraies valeurs
y_preds = []
y_trues = []

# Walk-forward avec sliding window
duration = 60 # max : duration = len(X)-window_size
for start in range(0, duration):


    end = start + window_size
    X_train, y_train = X.iloc[start:end], y.iloc[start:end]
    X_test, y_test = X.iloc[end:end+1], y.iloc[end:end+1]

    X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).dropna()
    y_train_clean = y_train.loc[X_train_clean.index]


    # Entraînement
    model.fit(X_train_clean, y_train_clean)


    # Prédiction
    y_pred = model.predict(X_test)

    # Stockage
    y_preds.append(y_pred.flatten())  # .flatten() car model.predict retourne 2D (1, 3)
    y_trues.append(y_test.values.flatten())

# Conversion en DataFrame
y_preds = np.array(y_preds)
y_trues = np.array(y_trues)

# Exemple d'évaluation
mse = mean_squared_error(y_trues, y_preds)
print(f"Mean Squared Error: {mse}")




y_preds_df = pd.DataFrame(y_preds, columns=targets)
y_trues_df = pd.DataFrame(y_trues, columns=targets)

y_preds_df.index = X.index[window_size:window_size+duration] # les window_size premiers indices ne font partie que du train et pas du test 
y_trues_df.index = X.index[window_size:window_size+duration]

y_preds_df.to_csv(PRED_DIR / "y_pred_df.csv", index=True)
y_trues_df.to_csv(PRED_DIR / "y_trues_df.csv", index=True)
