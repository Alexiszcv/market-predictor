import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from src.utils.path import FULL_DATASET, PRED_DIR

class WalkForwardPredictor:
    def __init__(self, base_model, target_names, window_size=500, duration=60):
        self.base_model = base_model
        self.window_size = window_size
        self.duration = duration
        self.targets = target_names
        self.model = MultiOutputRegressor(self.base_model)
        self.model_name = type(self.base_model).__name__
        self.y_preds = []
        self.y_trues = []

    def fit_predict(self, X, y):
        """Main function to run the walk-forward prediction."""
        for start in range(0, self.duration):
            print(f"Fitting model for day {start + 1} out of {self.duration}")
            end = start + self.window_size
            X_train, y_train = X.iloc[start:end], y.iloc[start:end]
            X_test, y_test = X.iloc[end:end+1], y.iloc[end:end+1]

            X_train_clean = X_train.replace([np.inf, -np.inf], np.nan).dropna()
            y_train_clean = y_train.loc[X_train_clean.index]

            X_train_clean = X_train_clean.clip(lower=-1e6, upper=1e6)
            X_test_clean = X_test.clip(lower=-1e6, upper=1e6)

            self.model.fit(X_train_clean, y_train_clean)
            y_pred = self.model.predict(X_test_clean)

            self.y_preds.append(y_pred.flatten())
            self.y_trues.append(y_test.values.flatten())

    def evaluate(self):
        """Calculate and return the Mean Squared Error per target."""
        y_preds_array = np.array(self.y_preds)
        y_trues_array = np.array(self.y_trues)

        rmse_per_target = {}
        for idx, target in enumerate(self.targets):
            rmse = np.sqrt(mean_squared_error(np.exp(y_trues_array[:, idx]), np.exp(y_preds_array[:, idx])))
            rmse_per_target[target] = rmse
            print(f"Mean Squared Error for {target} ({self.model_name}): {rmse:.5f}")

        return rmse_per_target


    def save_predictions(self, index, targets, save_dir):
        """Save predictions and true values to CSV."""
        y_preds_df = pd.DataFrame(np.array(self.y_preds), columns=targets)
        y_trues_df = pd.DataFrame(np.array(self.y_trues), columns=targets)

        # Align index
        y_preds_df.index = index[self.window_size:self.window_size + self.duration]
        y_trues_df.index = index[self.window_size:self.window_size + self.duration]

        # Save to CSV
        y_preds_df.to_csv(save_dir / f"y_pred_{self.model_name}.csv", index=True)
        y_trues_df.to_csv(save_dir / f"y_true_{self.model_name}.csv", index=True)


X = pd.read_parquet(FULL_DATASET)
targets = ['log_Close_CAC40', 'log_Close_STOXX600', 'log_Close_EUROSTOXX50']
cols = ['Open', 'Close', 'High', 'Low', 'Volume', 'log_Close']
indices = ["CAC40", "STOXX600", "EUROSTOXX50"]

col_to_exclude = [f'{col}_{ind}' for col in cols for ind in indices]
features = [col for col in X.columns if col not in col_to_exclude]

y = X[targets]
X = X[features]

if __name__ == "__main__":
    model = XGBRegressor(n_estimators=100, max_depth=5, verbosity=0)
    # model = LGBMRegressor(n_estimators=100, max_depth=5, verbose=-1)
    # model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    # model = GradientBoostingRegressor(n_estimators=100, max_depth=5)
    # model = ExtraTreesRegressor(n_estimators=100, max_depth=5)
    # model = Ridge(alpha=1.0)
    # model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    predictor = WalkForwardPredictor(model, target_names=targets)
    predictor.fit_predict(X, y)
    predictor.evaluate()
    predictor.save_predictions(X.index, targets, PRED_DIR)
