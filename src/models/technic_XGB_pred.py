from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import pandas as pd
from src.utils.path import FULL_DATASET, PRED_DIR

X = pd.read_parquet(FULL_DATASET)
X = X.dropna()

last_date = X.index[X.index < "2024-01-01"].max()
split_date = "2024-01-01"


y = X[["Return_CAC40_t+1", "Return_STOXX600_t+1", "Return_EUROSTOXX50_t+1"]]
Closing_prices = X.loc[X.index >= split_date, ["Close_CAC40", "Close_STOXX600", "Close_EUROSTOXX50"]]

excluded_cols = list(y.columns) + [col for col in X.columns if col.startswith("Close_")]
X = X.drop(columns=excluded_cols)


X_train = X[X.index < split_date]
X_test = X[X.index >= split_date]

y_train = y[y.index < split_date]
y_test = y[y.index >= split_date]


model = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=5))
model.fit(X_train, y_train)

y_pred_df = model.predict(X_test)
y_pred_df = pd.DataFrame(
    y_pred_df,
    index=y_test.index,
    columns=y_test.columns
)


y_pred_df.to_parquet(PRED_DIR/"y_pred_df.parquet")
y_test.to_parquet(PRED_DIR/"y_test_df.parquet")
Closing_prices.to_parquet(PRED_DIR/"closing_prices.parquet")











