from src.utils.path import RAW_DIR
import pandas as pd

macro_df = pd.read_csv(RAW_DIR/"Macro_indicators.csv", sep=";", decimal=",")

cols_to_drop = [
    "countryname", "exports", "imports", "investments", "fixedinvestments", 
    "consumptions", "govexpenditures", "govrevenue", "govtaxation", "govdeficit",
    "currentaccount", "realconsumptions"
]

countries = macro_df["ISO3"]

macro_df = macro_df.drop(columns=cols_to_drop)

macro_df["date"] = pd.to_datetime(macro_df["year"], format="%Y")

indicator_columns = [col for col in macro_df.columns if col not in ["ISO3", "date", "year"]]

macro_df = pd.pivot_table(macro_df, index="date", columns="ISO3", values=indicator_columns)
macro_df.columns = [f"{col[0]}_{col[1]}" for col in macro_df.columns]
macro_df = macro_df.reset_index()

# Répliquer les valeurs annuelles pour chaque mois
monthly_df = pd.DataFrame()
for year in macro_df['date'].dt.year.unique():
    year_data = macro_df[macro_df['date'].dt.year == year]
    for month in range(1, 13):
        month_data = year_data.copy()
        month_data['date'] = pd.to_datetime(f'{year}-{month:02d}-01')
        monthly_df = pd.concat([monthly_df, month_data], ignore_index=True)

# Trier par date
monthly_df = monthly_df.sort_values('date').reset_index(drop=True)

# Sauvegarder le fichier avec données mensuelles
monthly_df.to_csv(f"data/raw/Indicateurs_Macro_temp.csv", index=True)


print(monthly_df.info())