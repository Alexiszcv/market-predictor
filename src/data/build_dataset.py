import pandas as pd
from src.utils.path import RAW_DIR, PROCESSED_DIR

def load_and_format(file_name):
    df = pd.read_csv(RAW_DIR / file_name, parse_dates=["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    return df

def main():
    # Charger les indices
    cac40 = load_and_format("CAC40.csv")
    stoxx600 = load_and_format("STOXX600.csv")
    eurostoxx50 = load_and_format("EUROSTOXX50.csv")



    # Fusionner toutes les sources sur Date
    df = cac40.join([stoxx600, eurostoxx50], how="inner")

    # Sauvegarde en Parquet
    output_path = PROCESSED_DIR / "dataset_full.parquet"
    df.to_parquet(output_path, compression="snappy")
    print(f"✅ Dataset saved to {output_path} — shape: {df.shape}")

if __name__ == "__main__":
    main()
