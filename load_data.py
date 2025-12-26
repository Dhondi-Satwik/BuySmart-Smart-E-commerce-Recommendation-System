import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "data.csv")

def load_data(path):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df = df.dropna(subset=["CustomerID"])
    df = df[df["Quantity"] > 0]
    df["CustomerID"] = df["CustomerID"].astype(int)
    return df

if __name__ == "__main__":
    print("Looking for data at:", DATA_PATH)
    df = load_data(DATA_PATH)
    print("Data loaded successfully")
    print(df.head())
    print("Data shape:", df.shape)
