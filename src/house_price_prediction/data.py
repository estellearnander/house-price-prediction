from pathlib import Path
import pandas as pd


def load_house_data(data_path: str | Path) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower()
    return df
