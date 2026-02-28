from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/Data_Set_S1.txt")

def find_header_line(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            s = line.strip().lower()
            if s.startswith("word") and (
                "happiness_rank" in s or "happiness_average" in s
            ):
                return i
    raise ValueError("Header not found.")

def load_dataset(path: Path) -> pd.DataFrame:
    header_i = find_header_line(path)

    df = pd.read_csv(
        path,
        sep="\t",
        skiprows=header_i,
        header=0,
        na_values=["--"],
    )

    for col in df.columns:
        if col != "word":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)

    print("Header line:", find_header_line(DATA_PATH))
    print("Shape:", df.shape)
    print("\nHead:\n", df.head())
    print("\nDtypes:\n", df.dtypes)
    print("\nMissing values:\n", df.isna().sum())