import pandas as pd

DATASET_PATH = "../data/raw/url_features_extracted1.csv"


def inspect_dataset():
    df = pd.read_csv(DATASET_PATH)

    print("=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print("\nNumber of rows:", len(df))
    print("Number of columns:", len(df.columns))

    print("\nColumn names:")
    for column in df.columns:
        print("-", column)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDataset shape:")
    print(df.shape)

    print("\nClass label distribution:")
    print(df["ClassLabel"].value_counts(dropna=False))

    print("\nClass label percentages:")
    print(df["ClassLabel"].value_counts(normalize=True, dropna=False) * 100)


if __name__ == "__main__":
    inspect_dataset()