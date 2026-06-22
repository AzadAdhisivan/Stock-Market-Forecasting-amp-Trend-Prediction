import pandas as pd
import os

for file in os.listdir("../data"):
    if file.endswith(".csv"):
        filepath = f"../data/{file}"
        df = pd.read_csv(filepath, index_col="Date", parse_dates=True)

        print(f"\n{'='*50}")
        print(f"  {file} Dataset Summary")
        print(f"{'='*50}")

        # Number of observations
        print(f"\n Number of Observations: {len(df)} rows x {len(df.columns)} columns")

        # Data types
        print(f"\n Data Types:")
        print(df.dtypes)

        print(f"\n Missing Values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("  No missing values found!")
        else:
            print(missing[missing > 0])

        print(f"\n Basic Statistics:")
        print(df.describe().round(2))