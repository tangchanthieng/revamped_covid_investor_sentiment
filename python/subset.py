import pandas as pd
from config import MART_DATA_DIR

# Define input/output paths
data_path = MART_DATA_DIR / 'cleaned_data.csv'
subset_path = MART_DATA_DIR / 'subset.csv'

# Subset the data to the first n_rows and save it to a new CSV file
def subset_data(df: pd.DataFrame, n_rows: int = 1097) -> pd.DataFrame:
    
    print(f"Subsetting the data to the first {n_rows} rows.")
    df_subset = df.iloc[:n_rows]
    df_subset.to_csv(subset_path, index=False)
    
    return df_subset

# Summary of the subset_data function:
print(f"Subset data rows: {len(pd.read_csv(data_path)):,} rows")
print(f"Subset data columns: {len(pd.read_csv(data_path).columns):,}")

if __name__ == "__main__":
    data = pd.read_csv(data_path)
    subset_data(data)