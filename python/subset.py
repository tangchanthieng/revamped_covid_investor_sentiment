import pandas as pd
from config import MART_DATA_DIR

# Define input/output paths
data_path = MART_DATA_DIR / 'cleaned_data.csv'
subset_path = MART_DATA_DIR / 'subset.csv'

# Subset the data to the first n_rows and save it to a new CSV file
def subset_data(df: pd.DataFrame, n_rows: int = 1097) -> pd.DataFrame:
    """Save and return the first n_rows of the dataframe."""
    df_subset = df.iloc[:n_rows]
    df_subset.to_csv(subset_path, index=False)
    return df_subset

if __name__ == "__main__":
    data = pd.read_csv(data_path)
    subset_data(data)