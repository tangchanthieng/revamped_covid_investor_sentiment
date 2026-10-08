import pandas as pd
from config import MART_DATA_DIR
import os

# Define the path to the CSV file
sentiement_path = MART_DATA_DIR / "sentiment_index_second_run.csv"
data_path = MART_DATA_DIR / "standardized_data.csv"

# Define the output path for the merged CSV file
output_path = MART_DATA_DIR / "data.csv"

# Load the CSV file into a DataFrame
sentiment_df = pd.read_csv(sentiement_path)
market_df = pd.read_csv(data_path)

print(f"Sentiment data loaded: {len(sentiment_df):,} rows")
print(f"Market data loaded: {len(market_df):,} rows")

# Convert date columns to datetime format
sentiment_df["date"] = pd.to_datetime(
    sentiment_df["date"],
    errors="coerce"
).dt.date

market_df["date"] = pd.to_datetime(
    market_df["date"],
    errors="coerce"
).dt.date

# Join the two DataFrames on the 'date' column using a left join
print("\nMerging market data and sentiment data...")

merge_df = pd.merge(
    market_df,
    sentiment_df,
    on="date",
    how="left"
)

# Save the merged DataFrame to a new CSV file
merge_df.to_csv(output_path, index=False)

# Summary of the merged DataFrame
print("\nMerge completed successfully.")
print(f"Matched rows:     {len(merge_df):,}")
print(f"Output columns:   {len(merge_df.columns):,}")