import numpy as np
import pandas as pd
from config import MART_DATA_DIR


def _moving_average_impute(series: pd.Series, window: int = 7) -> pd.Series:
    """Fill missing values with nearby rolling averages, preserving edge values."""
    numeric_series = pd.to_numeric(series, errors="coerce").astype(float)
    trailing_average = numeric_series.rolling(window=window, min_periods=1).mean()
    leading_average = (
        numeric_series.iloc[::-1]
        .rolling(window=window, min_periods=1)
        .mean()
        .iloc[::-1]
    )
    imputed = numeric_series.fillna(trailing_average).fillna(leading_average)

    observed = numeric_series.dropna()
    if observed.empty:
        return numeric_series

    valid_positions = np.flatnonzero(numeric_series.notna().to_numpy())
    first_valid = int(valid_positions[0])
    last_valid = int(valid_positions[-1])

    if first_valid > 0:
        leading_mean = observed.iloc[:window].mean()
        imputed.iloc[:first_valid] = imputed.iloc[:first_valid].fillna(leading_mean)

    if last_valid < len(numeric_series) - 1:
        trailing_mean = observed.iloc[-window:].mean()
        imputed.iloc[last_valid + 1 :] = imputed.iloc[last_valid + 1 :].fillna(
            trailing_mean
        )

    moving_estimates = trailing_average.fillna(leading_average).interpolate(
        limit_area="inside",
        limit_direction="both",
    )
    imputed = imputed.fillna(moving_estimates)

    return pd.Series(imputed.to_numpy(), index=series.index, name=series.name)


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
sentiment_df["date"] = pd.to_datetime(sentiment_df["date"], errors="coerce").dt.date
market_df["date"] = pd.to_datetime(market_df["date"], errors="coerce").dt.date

# Join the two DataFrames on the 'date' column using a left join
print("\nMerging market data and sentiment data...")
merge_df = pd.merge(market_df, sentiment_df, on="date", how="left")

# Impute missing values in the sentiment series using nearby rolling averages.
if "SIS" in merge_df.columns:
    merge_df["SIS"] = _moving_average_impute(merge_df["SIS"], window=7)
    if merge_df["SIS"].isna().any():
        raise ValueError("The SIS column still contains missing values after imputation.")

# Save the merged DataFrame after imputation
merge_df.to_csv(output_path, index=False)

# Summary of the merged DataFrame
print("\nMerge completed successfully.")
print(f"Matched rows:     {len(merge_df):,}")
print(f"Output columns:   {len(merge_df.columns):,}")
print(f"Rows with missing values (after imputation): {merge_df.isnull().sum().sum():,}")
