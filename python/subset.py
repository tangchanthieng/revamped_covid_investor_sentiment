import pandas as pd
from config import MART_DATA_DIR

# Define input/output paths
input_path = MART_DATA_DIR / 'cleaned_data.csv'
output_path = MART_DATA_DIR / 'subset.csv'

# Subset the data to the first n_rows and save it to a new CSV file
def subset_data(df: pd.DataFrame, n_rows: int = 1097) -> pd.DataFrame:
    
    print(f"Subsetting the data to the first {n_rows} rows.")
    df_subset = df.iloc[:n_rows]
    df_subset.to_csv(output_path, index=False)
    
    return df_subset

SECTOR_RETURN_COLUMNS: tuple[str, ...] = (
    "return_wholesale",
    "return_groce",
    "return_retail",
    "return_petro",
    "return_service",
    "return_hospitality",
    "return_pharma",
    "return_finance",
    "return_logistics",
    "return_estate",
)

def calculate_market_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return the data with liquidity, momentum, volatility, and breadth columns."""
    required_columns = {
        "date",
        "market_volume",
        "nout_volume",
        "price",
        "return",
        *SECTOR_RETURN_COLUMNS,
    }

    # Check for missing required columns
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    # Calculate market indicators
    indicators = df.copy()
    indicators["date"] = pd.to_datetime(indicators["date"], errors="raise")
    indicators = indicators.sort_values("date", kind="stable")

    # Calculate liquidity, momentum, volatility, and market breadth
    # Liquidity: ratio of market volume to nout volume
    indicators["liquidity"] = (
        indicators["market_volume"] / indicators["nout_volume"]
    )

    # Momentum: percentage change in price over 5 and 20 days
    indicators["momentum_5d"] = indicators["price"].div(
        indicators["price"].shift(5)
    ).sub(1)
    indicators["momentum_20d"] = indicators["price"].div(
        indicators["price"].shift(20)
    ).sub(1)

    # Volatility: standard deviation of returns over 20 days
    indicators["volatility_20d"] = indicators["return"].rolling(
        window=20, min_periods=20
    ).std()

    # Market breadth: proportion of sectors with positive returns
    sector_returns = indicators.loc[:, SECTOR_RETURN_COLUMNS]
    indicators["market_breadth"] = sector_returns.gt(0).sum(axis=1).div(
        sector_returns.notna().sum(axis=1)
    )

    return indicators


def add_market_indicators(
    input_path=output_path,
    output_path=output_path,
) -> pd.DataFrame:
    """Read a subset CSV, append market indicators, and save the new CSV."""
    data = pd.read_csv(input_path)
    appended_data = calculate_market_indicators(data)
    appended_data.to_csv(output_path, index=False)
    return appended_data

if __name__ == "__main__":
    data = pd.read_csv(input_path)
    subset = subset_data(data)
    enriched_subset = calculate_market_indicators(subset)
    enriched_subset.to_csv(output_path, index=False)
    print(f"Saved {len(enriched_subset):,} rows to {output_path}")