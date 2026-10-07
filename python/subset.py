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


INDICATOR_IMPUTATION_WINDOWS: dict[str, int] = {
    "momentum_5d": 5,
    "momentum_20d": 20,
    "volatility_20d": 20,
}


def impute_indicator_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing indicator values using adjacent rolling averages."""
    imputed = df.copy()
    missing_columns = sorted(
        set(INDICATOR_IMPUTATION_WINDOWS).difference(imputed.columns)
    )

    # Check for missing indicator columns
    if missing_columns:
        raise ValueError(f"Data is missing indicator columns: {missing_columns}")

    for column, window in INDICATOR_IMPUTATION_WINDOWS.items():
        values = pd.to_numeric(imputed[column], errors="raise")
        observed = values.dropna()
        if observed.empty:
            raise ValueError(f"Cannot impute '{column}': it has no numeric values.")

        trailing_average = values.rolling(window=window, min_periods=1).mean()
        leading_average = (
            values.iloc[::-1]
            .rolling(window=window, min_periods=1)
            .mean()
            .iloc[::-1]
        )
        filled_values = values.fillna(trailing_average).fillna(leading_average)

        valid_positions = pd.RangeIndex(len(values))[values.notna().to_numpy()]
        first_valid = int(valid_positions[0])
        last_valid = int(valid_positions[-1])
        if first_valid > 0:
            filled_values.iloc[:first_valid] = filled_values.iloc[:first_valid].fillna(
                observed.iloc[:window].mean()
            )
        if last_valid < len(values) - 1:
            filled_values.iloc[last_valid + 1 :] = filled_values.iloc[
                last_valid + 1 :
            ].fillna(observed.iloc[-window:].mean())

        imputed[column] = filled_values

        if imputed[column].isna().any():
            raise ValueError(f"Cannot impute all missing values in '{column}'.")

    return imputed


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

    return impute_indicator_missing_values(indicators)


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