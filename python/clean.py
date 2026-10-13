from os import PathLike
from pathlib import Path
import numpy as np
import pandas as pd
from config import CLEANED_DATA_DIR, MERGED_DATA_DIR

INPUT_FILE: Path = MERGED_DATA_DIR / "third_merge.csv"
IMPUTED_FILE: Path = CLEANED_DATA_DIR / "imputed_data.csv"
OUTPUT_FILE: Path = CLEANED_DATA_DIR / "cleaned_data.csv"

RETAINED_COLUMNS: tuple[str, ...] = (
    "date",
    "market_me",
    "market_mtb",
    "market_volume",
    "nout_volume",
    "oil",
    "gold",
    "exchange",
    "rsi",
    "pli",
    "i",
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
    "new",
    "cumulative",
    "phase_1",
    "phase_2",
    "phase_3",
    "phase_4",
    "deaths",
    "price",
    "open",
    "high",
    "low",
    "change_%",
)

ZERO_FILL_COLUMNS: tuple[str, ...] = (
    "new",
    "cumulative",
    "phase_1",
    "phase_2",
    "phase_3",
    "phase_4",
    "deaths",
)
ROLLING_MEAN_COLUMNS: tuple[str, ...] = ("rsi", "pli")
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

def _moving_average_impute(series: pd.Series, window: int) -> pd.Series:
    """Fill missing values with local averages and nearest edge averages."""
    numeric_series = pd.Series(
        pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64"),
        index=pd.RangeIndex(len(series)),
        name=series.name,
    )
    trailing_average = numeric_series.rolling(window=window, min_periods=1).mean()
    leading_average = (
        numeric_series.iloc[::-1]
        .rolling(window=window, min_periods=1)
        .mean()
        .iloc[::-1]
    )
    imputed = numeric_series.fillna(trailing_average).fillna(leading_average)
    observed = numeric_series.dropna()

    if not observed.empty:
        valid_positions = np.flatnonzero(numeric_series.notna().to_numpy())
        first_valid = int(valid_positions[0])
        last_valid = int(valid_positions[-1])

        if first_valid > 0:
            leading_mean = observed.iloc[:window].mean()
            imputed.iloc[:first_valid] = imputed.iloc[:first_valid].fillna(
                leading_mean
            )

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


def _parse_numeric_column(
    series: pd.Series, column: str, *, is_percent: bool = False
) -> pd.Series:
    """Parse plain numbers, thousands separators, and optional percent signs."""
    values = series.astype("string").str.strip().str.replace(",", "", regex=False)
    if is_percent:
        values = values.str.replace("%", "", regex=False).str.strip()
    values = values.mask(values == "", pd.NA)

    numeric = pd.to_numeric(values, errors="coerce")
    invalid = (series.notna() & values.notna() & numeric.isna()).fillna(False)
    if invalid.any():
        examples = series.loc[invalid].astype(str).unique().tolist()[:5]
        raise ValueError(
            f"Cannot parse values in '{column}'; unexpected values: {examples}"
        )

    return numeric.replace([float("inf"), float("-inf")], float("nan"))


def clean_data(
    input_file: str | PathLike[str] = INPUT_FILE,
    output_file: str | PathLike[str] = OUTPUT_FILE,
) -> pd.DataFrame:
    """Clean the merged dataset and save the standardized intermediate CSV."""
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data: pd.DataFrame = pd.read_csv(input_path)
    data.columns = data.columns.str.strip().str.lower()

    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"Input contains duplicate column names: {duplicates}")

    if "cummulative" in data.columns and "cumulative" not in data.columns:
        data = data.rename(columns={"cummulative": "cumulative"})

    missing_columns = [
        column for column in RETAINED_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    data = data.loc[:, list(RETAINED_COLUMNS)].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        invalid_dates = data.loc[data["date"].isna()].index.tolist()[:5]
        raise ValueError(
            f"Input contains missing or invalid dates at rows: {invalid_dates}"
        )
    if data["date"].duplicated().any():
        duplicate_dates = data.loc[data["date"].duplicated(), "date"].astype(str)
        raise ValueError(
            f"Input contains duplicate dates: {duplicate_dates.unique().tolist()[:5]}"
        )

    data = data.sort_values("date").set_index("date")
    value_columns = [column for column in RETAINED_COLUMNS if column != "date"]
    for column in value_columns:
        data[column] = _parse_numeric_column(
            data[column], column, is_percent=column == "change_%"
        )

    data.loc[:, list(ZERO_FILL_COLUMNS)] = data.loc[
        :, list(ZERO_FILL_COLUMNS)
    ].fillna(0)
    for column in value_columns:
        if column in ZERO_FILL_COLUMNS:
            continue
        window = 14 if column in ROLLING_MEAN_COLUMNS else 10
        data[column] = _moving_average_impute(data[column], window)
        if data[column].isna().any():
            raise ValueError(
                f"Cannot impute '{column}': it has no numeric values to average."
            )

    data = data.rename(columns={"change_%": "return"}).reset_index()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    print(f"Saved {len(data):,} rows and {len(data.columns):,} columns to {output_path}")
    return data


def subset_data(df: pd.DataFrame, n_rows: int = 1097) -> pd.DataFrame:
    """Keep the first n_rows observations from the cleaned data."""
    if n_rows < 0:
        raise ValueError("n_rows must be non-negative.")
    print(f"Subsetting the data to the first {n_rows} rows.")
    return df.iloc[:n_rows].copy()


def impute_indicator_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing indicator values using adjacent rolling averages."""
    imputed = df.copy()
    missing_columns = sorted(
        set(INDICATOR_IMPUTATION_WINDOWS).difference(imputed.columns)
    )
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
            filled_values.iloc[:first_valid] = filled_values.iloc[
                :first_valid
            ].fillna(observed.iloc[:window].mean())
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
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    indicators = df.copy()
    indicators["date"] = pd.to_datetime(indicators["date"], errors="raise")
    indicators = indicators.sort_values("date", kind="stable")
    indicators["liquidity"] = (
        indicators["market_volume"] / indicators["nout_volume"]
    )
    indicators["momentum_5d"] = indicators["price"].div(
        indicators["price"].shift(5)
    ).sub(1)
    indicators["momentum_20d"] = indicators["price"].div(
        indicators["price"].shift(20)
    ).sub(1)
    indicators["volatility_20d"] = indicators["return"].rolling(
        window=20, min_periods=20
    ).std()
    sector_returns = indicators.loc[:, list(SECTOR_RETURN_COLUMNS)]
    indicators["market_breadth"] = sector_returns.gt(0).sum(axis=1).div(
        sector_returns.notna().sum(axis=1)
    )
    return impute_indicator_missing_values(indicators)


def clean_and_subset(
    input_file: str | PathLike[str] = INPUT_FILE,
    cleaned_file: str | PathLike[str] = IMPUTED_FILE,
    output_file: str | PathLike[str] = OUTPUT_FILE,
    n_rows: int = 1097,
) -> pd.DataFrame:
    """Clean, subset, enrich, and save the final analysis dataset."""
    cleaned_data = clean_data(input_file, cleaned_file)
    subset = subset_data(cleaned_data, n_rows)
    enriched_subset = calculate_market_indicators(subset)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_subset.to_csv(output_path, index=False)
    print(f"Saved {len(enriched_subset):,} rows to {output_path}")
    return enriched_subset


if __name__ == "__main__":
    clean_and_subset()
