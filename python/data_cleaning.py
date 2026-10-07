from pathlib import Path
import re
from os import PathLike
import numpy as np
import pandas as pd
from config import MART_DATA_DIR, STAGING_DATA_DIR

# Define the input and output file paths
INPUT_FILE: Path = STAGING_DATA_DIR / "third_merge.csv"
OUTPUT_FILE: Path = MART_DATA_DIR / "cleaned_data.csv"

# Define the columns to retain, fill with zeros, and apply rolling mean
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
ROLLING_MEAN_COLUMNS: tuple[str, str] = ("rsi", "pli")

# VOLUME_MULTIPLIERS: dict[str, int] = {
#     "": 1,
#     "K": 1_000,
#     "M": 1_000_000,
#     "B": 1_000_000_000,
# }

# Define imputation functions for missing values in numeric columns
def _moving_average_impute(
    series: pd.Series[object], window: int
) -> pd.Series[float]:
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
    series: pd.Series[object], column: str, *, is_percent: bool = False
) -> pd.Series[float]:
    """Parse plain numbers, thousands separators, and optional percent signs."""
    values = series.astype("string").str.strip().str.replace(",", "", regex=False)
    if is_percent:
        values = values.str.rstrip("%").str.strip()

    numeric = pd.to_numeric(values.replace("", pd.NA), errors="coerce")
    invalid = (series.notna() & values.ne("") & numeric.isna()).fillna(False)
    if invalid.any():
        examples = series.loc[invalid].astype(str).unique().tolist()[:5]
        raise ValueError(
            f"Cannot parse values in '{column}'; unexpected values: {examples}"
        )

    return numeric.replace([float("inf"), float("-inf")], float("nan"))


# def _parse_volume(series: pd.Series[object]) -> pd.Series[float]:
#     """Convert volume strings such as '268.79K' or '1.2M' to whole units."""
#     values = series.astype("string").str.strip().str.replace(",", "", regex=False)
#     parts = values.str.extract(
#         r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([KMB]?)$",
#         flags=re.IGNORECASE,
#         expand=True,
#     )
#     invalid = (series.notna() & values.ne("") & parts[0].isna()).fillna(False)
#     if invalid.any():
#         examples = series.loc[invalid].astype(str).unique().tolist()[:5]
#         raise ValueError(f"Cannot parse volume values: {examples}")

#     number = pd.to_numeric(parts[0], errors="coerce")
#     multiplier = parts[1].str.upper().map(VOLUME_MULTIPLIERS)
#     return number * multiplier

def clean_data(
    input_file: str | PathLike[str] = INPUT_FILE,
    output_file: str | PathLike[str] = OUTPUT_FILE,
) -> pd.DataFrame:
    """Clean the merged dataset and save the standardized mart CSV."""
    data = pd.read_csv(input_file)
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

    data = data.loc[:, RETAINED_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        invalid_dates = data.loc[data["date"].isna()].index.tolist()[:5]
        raise ValueError(f"Input contains missing or invalid dates at rows: {invalid_dates}")
    if data["date"].duplicated().any():
        duplicate_dates = data.loc[data["date"].duplicated(), "date"].astype(str)
        raise ValueError(
            f"Input contains duplicate dates: {duplicate_dates.unique().tolist()[:5]}"
        )

    data = data.sort_values("date").set_index("date")
    value_columns = [column for column in RETAINED_COLUMNS if column != "date"]
    # for column in value_columns:
    #     if column == "market_volume":
    #         data[column] = _parse_volume(data[column])
    #     else:
    #         data[column] = _parse_numeric_column(
    #             data[column], column, is_percent=column == "change_%"
    #         )
    for column in value_columns:
        data[column] = _parse_numeric_column(
        data[column], column, is_percent=column == "change_%"
        )
            

    data.loc[:, ZERO_FILL_COLUMNS] = data.loc[:, ZERO_FILL_COLUMNS].fillna(0)
    for column in value_columns:
        if column in ZERO_FILL_COLUMNS:
            continue
        window = 14 if column in ROLLING_MEAN_COLUMNS else 10
        data[column] = _moving_average_impute(data[column], window)
        if data[column].isna().any():
            raise ValueError(
                f"Cannot impute '{column}': it has no numeric values to average."
            )

    data = data.rename(columns={"change_%": "return"})
    data = data.reset_index()

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_file, index=False)

    print(
        f"Saved {len(data):,} rows and {len(data.columns):,} columns to {output_file}"
    )
    return data


if __name__ == "__main__":
    clean_data()
