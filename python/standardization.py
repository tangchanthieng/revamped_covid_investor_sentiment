import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import MART_DATA_DIR

INPUT_PATH = MART_DATA_DIR / "subset.csv"
STANDARDIZED_SET01_PATH = MART_DATA_DIR / "standardized_set01.csv"
STANDARDIZED_SET02_PATH = MART_DATA_DIR / "standardized_set02.csv"

EXCLUDED_COLUMNS: tuple[str, ...] = (
    "date",
    "phase_1",
    "phase_2",
    "phase_3",
    "phase_4",
)


def transform_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create StandardScaler and signed natural-log versions of the dataset."""
    missing_columns = sorted(set(EXCLUDED_COLUMNS).difference(df.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    feature_columns = [column for column in df.columns if column not in EXCLUDED_COLUMNS]
    non_numeric_columns = df[feature_columns].select_dtypes(
        exclude="number"
    ).columns.tolist()
    if non_numeric_columns:
        raise TypeError(
            f"Feature columns must be numeric: {non_numeric_columns}"
        )

    standard_scaler = StandardScaler()
    standardized_values = standard_scaler.fit_transform(df[feature_columns])
    logged_values = np.sign(df[feature_columns]) * np.log1p(
        np.abs(df[feature_columns])
    )

    standardized_set01 = df.copy()
    standardized_set01.loc[:, feature_columns] = standardized_values

    standardized_set02 = df.copy()
    standardized_set02.loc[:, feature_columns] = logged_values

    return standardized_set01, standardized_set02


def standardize_subset(
    input_path=INPUT_PATH,
    set01_path=STANDARDIZED_SET01_PATH,
    set02_path=STANDARDIZED_SET02_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read subset.csv, transform its features, and save both output datasets."""
    data = pd.read_csv(input_path)
    standardized_set01, standardized_set02 = transform_data(data)

    standardized_set01.to_csv(set01_path, index=False)
    standardized_set02.to_csv(set02_path, index=False)
    return standardized_set01, standardized_set02


if __name__ == "__main__":
    set01, set02 = standardize_subset()
    print(f"Saved {len(set01):,} rows to {STANDARDIZED_SET01_PATH}")
    print(f"Saved {len(set02):,} rows to {STANDARDIZED_SET02_PATH}")