import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import CLEANED_DATA_DIR

INPUT_PATH = CLEANED_DATA_DIR / "cleaned_data.csv"
STANDARDIZED_DATA_PATH = CLEANED_DATA_DIR / "standardized_data.csv"

EXCLUDED_COLUMNS: tuple[str, ...] = (
    "date",
    "phase_1",
    "phase_2",
    "phase_3",
    "phase_4",
)


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize numeric features while preserving excluded columns."""
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

    standardized_data = df.copy()
    standardized_data.loc[:, feature_columns] = standardized_values
    return standardized_data


def standardize_set(input_path=INPUT_PATH, standardized_data_path=STANDARDIZED_DATA_PATH,) -> pd.DataFrame:
    """Read cleaned_data.csv and save its StandardScaler-transformed features."""
    data = pd.read_csv(input_path)
    standardized_data = transform_data(data)
    standardized_data.to_csv(standardized_data_path, index=False)
    return standardized_data


if __name__ == "__main__":
    standardized_data = standardize_set()
    print(f"Saved {len(standardized_data):,} rows to {STANDARDIZED_DATA_PATH}")