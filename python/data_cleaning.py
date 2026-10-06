from pathlib import Path
import pandas as pd
from config import MART_DATA_DIR, STAGING_DATA_DIR

# Define input and output file paths
INPUT_FILE = STAGING_DATA_DIR / "third_merge.csv"
OUTPUT_FILE = MART_DATA_DIR / "cleaned_data.csv"

# Define the columns to retain in the cleaned dataset
RETAINED_COLUMNS = (
	"date",
	"market_me",
	"market_mtb",
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
	"cummulative",
	"phase_1",
	"phase_2",
	"phase_3",
	"phase_4",
	"deaths",
	"price",
	"open",
	"high",
	"low",
	"vol.",
	"change_%",
)

# Define columns that should be filled with zeros if missing
ZERO_FILL_COLUMNS = (
	"new",
	"cumulative",
	"phase_1",
	"phase_2",
	"phase_3",
	"phase_4",
	"deaths",
)

# Define columns for which rolling mean imputation should be applied
ROLLING_MEAN_COLUMNS = ("rsi", "pli")
MAX_INTERPOLATION_DAYS = 5


def clean_data(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
	"""Clean the merged dataset and save it to the mart directory."""
	data = pd.read_csv(input_file)

	# if "cummulative" not in data.columns and "cumulative" in data.columns:
	# 	data = data.rename(columns={"cumulative": "cummulative"})
	
    # Check for missing required columns
	missing_columns = [
		column for column in RETAINED_COLUMNS if column not in data.columns
	]
	if missing_columns:
		raise ValueError(f"Input data is missing required columns: {missing_columns}")

    # Retain only the necessary columns
	data = data.loc[:, RETAINED_COLUMNS].copy()

    # Convert the "date" column to datetime
	data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Check for missing or invalid dates
	if data["date"].isna().any():
		raise ValueError("Input data contains missing or invalid dates.")

	data = data.sort_values("date").set_index("date")

    # Convert value columns to numeric, coercing errors to NaN
	value_columns = [column for column in RETAINED_COLUMNS if column != "date"]
	data[value_columns] = data[value_columns].apply(pd.to_numeric, errors="coerce")

    # Fill missing values for specific columns
	data.loc[:, ZERO_FILL_COLUMNS] = data.loc[:, ZERO_FILL_COLUMNS].fillna(0)

    # Apply rolling mean imputation for specified columns
	for column in ROLLING_MEAN_COLUMNS:
		rolling_mean = data[column].rolling(
			window="14D", center=True, min_periods=1
		).mean()
		data[column] = data[column].fillna(rolling_mean)

    # Apply time-based interpolation for the remaining columns
	interpolation_columns = [
		column
		for column in value_columns
		if column not in ZERO_FILL_COLUMNS and column not in ROLLING_MEAN_COLUMNS
	]
	data.loc[:, interpolation_columns] = data.loc[:, interpolation_columns].interpolate(
		method="time",
		limit=MAX_INTERPOLATION_DAYS,
		limit_area="inside",
	)

    # Rename columns for consistency and reset the index
	data = data.rename(columns={"vol.": "volume", "change_%": "return"})
	# data = data.reset_index()

    # Save the cleaned data to the specified output file
	output_file = Path(output_file)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	data.to_csv(output_file, index=False)

    # Check for any remaining missing values and print them
	remaining_missing = data.isna().sum()
	remaining_missing = remaining_missing[remaining_missing > 0]
	if not remaining_missing.empty:
		print("Remaining missing values (no eligible imputation values):")
		print(remaining_missing.to_string())

	return data


if __name__ == "__main__":
	cleaned_data = clean_data()
	print(f"Cleaned {len(cleaned_data):,} rows: {OUTPUT_FILE}")
