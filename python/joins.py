import pandas as pd
from config import RAW_DATA_DIR, MERGED_DATA_DIR

# Raw data file paths
market_data = RAW_DATA_DIR/"market_data.csv"
covid_cases = RAW_DATA_DIR/"covid_cases.csv"
death_cases = RAW_DATA_DIR/"death_cases.csv"
historical_data = RAW_DATA_DIR/"historical_data.csv"

# Merged output file paths
first_merge = MERGED_DATA_DIR/"first_merge.csv"
second_merge = MERGED_DATA_DIR/"second_merge.csv"
third_merge = MERGED_DATA_DIR/"third_merge.csv"

# ===============================================
# FIRST MERGE: Merge covid cases with death cases
# ===============================================

# Load raw CSV files
df_covid_cases = pd.read_csv(covid_cases)
df_death_cases = pd.read_csv(death_cases)
df_market_data = pd.read_csv(market_data)
df_historical_data = pd.read_csv(historical_data)

print(f"Covid cases loaded: {len(df_covid_cases):,} rows")
print(f"Death cases loaded: {len(df_death_cases):,} rows")
print(f"Market data loaded: {len(df_market_data):,} rows")
print(f"Historical data loaded: {len(df_historical_data):,} rows")

# Change historical data column names to lowercase and replace spaces with underscores
df_historical_data.columns = df_historical_data.columns.str.lower().str.replace(" ", "_")

# Convert date columns
df_covid_cases["date"] = pd.to_datetime(df_covid_cases["date"], errors="coerce").dt.date
df_death_cases["date"] = pd.to_datetime(df_death_cases["date"], errors="coerce").dt.date
df_market_data["date"] = pd.to_datetime(df_market_data["date"], errors="coerce").dt.date
df_historical_data["date"] = pd.to_datetime(df_historical_data["date"], errors="coerce").dt.date

# LEFT JOIN (COVID CASES AND DEATHS)
print("\nMerging covid cases and death cases...")
first_merged = pd.merge(df_covid_cases, df_death_cases, on="date", how="left")
first_merged.to_csv(first_merge, index=False)

# Summary
print("\nFirst merge completed successfully.")
print(f"Covid cases rows:      {len(df_covid_cases):,}")
print(f"Death cases rows:      {len(df_death_cases):,}")
print(f"Matched rows:     {len(first_merged):,}")
print(f"Output columns:   {len(first_merged.columns):,}")
print(f"Output file:      {first_merge}")

# ================================================================
# SECOND MERGE: Merge market data with covid cases and death cases
# ================================================================

# Load first merge output CSV data
df_cases_deaths = pd.read_csv(first_merge)

print(f"Covid cases deaths loaded: {len(df_cases_deaths):,} rows")
print(f"Market data loaded: {len(df_market_data):,} rows")

# Convert date columns
df_cases_deaths["date"] = pd.to_datetime(df_cases_deaths["date"], errors="coerce").dt.date

# LEFT JOIN (CASES_DEATHS AND MARKET DATA)
print("\nMerging market data with covid cases and death cases...")
second_merged = pd.merge(df_market_data, df_cases_deaths, on="date", how="left")
second_merged.to_csv(second_merge, index=False)

# Summary
print("\nSecond merge completed successfully.")
print(f"Covid cases and deaths rows:      {len(df_cases_deaths):,}")
print(f"Market data rows:      {len(df_market_data):,}")
print(f"Matched rows:     {len(second_merged):,}")
print(f"Output columns:   {len(second_merged.columns):,}")
print(f"Output file:      {second_merge}")

# =================================================================================
# THIRD MERGE: Merge historical data with covid cases, death cases, and market data
# =================================================================================

# Load second merge output CSV data
df_cases_deaths_market = pd.read_csv(second_merge)

print(f"Covid cases, deaths, and market data loaded: {len(df_cases_deaths_market):,} rows")

# Convert date columns
df_cases_deaths_market["date"] = pd.to_datetime(df_cases_deaths_market["date"],errors="coerce").dt.date

# LEFT JOIN (MARKET_COVID AND OCLH DATA)
print("\nMerging historical data with covid cases, death cases, and market data...")

third_merged = pd.merge(df_cases_deaths_market, df_historical_data, on="date", how="left")
third_merged.to_csv(third_merge, index=False)

# Summary
print("\nThird merge completed successfully.")
print(f"Historical data rows:      {len(df_historical_data):,}")
print(f"Covid cases, deaths, and market data rows:      {len(df_cases_deaths_market):,}")
print(f"Matched rows:     {len(third_merged):,}")
print(f"Output columns:   {len(third_merged.columns):,}")
print(f"Output file:      {third_merge}")