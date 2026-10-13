from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_arch,
    het_breuschpagan,
)
from statsmodels.stats.stattools import durbin_watson

from config import CLEANED_DATA_DIR

MARKET_DATA_PATH = CLEANED_DATA_DIR / "standardized_data.csv"
SENTIMENT_DATA_PATH = CLEANED_DATA_DIR / "SIS.csv"
MODEL_DATA_PATH = CLEANED_DATA_DIR / "data.csv"

DIFFERENCE_COLUMNS = (
    "market_me",
    "market_mtb",
    "market_volume",
    "nout_volume",
    "oil",
    "gold",
    "exchange",
    "i",
    "liquidity",
)
LEVEL_PREDICTORS = (
    "SIS",
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "market_breadth",
)
AUTOCORRELATION_LAGS = 5
ARCH_LAGS = 5


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


def prepare_model_data(
    market_data_path: str | Path = MARKET_DATA_PATH,
    sentiment_data_path: str | Path = SENTIMENT_DATA_PATH,
    output_path: str | Path = MODEL_DATA_PATH,
) -> pd.DataFrame:
    """Merge market data with the SIS sentiment series and save the benchmark dataset."""
    market_df = pd.read_csv(market_data_path)
    sentiment_df = pd.read_csv(sentiment_data_path)

    market_df.columns = market_df.columns.str.strip()
    sentiment_df.columns = sentiment_df.columns.str.strip()

    market_df["date"] = pd.to_datetime(market_df["date"], errors="coerce").dt.date
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"], errors="coerce").dt.date

    if "SIS" not in sentiment_df.columns:
        raise ValueError(f"{sentiment_data_path} must contain an 'SIS' column.")

    print(f"Sentiment data loaded: {len(sentiment_df):,} rows")
    print(f"Market data loaded: {len(market_df):,} rows")
    print("\nMerging market data and sentiment data...")

    merge_df = pd.merge(market_df, sentiment_df, on="date", how="left")
    merge_df["SIS"] = _moving_average_impute(merge_df["SIS"], window=7)
    if merge_df["SIS"].isna().any():
        raise ValueError("The SIS column still contains missing values after imputation.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merge_df.to_csv(output_path, index=False)

    print("\nMerge completed successfully.")
    print(f"Matched rows:     {len(merge_df):,}")
    print(f"Output columns:   {len(merge_df.columns):,}")
    print(f"Rows with missing values (after imputation): {merge_df.isnull().sum().sum():,}")
    return merge_df


def run_benchmark(
    input_file: str | Path = MODEL_DATA_PATH,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Estimate the benchmark OLS model and print diagnostic test results."""
    data = pd.read_csv(input_file)
    data.columns = data.columns.str.strip()

    return_predictors = [
        column for column in data.columns if column.startswith("return_")
    ]
    required_columns = {
        "date",
        "return",
        *DIFFERENCE_COLUMNS,
        *LEVEL_PREDICTORS,
    }

    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")
    if not return_predictors:
        raise ValueError("Input data must contain at least one 'return_' predictor.")

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        invalid_rows = data.index[data["date"].isna()].tolist()[:5]
        raise ValueError(
            f"Input contains missing or invalid dates at rows: {invalid_rows}"
        )
    data = data.sort_values("date", kind="stable").copy()

    numeric_columns = [
        "return",
        *DIFFERENCE_COLUMNS,
        *LEVEL_PREDICTORS,
        *return_predictors,
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    difference_predictors = [
        "market_me",
        "market_mtb",
        "oil",
        "gold",
        "exchange",
        "i",
        "liquidity",
    ]
    for column in DIFFERENCE_COLUMNS:
        data[f"diff_{column}"] = data[column].diff()

    predictors = [
        *LEVEL_PREDICTORS,
        *return_predictors,
        *(f"diff_{column}" for column in difference_predictors),
    ]

    model_data = data[["return", *predictors]].dropna()
    if model_data.empty:
        raise ValueError(
            "No complete observations remain after differencing and dropping missing values."
        )
    if len(model_data) <= AUTOCORRELATION_LAGS:
        raise ValueError(
            "At least six complete observations are required for the "
            f"{AUTOCORRELATION_LAGS}-lag autocorrelation test."
        )

    design = sm.add_constant(model_data[predictors], has_constant="add")
    results = sm.OLS(model_data["return"], design).fit()

    print(f"Input file: {Path(input_file).resolve()}")
    print(f"Observations used: {int(results.nobs):,}")
    print("\nBenchmark OLS results")
    print(results.summary())

    bp_lm, bp_lm_pvalue, bp_f, bp_f_pvalue = het_breuschpagan(
        results.resid,
        results.model.exog,
    )
    bg_lm, bg_lm_pvalue, bg_f, bg_f_pvalue = acorr_breusch_godfrey(
        results,
        nlags=AUTOCORRELATION_LAGS,
        result_object=False,
    )
    arch_lm, arch_lm_pvalue, arch_f, arch_f_pvalue = het_arch(
        results.resid,
        nlags=ARCH_LAGS,
    )

    print("\nBreusch-Pagan heteroskedasticity test")
    print("Null hypothesis: homoskedastic errors")
    print(f"LM statistic: {bp_lm:.4f}; p-value: {bp_lm_pvalue:.6g}")
    print(f"F statistic:  {bp_f:.4f}; p-value: {bp_f_pvalue:.6g}")

    print(f"\nBreusch-Godfrey autocorrelation test ({AUTOCORRELATION_LAGS} lags)")
    print("Null hypothesis: no serial correlation through the specified lag")
    print(f"LM statistic: {bg_lm:.4f}; p-value: {bg_lm_pvalue:.6g}")
    print(f"F statistic:  {bg_f:.4f}; p-value: {bg_f_pvalue:.6g}")
    print(f"Durbin-Watson statistic: {durbin_watson(results.resid):.4f}")

    print(f"\nEngle ARCH LM test ({ARCH_LAGS} lags)")
    print("Null hypothesis: no ARCH effects")
    print(f"LM statistic: {arch_lm:.4f}; p-value: {arch_lm_pvalue:.6g}")
    print(f"F statistic:  {arch_f:.4f}; p-value: {arch_f_pvalue:.6g}")

    return results


if __name__ == "__main__":
    prepare_model_data()
    run_benchmark()
