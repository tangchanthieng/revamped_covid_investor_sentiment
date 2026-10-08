from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_arch,
    het_breuschpagan,
)
from statsmodels.stats.stattools import durbin_watson

# Define data paths and constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "mart" / "data.csv"

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


def run_benchmark(
    input_file: str | Path = INPUT_FILE,
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

    # Check for missing required columns and return predictors
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")
    if not return_predictors:
        raise ValueError("Input data must contain at least one 'return_' predictor.")

    # Convert the 'date' column to datetime and check for missing or invalid dates
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        invalid_rows = data.index[data["date"].isna()].tolist()[:5]
        raise ValueError(
            f"Input contains missing or invalid dates at rows: {invalid_rows}"
        )
    data = data.sort_values("date", kind="stable").copy()

    # Convert numeric columns to numeric types and raise errors for non-numeric values
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
            "No complete observations remain after differencing and "
            "dropping missing values."
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
    run_benchmark()