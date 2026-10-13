from pathlib import Path
import warnings
from config import CLEANED_DATA_DIR
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from statsmodels.stats.diagnostic import het_arch


# PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = CLEANED_DATA_DIR / "data.csv"
OUTPUT_FILE = CLEANED_DATA_DIR / "market_variance.csv"
LEVEL_EXOGENOUS = ("oil", "gold", "exchange", "i")
COVID_PREDICTORS = ("new", "phase_1", "phase_2", "phase_3", "phase_4", "deaths")
ARCH_LAGS = 5


def run_garch(
    input_file: str | Path = INPUT_FILE,
    output_file: str | Path = OUTPUT_FILE,
) -> ARCHModelResult:
    """Fit GARCH models and save TARCH conditional variance with the input data."""
    source_data = pd.read_csv(input_file)
    output_data = source_data.copy()
    data = source_data.copy()
    data.columns = data.columns.str.strip()
    output_data.columns = output_data.columns.str.strip()

    return_predictors = [
        column for column in data.columns if column.startswith("return_")
    ]
    differenced_columns = (
        "market_me",
        "market_mtb",
        "market_volume",
        "oil",
        "gold",
        "exchange",
        "i",
        "liquidity",
    )
    level_predictors = (
        "SIS",
        "momentum_5d",
        "momentum_20d",
        "volatility_20d",
        "market_breadth",
    )
    required_columns = {
        "date",
        "return",
        *differenced_columns,
        *level_predictors,
        *LEVEL_EXOGENOUS,
        *COVID_PREDICTORS,
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
        *differenced_columns,
        *level_predictors,
        *COVID_PREDICTORS,
        *return_predictors,
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    for column in ("phase_1", "phase_2", "phase_3", "phase_4"):
        if not data[column].dropna().isin((0, 1)).all():
            raise ValueError(f"Phase dummy '{column}' must contain only 0/1 values.")
    if (data[["phase_1", "phase_2", "phase_3", "phase_4"]].sum(axis=1) > 1).any():
        raise ValueError("Phase dummy columns must be mutually exclusive.")

    difference_predictors = [f"diff_{column}" for column in differenced_columns]
    for column, difference_column in zip(differenced_columns, difference_predictors):
        data[difference_column] = data[column].diff()

    predictors = [
        *level_predictors,
        *return_predictors,
        *difference_predictors,
        *LEVEL_EXOGENOUS,
        *COVID_PREDICTORS,
    ]
    model_data = data[["return", *predictors]].dropna()
    if model_data.empty:
        raise ValueError(
            "No complete observations remain after differencing and "
            "dropping missing values."
        )
    if len(model_data) <= ARCH_LAGS:
        raise ValueError(
            f"At least {ARCH_LAGS + 1} complete observations are required "
            f"for the {ARCH_LAGS}-lag ARCH test."
        )

    exogenous = model_data[predictors]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        results = arch_model(
            model_data["return"],
            x=exogenous,
            mean="ARX",
            lags=0,
            vol="GARCH",
            p=1,
            q=1,
            dist="normal",
            rescale=False,
        ).fit(disp="off")
        tarch_results = arch_model(
            model_data["return"],
            x=exogenous,
            mean="ARX",
            lags=0,
            vol="GARCH",
            p=1,
            o=1,
            q=1,
            power=1.0,
            dist="normal",
            rescale=False,
        ).fit(disp="off")

    if results.convergence_flag != 0:
        raise RuntimeError(
            "GARCH optimizer did not converge "
            f"(convergence flag: {results.convergence_flag})."
        )
    if tarch_results.convergence_flag != 0:
        raise RuntimeError(
            "TARCH optimizer did not converge "
            f"(convergence flag: {tarch_results.convergence_flag})."
        )

    market_variance = pd.Series(
        tarch_results.conditional_volatility,
        index=model_data.index,
    ).pow(2)
    output_data["market_variance"] = float("nan")
    output_data.loc[market_variance.index, "market_variance"] = market_variance
    missing_variance_count = int(output_data["market_variance"].isna().sum())
    variance_mean = output_data["market_variance"].mean()
    if pd.isna(variance_mean):
        raise ValueError("Cannot impute market_variance: no estimates are available.")
    output_data["market_variance"] = output_data["market_variance"].fillna(
        variance_mean
    )
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data.to_csv(output_path, index=False)

    arch_lm, arch_lm_pvalue, arch_f, arch_f_pvalue = het_arch(
        results.std_resid.dropna(),
        nlags=ARCH_LAGS,
    )
    tarch_lm, tarch_lm_pvalue, tarch_f, tarch_f_pvalue = het_arch(
        tarch_results.std_resid.dropna(),
        nlags=ARCH_LAGS,
    )

    print(f"Input file: {Path(input_file).resolve()}")
    print(f"Observations used: {int(results.nobs):,}")
    print("\nGARCH(1, 1) model results")
    print(results.summary())
    print("\nTARCH(1, 1) model results")
    print(tarch_results.summary())
    print(f"\nEngle ARCH LM test on standardized residuals ({ARCH_LAGS} lags)")
    print("GARCH null hypothesis: no remaining ARCH effects")
    print(f"LM statistic: {arch_lm:.4f}; p-value: {arch_lm_pvalue:.6g}")
    print(f"F statistic:  {arch_f:.4f}; p-value: {arch_f_pvalue:.6g}")
    print("TARCH null hypothesis: no remaining ARCH effects")
    print(f"LM statistic: {tarch_lm:.4f}; p-value: {tarch_lm_pvalue:.6g}")
    print(f"F statistic:  {tarch_f:.4f}; p-value: {tarch_f_pvalue:.6g}")
    print(f"\nSaved TARCH conditional variance to: {output_path.resolve()}")
    print(f"Rows imputed with mean market variance: {missing_variance_count:,}")
    print(f"Mean market variance used for imputation: {variance_mean:.8g}")

    return results


if __name__ == "__main__":
    run_garch()