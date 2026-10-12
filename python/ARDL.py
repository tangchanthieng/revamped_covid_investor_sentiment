from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    breaks_cusumolsresid,
    recursive_olsresiduals,
)
from statsmodels.tsa.ardl import ARDL


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "mart" / "conditional_volatility.csv"
FIGURE_FILE = PROJECT_DIR / "assets" / "CUSUM_CUSMSQ.png"

DEPENDENT_VARIABLE = "market_variance"
EXOGENOUS_VARIABLES = (
    "SIS",
    "momentum_5d",
    "return_finance",
    "return_logistics",
    "return_estate",
    "oil",
    "exchange",
    "liquidity",
    "new",
    "phase_1",
    "phase_2",
    "phase_3",
    "phase_4",
    "deaths",
)
ENDOGENOUS_LAGS = 1
EXOGENOUS_LAGS = 1
CUSUMSQ_SIMULATIONS = 10_000
RANDOM_SEED = 2026


def _build_ardl_design(data: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Build the OLS design corresponding to the ARDL(1, 1, ..., 1) model."""
    endog = data[DEPENDENT_VARIABLE]
    design = pd.DataFrame(index=data.index)
    design["const"] = 1.0
    design[f"{DEPENDENT_VARIABLE}.L1"] = endog.shift(1)
    for variable in EXOGENOUS_VARIABLES:
        design[f"{variable}.L0"] = data[variable]
        design[f"{variable}.L1"] = data[variable].shift(1)

    model_data = pd.concat([endog.rename(DEPENDENT_VARIABLE), design], axis=1)
    model_data = model_data.dropna()
    return model_data[DEPENDENT_VARIABLE], model_data.drop(
        columns=DEPENDENT_VARIABLE
    )


def _recursive_residuals(
    results: sm.regression.linear_model.RegressionResultsWrapper,
) -> tuple[np.ndarray, int]:
    """Return standardized recursive residuals and the initial estimation size."""
    design = np.asarray(results.model.exog)
    parameter_count = design.shape[1]
    initial_size = next(
        (
            size
            for size in range(parameter_count, len(design) + 1)
            if np.linalg.matrix_rank(design[:size]) == parameter_count
        ),
        None,
    )
    if initial_size is None or initial_size >= len(design):
        raise ValueError(
            "The ARDL design does not provide enough observations for "
            "recursive residual diagnostics."
        )

    recursive_results = recursive_olsresiduals(
        results,
        skip=initial_size,
        alpha=0.95,
    )
    standardized_residuals = np.asarray(recursive_results[3])
    standardized_residuals = standardized_residuals[
        np.isfinite(standardized_residuals)
    ]
    if len(standardized_residuals) < 2:
        raise ValueError(
            "Too few recursive residuals are available for CUSUM diagnostics."
        )
    return standardized_residuals, initial_size


def _cusumsq_test(
    recursive_residuals: np.ndarray,
    simulations: int = CUSUMSQ_SIMULATIONS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Test CUSUM of squares using a finite-sample simulated 95% envelope."""
    squared_residuals = np.square(recursive_residuals)
    cumulative_squares = np.cumsum(squared_residuals)
    cusumsq = cumulative_squares / cumulative_squares[-1]

    periods = np.arange(1, len(cusumsq) + 1, dtype=float)
    expected = periods / len(cusumsq)
    statistic = float(np.max(np.abs(cusumsq - expected)))

    rng = np.random.default_rng(seed)
    simulated_statistics = np.empty(simulations)
    batch_size = 500
    for start in range(0, simulations, batch_size):
        batch_count = min(batch_size, simulations - start)
        simulated_squares = rng.chisquare(
            df=1,
            size=(batch_count, len(cusumsq)),
        )
        simulated_paths = np.cumsum(simulated_squares, axis=1)
        simulated_paths /= simulated_paths[:, -1, np.newaxis]
        simulated_statistics[start : start + batch_count] = np.max(
            np.abs(simulated_paths - expected),
            axis=1,
        )

    critical_value = float(np.quantile(simulated_statistics, 0.95))
    p_value = float(
        (1 + np.count_nonzero(simulated_statistics >= statistic))
        / (simulations + 1)
    )
    lower = np.clip(expected - critical_value, 0.0, 1.0)
    upper = np.clip(expected + critical_value, 0.0, 1.0)
    return statistic, p_value, critical_value, lower, upper


def run_ardl(
    input_file: str | Path = INPUT_FILE,
    figure_file: str | Path = FIGURE_FILE,
    show_figure: bool = True,
) -> None:
    """Fit the ARDL model, print diagnostic tests, and save/display test plots."""
    input_path = Path(input_file)
    data = pd.read_csv(input_path)
    data.columns = data.columns.str.strip()

    required_columns = {DEPENDENT_VARIABLE, *EXOGENOUS_VARIABLES}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required columns: {missing_columns}")

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        if data["date"].isna().any():
            invalid_rows = data.index[data["date"].isna()].tolist()[:5]
            raise ValueError(
                f"Input contains missing or invalid dates at rows: {invalid_rows}"
            )
        data = data.sort_values("date", kind="stable")

    data = data.reset_index(drop=True)
    numeric_columns = [DEPENDENT_VARIABLE, *EXOGENOUS_VARIABLES]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    model_data = data[numeric_columns]
    missing_rows = model_data.index[model_data.isna().any(axis=1)].tolist()
    if missing_rows:
        raise ValueError(
            "Model columns contain missing values at rows: "
            f"{missing_rows[:5]}"
        )
    if len(model_data) < 3:
        raise ValueError("At least three complete observations are required.")
    if not np.isfinite(model_data.to_numpy()).all():
        raise ValueError("Model columns contain infinite values.")

    model = ARDL(
        endog=model_data[DEPENDENT_VARIABLE],
        lags=ENDOGENOUS_LAGS,
        exog=model_data[list(EXOGENOUS_VARIABLES)],
        order=EXOGENOUS_LAGS,
        trend="c",
        missing="raise",
    )
    results = model.fit()

    design_endog, design_exog = _build_ardl_design(model_data)
    ols_data = pd.concat([design_endog, design_exog], axis=1)
    ols_results = sm.OLS(
        ols_data[DEPENDENT_VARIABLE],
        ols_data.drop(columns=DEPENDENT_VARIABLE),
    ).fit()
    if not np.allclose(
        np.asarray(results.params),
        np.asarray(ols_results.params),
        rtol=1e-7,
        atol=1e-10,
    ):
        raise RuntimeError(
            "The diagnostic regression does not match the fitted ARDL model."
        )

    recursive_resid, initial_size = _recursive_residuals(ols_results)
    cusum_stat, cusum_pvalue, critical_values = breaks_cusumolsresid(
        recursive_resid,
        ddof=0,
    )
    cusum_critical_value = float(
        next(value for level, value in critical_values if level == 5)
    )
    cusum_path = np.cumsum(recursive_resid) / np.sqrt(
        np.sum(np.square(recursive_resid))
    )
    cusumsq_stat, cusumsq_pvalue, cusumsq_critical, lower, upper = _cusumsq_test(
        recursive_resid
    )

    recursive_dates = None
    if "date" in data.columns:
        estimation_dates = data.loc[ols_data.index, "date"].reset_index(drop=True)
        recursive_dates = estimation_dates.iloc[
            len(estimation_dates) - len(recursive_resid) :
        ]
    x_axis = (
        recursive_dates
        if recursive_dates is not None
        else np.arange(1, len(recursive_resid) + 1)
    )

    print(f"Input file: {input_path.resolve()}")
    print(f"ARDL order: ARDL({ENDOGENOUS_LAGS}, {EXOGENOUS_LAGS})")
    print(f"Observations used: {int(results.nobs):,}")
    print("\nARDL model results")
    print(results.summary())
    print(
        "\nCUSUM test on standardized recursive residuals "
        f"({len(recursive_resid)} recursive observations)"
    )
    print("Null hypothesis: model parameters are stable over time")
    print(
        f"Statistic: {cusum_stat:.4f}; 5% critical value: "
        f"{cusum_critical_value:.4f}; p-value: {cusum_pvalue:.6g}"
    )
    print(
        "\nCUSUMSQ test on standardized recursive residuals "
        f"({len(recursive_resid)} recursive observations)"
    )
    print("Null hypothesis: model parameters are stable over time")
    print(
        f"Statistic: {cusumsq_stat:.4f}; simulated 5% critical value: "
        f"{cusumsq_critical:.4f}; p-value: {cusumsq_pvalue:.6g}"
    )
    if recursive_dates is not None:
        print(
            "Recursive diagnostics begin at: "
            f"{recursive_dates.iloc[0].date()} "
            f"(initial full-rank sample: {initial_size} observations)"
        )

    figure, (cusum_axis, cusumsq_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
    )
    cusum_axis.plot(x_axis, cusum_path, label="CUSUM")
    cusum_axis.axhline(
        cusum_critical_value,
        color="red",
        linestyle="--",
        label="5% critical bounds",
    )
    cusum_axis.axhline(-cusum_critical_value, color="red", linestyle="--")
    cusum_axis.axhline(0, color="black", linewidth=0.8)
    cusum_axis.set_title("CUSUM test")
    cusum_axis.set_ylabel("Cumulative sum")
    cusum_axis.legend()
    cusum_axis.grid(alpha=0.25)

    cusumsq_axis.plot(
        x_axis,
        np.cumsum(recursive_resid**2) / np.sum(recursive_resid**2),
        label="CUSUM of squares",
    )
    cusumsq_axis.fill_between(
        x_axis,
        lower,
        upper,
        color="red",
        alpha=0.15,
        label="95% simultaneous null envelope",
    )
    cusumsq_axis.plot(
        x_axis,
        np.arange(1, len(recursive_resid) + 1) / len(recursive_resid),
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="Stable-parameter path",
    )
    cusumsq_axis.set_title("CUSUMSQ test")
    cusumsq_axis.set_ylabel("Cumulative proportion")
    cusumsq_axis.set_xlabel("Date" if recursive_dates is not None else "Observation")
    cusumsq_axis.legend()
    cusumsq_axis.grid(alpha=0.25)

    output_path = Path(figure_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    print(f"\nSaved diagnostic figures to: {output_path.resolve()}")
    if show_figure and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    run_ardl()