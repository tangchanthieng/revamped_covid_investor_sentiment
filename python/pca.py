from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from config import ASSETS_DIR, CLEANED_DATA_DIR

INPUT_PATH = CLEANED_DATA_DIR / "standardized_data.csv"
IS_INDEX_OUTPUT_PATH = CLEANED_DATA_DIR / "IS.csv"
SIS_INDEX_OUTPUT_PATH = CLEANED_DATA_DIR / "SIS.csv"
IS_CORRELATION_MAP_PATH = ASSETS_DIR / "IS_correlation_map.png"
SIS_CORRELATION_MAP_PATH = ASSETS_DIR / "SIS_correlation_map.png"

BASE_VARIABLES: tuple[str, ...] = (
    "return",
    "market_me",
    "market_mtb",
    "liquidity",
    "rsi",
    "pli",
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "market_breadth",
)

PCA_VARIABLES: tuple[str, ...] = (
    "pli",
    "rsi",
    "market_breadth_lag1",
    "momentum_5d",
    "return_estate_lag1",
    "return_finance_lag1",
    "momentum_20d",
    "return_lag1",
    "return_logistics_lag1",
    "return_pharma_lag1",
    "return_wholesale_lag1",
    "return_groce_lag1",
    "return_retail_lag1",
    "market_mtb",
    "return_service_lag1",
    "return_hospitality_lag1",
    "liquidity",
    "market_me",
    "return_petro_lag1",
    "volatility_20d",
)


def calculate_kmo(data: pd.DataFrame) -> tuple[float, pd.Series]:
    """Calculate overall and per-variable Kaiser-Meyer-Olkin measures."""
    correlation = data.corr().to_numpy().copy()
    try:
        precision = np.linalg.inv(correlation)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "KMO cannot be calculated because the feature correlation matrix "
            "is singular."
        ) from error

    diagonal_scale = np.sqrt(np.diag(precision))
    partial_correlation = -precision / np.outer(diagonal_scale, diagonal_scale)
    np.fill_diagonal(correlation, 0)
    np.fill_diagonal(partial_correlation, 0)

    correlation_squared = correlation**2
    partial_squared = partial_correlation**2
    correlation_sums = correlation_squared.sum(axis=0)
    partial_sums = partial_squared.sum(axis=0)
    denominator = correlation_sums + partial_sums
    if np.any(denominator == 0):
        invalid = data.columns[denominator == 0].tolist()
        raise ValueError(f"KMO is undefined for variables with no correlations: {invalid}")

    variable_kmo = pd.Series(
        correlation_sums / denominator,
        index=data.columns,
        name="kmo",
    )
    overall_kmo = float(
        correlation_squared.sum()
        / (correlation_squared.sum() + partial_squared.sum())
    )
    return overall_kmo, variable_kmo


def _test_index_correlations(
    index: pd.Series,
    variables: pd.DataFrame,
    index_name: str,
    output_path: Path,
) -> pd.DataFrame:
    """Test and plot Pearson correlations between an index and its inputs."""
    correlations: list[dict[str, float | int | str]] = []
    for column in variables.columns:
        paired = pd.concat(
            [index.rename(index_name), variables[column]], axis=1
        ).dropna()
        if len(paired) < 2:
            raise ValueError(
                f"At least two paired observations are required for '{column}'."
            )
        if paired[index_name].nunique() < 2 or paired[column].nunique() < 2:
            raise ValueError(
                f"Pearson correlation is undefined for constant data in '{column}'."
            )

        result = pearsonr(paired[index_name], paired[column])
        correlations.append(
            {
                "variable": column,
                "correlation": float(result.statistic), # type: ignore
                "p_value": float(result.pvalue), # type: ignore
                "observations": len(paired),
            }
        )

    results = pd.DataFrame(correlations).sort_values(
        "correlation", key=lambda values: values.abs(), ascending=False
    )
    print(f"\nPearson correlation tests: {index_name} vs PCA variables")
    print(
        results.to_string(
            index=False,
            formatters={
                "correlation": "{:.4f}".format,
                "p_value": "{:.4g}".format,
            },
        )
    )

    heatmap_data = results.set_index("variable")[["correlation"]].T
    figure, axis = plt.subplots(figsize=(max(14, len(results) * 0.55), 4))
    sns.heatmap(
        heatmap_data,
        ax=axis,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": f"Pearson correlation with {index_name}"},
    )
    axis.set_title(f"Correlation of {index_name} with PCA Variables")
    axis.set_xlabel("PCA variable")
    axis.set_ylabel("")
    axis.tick_params(axis="x", labelrotation=65)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"\nSaved correlation map to {output_path}")
    return results


def test_index_correlations(
    sentiment_index: pd.Series,
    variables: pd.DataFrame,
    output_path: Path = IS_CORRELATION_MAP_PATH,
) -> pd.DataFrame:
    """Run Pearson correlation tests against IS and save a correlation map."""
    return _test_index_correlations(
        sentiment_index, variables, "IS", output_path
    )


def test_sis_correlations(
    sis: pd.Series,
    variables: pd.DataFrame,
    output_path: Path = SIS_CORRELATION_MAP_PATH,
) -> pd.DataFrame:
    """Test Pearson correlations of SIS against each PCA input."""
    return _test_index_correlations(sis, variables, "SIS", output_path)


def run_pca(
    input_path: Path = INPUT_PATH,
    index_output_path: Path = IS_INDEX_OUTPUT_PATH,
    correlation_map_path: Path = IS_CORRELATION_MAP_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.Series]:
    """Build the IS index, test correlations, and display KMO/PCA results."""
    data = pd.read_csv(input_path)
    if "date" not in data.columns:
        raise ValueError("Input data must contain a 'date' column.")

    return_variables = [
        column for column in data.columns if column.startswith("return_")
    ]
    variables = list(dict.fromkeys((*BASE_VARIABLES, *return_variables)))
    missing_variables = [column for column in variables if column not in data.columns]
    if missing_variables:
        raise ValueError(f"Input data is missing required variables: {missing_variables}")

    selected = data.loc[:, ["date", *variables]].copy()
    selected["date"] = pd.to_datetime(selected["date"], errors="raise")
    selected = selected.sort_values("date", kind="stable").reset_index(drop=True)
    for column in variables:
        selected[f"{column}_lag1"] = selected[column].shift(1)

    analysis = selected.drop(columns="date").dropna()
    if len(analysis) < 2:
        raise ValueError("At least two complete rows are required to run PCA and KMO.")
    non_numeric = analysis.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise TypeError(f"PCA variables must be numeric: {non_numeric}")

    overall_kmo, variable_kmo = calculate_kmo(analysis)
    pca = PCA(n_components=1)
    component_scores = pca.fit_transform(analysis)[:, 0]
    sentiment_index = pd.Series(
        component_scores,
        index=analysis.index,
        name="IS",
    )
    loadings = pd.Series(
        pca.components_[0],
        index=analysis.columns,
        name="PC1 loading",
    ).sort_values(key=np.abs, ascending=False)

    sentiment_data = pd.DataFrame(
        {
            "date": selected.loc[analysis.index, "date"].to_numpy(),
            "IS": sentiment_index.to_numpy(),
        }
    )
    index_output_path.parent.mkdir(parents=True, exist_ok=True)
    sentiment_data.to_csv(index_output_path, index=False)

    correlation_results = test_index_correlations(
        sentiment_index,
        analysis,
        correlation_map_path,
    )

    print(f"Rows analyzed: {len(analysis):,}")
    print(f"Variables analyzed (including 1-day lags): {analysis.shape[1]}")
    print(f"Overall KMO: {overall_kmo:.6f}")
    print("\nKMO by variable:")
    print(variable_kmo.sort_values(ascending=False).to_string(float_format="%.4f"))
    print(f"\nPC1 explained variance: {pca.explained_variance_ratio_[0]:.6f}")
    print("\nPC1 loadings (sorted by absolute magnitude):")
    print(loadings.to_string(float_format="%.6f"))
    print(f"\nSaved sentiment index to {index_output_path}")

    return sentiment_data, correlation_results, overall_kmo, loadings


def _source_column(variable: str) -> str:
    """Return the source column used to create a one-day lag variable."""
    return variable.removesuffix("_lag1") if variable.endswith("_lag1") else variable


def test_is_sis_correlation(
    is_path: Path = IS_INDEX_OUTPUT_PATH,
    sis_path: Path = SIS_INDEX_OUTPUT_PATH,
) -> tuple[float, float, int]:
    """Run and display a date-aligned Pearson correlation test for IS and SIS."""
    is_data = pd.read_csv(is_path)
    sis_data = pd.read_csv(sis_path)
    if "date" not in is_data.columns or "IS" not in is_data.columns:
        raise ValueError(f"{is_path} must contain 'date' and 'IS' columns.")
    if "date" not in sis_data.columns or "SIS" not in sis_data.columns:
        raise ValueError(f"{sis_path} must contain 'date' and 'SIS' columns.")

    is_data = is_data.loc[:, ["date", "IS"]].copy()
    sis_data = sis_data.loc[:, ["date", "SIS"]].copy()
    is_data["date"] = pd.to_datetime(is_data["date"], errors="raise")
    sis_data["date"] = pd.to_datetime(sis_data["date"], errors="raise")
    if is_data["date"].duplicated().any() or sis_data["date"].duplicated().any():
        raise ValueError("IS and SIS index files must have unique dates.")

    paired = is_data.merge(sis_data, on="date", how="inner").dropna(
        subset=["IS", "SIS"]
    )
    if len(paired) < 2:
        raise ValueError("At least two dates shared by IS and SIS are required.")
    if paired["IS"].nunique() < 2 or paired["SIS"].nunique() < 2:
        raise ValueError("Pearson correlation is undefined for a constant index.")

    result = pearsonr(paired["IS"], paired["SIS"])
    correlation = float(result.statistic) # type: ignore
    p_value = float(result.pvalue) # type: ignore
    observations = len(paired)
    print("\nPearson correlation test: IS vs SIS")
    print(f"Correlation: {correlation:.6f}")
    print(f"p-value: {p_value:.6g}")
    print(f"Paired observations: {observations:,}")
    return correlation, p_value, observations


def run_second_pca(
    input_path: Path = INPUT_PATH,
    index_output_path: Path = SIS_INDEX_OUTPUT_PATH,
    correlation_map_path: Path = SIS_CORRELATION_MAP_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.Series]:
    """Extract SIS with first-component PCA, test KMO and correlations."""
    data = pd.read_csv(input_path)
    if "date" not in data.columns:
        raise ValueError("Input data must contain a 'date' column.")

    source_columns = list(dict.fromkeys(_source_column(name) for name in PCA_VARIABLES))
    missing_columns = sorted(set(source_columns).difference(data.columns))
    if missing_columns:
        raise ValueError(f"Input data is missing required variables: {missing_columns}")

    selected = data.loc[:, ["date", *source_columns]].copy()
    selected["date"] = pd.to_datetime(selected["date"], errors="raise")
    selected = selected.sort_values("date", kind="stable").reset_index(drop=True)

    for variable in PCA_VARIABLES:
        if variable.endswith("_lag1"):
            source = _source_column(variable)
            selected[variable] = selected[source].shift(1)

    analysis = selected.loc[:, list(PCA_VARIABLES)].dropna()
    if len(analysis) < 2:
        raise ValueError("At least two complete rows are required to run PCA and KMO.")
    non_numeric = analysis.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise TypeError(f"PCA variables must be numeric: {non_numeric}")

    overall_kmo, variable_kmo = calculate_kmo(analysis)
    pca = PCA(n_components=1)
    scores = pca.fit_transform(analysis)[:, 0]
    sis = pd.Series(scores, index=analysis.index, name="SIS")

    index_data = pd.DataFrame(
        {
            "date": selected.loc[analysis.index, "date"].to_numpy(),
            "SIS": sis.to_numpy(),
        }
    )
    index_output_path.parent.mkdir(parents=True, exist_ok=True)
    index_data.to_csv(index_output_path, index=False)

    correlations = test_sis_correlations(
        sis,
        analysis,
        output_path=correlation_map_path,
    )

    print(f"\nRows analyzed: {len(analysis):,}")
    print(f"Variables analyzed: {analysis.shape[1]}")
    print(f"Overall KMO: {overall_kmo:.6f}")
    print("\nKMO by variable:")
    print(variable_kmo.sort_values(ascending=False).to_string(float_format="%.4f"))
    print(f"\nFirst component explained variance: {pca.explained_variance_ratio_[0]:.6f}")
    print(f"Saved SIS index to {index_output_path}")

    return index_data, correlations, overall_kmo, variable_kmo


if __name__ == "__main__":
    run_pca()
    run_second_pca()
    test_is_sis_correlation()
