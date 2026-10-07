from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from config import ASSETS_DIR, MART_DATA_DIR

INPUT_PATH = MART_DATA_DIR / "standardized_data.csv"
INDEX_OUTPUT_PATH = MART_DATA_DIR / "sentiment_index_second_run.csv"
CORRELATION_MAP_PATH = ASSETS_DIR / "SIS_correlation_map.png"
IS_INDEX_PATH = MART_DATA_DIR / "sentiment_index.csv"

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


def _source_column(variable: str) -> str:
    """Return the source column used to create a one-day lag variable."""
    return variable.removesuffix("_lag1") if variable.endswith("_lag1") else variable


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
    denominators = correlation_sums + partial_sums
    if np.any(denominators == 0):
        invalid = data.columns[denominators == 0].tolist()
        raise ValueError(f"KMO is undefined for variables with no correlations: {invalid}")

    variable_kmo = pd.Series(
        correlation_sums / denominators,
        index=data.columns,
        name="kmo",
    )
    overall_kmo = float(
        correlation_squared.sum()
        / (correlation_squared.sum() + partial_squared.sum())
    )
    return overall_kmo, variable_kmo


def test_sis_correlations(
    sis: pd.Series,
    variables: pd.DataFrame,
    output_path: Path = CORRELATION_MAP_PATH,
) -> pd.DataFrame:
    """Test Pearson correlations of SIS against each PCA input and save a map."""
    results: list[dict[str, float | int | str]] = []
    for column in variables.columns:
        paired = pd.concat([sis.rename("SIS"), variables[column]], axis=1).dropna()
        if len(paired) < 2:
            raise ValueError(
                f"At least two paired observations are required for '{column}'."
            )
        if paired["SIS"].nunique() < 2 or paired[column].nunique() < 2:
            raise ValueError(
                f"Pearson correlation is undefined for constant data in '{column}'."
            )

        result = pearsonr(paired["SIS"], paired[column])
        results.append(
            {
                "variable": column,
                "correlation": float(result.statistic),
                "p_value": float(result.pvalue),
                "observations": len(paired),
            }
        )

    correlations = pd.DataFrame(results).sort_values(
        "correlation", key=lambda values: values.abs(), ascending=False
    )
    print("\nPearson correlation tests: SIS vs PCA variables")
    print(
        correlations.to_string(
            index=False,
            formatters={
                "correlation": "{:.4f}".format,
                "p_value": "{:.4g}".format,
            },
        )
    )

    heatmap_data = correlations.set_index("variable")[["correlation"]].T
    figure, axis = plt.subplots(figsize=(max(12, len(correlations) * 0.55), 4))
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
        cbar_kws={"label": "Pearson correlation with SIS"},
    )
    axis.set_title("Correlation of SIS with PCA Variables")
    axis.set_xlabel("PCA variable")
    axis.set_ylabel("")
    axis.tick_params(axis="x", labelrotation=65)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    print(f"\nSaved correlation map to {output_path}")
    return correlations


def test_is_sis_correlation(
    is_path: Path = IS_INDEX_PATH,
    sis_path: Path = INDEX_OUTPUT_PATH,
) -> tuple[float, float, int]:
    """Run and display a date-aligned Pearson correlation test for IS and SIS."""
    is_data = pd.read_csv(is_path)
    sis_data = pd.read_csv(sis_path)
    required_columns = {"date"}
    if not required_columns.issubset(is_data.columns) or "IS" not in is_data.columns:
        raise ValueError(f"{is_path} must contain 'date' and 'IS' columns.")
    if not required_columns.issubset(sis_data.columns) or "SIS" not in sis_data.columns:
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
    correlation = float(result.statistic)
    p_value = float(result.pvalue)
    observations = len(paired)
    print("\nPearson correlation test: IS vs SIS")
    print(f"Correlation: {correlation:.6f}")
    print(f"p-value: {p_value:.6g}")
    print(f"Paired observations: {observations:,}")
    return correlation, p_value, observations


def run_second_pca(
    input_path: Path = INPUT_PATH,
    index_output_path: Path = INDEX_OUTPUT_PATH,
    correlation_map_path: Path = CORRELATION_MAP_PATH,
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
    run_second_pca()
    test_is_sis_correlation()