from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

from config import ASSETS_DIR, MART_DATA_DIR

INPUT_PATH = MART_DATA_DIR / "standardized_data.csv"
INDEX_OUTPUT_PATH = MART_DATA_DIR / "sentiment_index.csv"
CORRELATION_MAP_PATH = ASSETS_DIR / "IS_correlation_map.png"

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


def calculate_kmo(
    data: pd.DataFrame,
) -> tuple[float, pd.Series]:
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
    overall_kmo = float(correlation_squared.sum() / (
        correlation_squared.sum() + partial_squared.sum()
    ))
    return overall_kmo, variable_kmo


def test_index_correlations(
    sentiment_index: pd.Series,
    variables: pd.DataFrame,
    output_path: Path = CORRELATION_MAP_PATH,
) -> pd.DataFrame:
    """Run Pearson correlation tests against IS and display/save a correlation map."""
    correlations: list[dict[str, float | int | str]] = []
    for column in variables.columns:
        paired = pd.concat(
            [sentiment_index.rename("IS"), variables[column]], axis=1
        ).dropna()
        if len(paired) < 2:
            raise ValueError(
                f"At least two paired observations are required for '{column}'."
            )
        if paired["IS"].nunique() < 2 or paired[column].nunique() < 2:
            raise ValueError(
                f"Pearson correlation is undefined for constant data in '{column}'."
            )

        result = pearsonr(paired["IS"], paired[column])
        correlations.append(
            {
                "variable": column,
                "correlation": float(result.statistic),
                "p_value": float(result.pvalue),
                "observations": len(paired),
            }
        )

    results = pd.DataFrame(correlations).sort_values(
        "correlation", key=lambda values: values.abs(), ascending=False
    )
    print("\nPearson correlation tests: IS vs PCA variables")
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
    figure_width = max(14, len(results) * 0.48)
    figure, axis = plt.subplots(figsize=(figure_width, 4))
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
        cbar_kws={"label": "Pearson correlation with IS"},
    )
    axis.set_title("Correlation of IS with PCA Variables")
    axis.set_xlabel("PCA variable")
    axis.set_ylabel("")
    axis.tick_params(axis="x", labelrotation=65)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(figure)
    print(f"\nSaved correlation map to {output_path}")
    return results


def run_pca(
    input_path: Path = INPUT_PATH,
    index_output_path: Path = INDEX_OUTPUT_PATH,
    correlation_map_path: Path = CORRELATION_MAP_PATH,
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


if __name__ == "__main__":
    run_pca()