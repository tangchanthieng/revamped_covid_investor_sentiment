# Python analysis scripts

These scripts prepare the project's daily COVID-19 and HOSE market data,
construct sentiment measures, and estimate volatility and ARDL models.

## Run the pipeline

From the repository root, install dependencies and run the scripts in order:

```powershell
python -m pip install -r requirements.txt
python python/joins.py
python python/clean.py
python python/standardization.py
python python/pca.py
python python/model_benchmark.py
python python/GARCH.py
python python/ARDL.py
```

`model_benchmark.py` creates `data/cleaned/data.csv`, which is used by the
GARCH/TARCH step. It also runs optional OLS benchmark diagnostics.

## Script reference

- `config.py` defines project-relative paths for data, assets, and notebooks.
- `joins.py` combines the raw COVID-19 and market datasets.
- `clean.py` prepares the merged data and handles data cleaning and imputation.
- `standardization.py` creates standardized model inputs.
- `pca.py` constructs the Investor Sentiment Index (IS) and Sentiment Index
  (SIS) outputs.
- `model_benchmark.py` builds the benchmark modeling dataset and runs OLS
  diagnostics.
- `GARCH.py` estimates GARCH/TGARCH models and market volatility.
- `ARDL.py` estimates autoregressive distributed lag models.
- `utils.py` contains shared utilities.

Generated datasets are written under `data/`; charts and diagnostic
visualizations are written under `assets/`.
