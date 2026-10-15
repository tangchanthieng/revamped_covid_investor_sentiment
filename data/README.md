# Data

This folder contains the CSV inputs and intermediate/model-ready datasets for
the project's analysis of COVID-19, investor sentiment, and the HOSE market.

## Folders

- `raw/` contains the source inputs: COVID-19 case counts, death counts,
  historical market prices, and market indicators.
- `merged/` contains successive date-aligned intermediate datasets produced
  from the raw inputs (`first_merge.csv`, `second_merge.csv`, and
  `third_merge.csv`).
- `cleaned/` contains cleaned and imputed data, standardized variables,
  sentiment index series, the benchmark dataset, and market variance outputs.

## Workflow

Run the preparation scripts from the repository root in this order:

```powershell
python python/joins.py
python python/clean.py
python python/standardization.py
python python/pca.py
python python/model_benchmark.py
python python/GARCH.py
python python/ARDL.py
```

The scripts write derived files into `merged/` and `cleaned/`. Review generated
outputs before using them in further analysis. Treat files in `raw/` as source
data and preserve them when rerunning the workflow.
