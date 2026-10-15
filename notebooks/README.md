# Notebooks

This folder contains Jupyter notebooks for exploratory analysis and review of
the project's COVID-19 and HOSE market datasets.

## `data_assessment.ipynb`

Loads the raw, merged, and cleaned CSV files from `../data/` for data
assessment and visualization. Run it from this folder so its relative paths
resolve as written. The notebook reads existing project data; run the
preparation scripts in `python/` first if those files need to be regenerated.

Install the project dependencies from the repository root with:

```powershell
python -m pip install -r requirements.txt
```
