### Investor Sentiment and Vietnam Stock Market

Update on Oct 2026

### Project Summary

This project revamps and expands early 2021 research on the impact of COVID-19 on the Ho Chi Minh City Stock Exchange (HOSE). While initial findings confirmed that COVID-19 affected stock returns before fading after the second pandemic phase (COVID phases were officially announced by the Vietnam Ministry of Health), this updated project examines whether positive and negative pandemic shocks indirectly drove market movements through investor sentiment and, therefore, reveals how HOSE is remarkably affected by investor sentimental behaviors. At the end of this project, there are three major takeaways:

- Sentiment Index Construction: Employs first- and second-stage Principal Component Analysis (PCA) to select the most suitable financial proxies for an empirical Investor Sentiment Index.

- Volatility Modeling (GARCH / TGARCH): Examines conditional variance to capture market volatility and assess how pandemic shocks influenced investor behavior.

- Co-integration Analysis (ARDL): Utilizes an Autoregressive Distributed Lag model to evaluate the short- and long-term relationships between investor sentiment, COVID-19 shocks, and market volatility.

### Business Challenge

``Problem definition``

Developed from the previous project, the main challenge faced by investors on HOSE, especially “new investors”, before and during the pandemic was driven by both the market and other organizational investors. Many papers have revealed the nexus between investor behaviors and the sentiment-related movements, though the results varied depending on periods and events, there are some common findings:

- Financial markets experienced fast-moving changes in liquidity, volatility, and risk appetite as COVID-19 evolved across waves and policy interventions.
- Investors and portfolio managers needed a structured way to determine whether pandemic indicators and sentiment shifts were signaling a structural change in risk rather than temporary noise.
- The main challenge was to translate heterogeneous raw data into a concise but actionable market signal that could be linked to pandemic phases and conditional volatility.
- Traditional return-based analysis alone was insufficient because market stress was driven by both macro epidemiological shocks and investor behavior, including sentiment, breadth, and momentum shifts.

This project, on the other hand, addresses only one challenge: does it matter if market variances driven by unanticipated shocks persist in short-term/long-term periods? If not, how should investors react based on the market volatilities?

### Business Questions

``Core questions``

- Which market factors are most strongly associated with investor sentiment during the pandemic?

- Did changes in COVID-19 case and death counts influence risk perception, market breadth, and sector-specific performance?

- Can a PCA-based sentiment index explain changes in market variance and volatility clustering?

- Which pandemic phases or macro conditions correspond to elevated market stress and instability?

- How can sentiment and macroeconomic indicators be used to improve interpretation of period-specific market behavior?

### Project Objectives

``Goals``

- Build a clean, merged COVID-market dataset with daily observations and consistent time alignment.
- Derive an investor sentiment index from multiple market factors, including liquidity, momentum, valuation, breadth, and return-related signals.
- Evaluate how the sentiment series correlates with underlying market variables and whether it captures shared risk sentiment.
- Model conditional volatility using GARCH/TARCH specifications to quantify market variance under pandemic stress.
- Estimate an ARDL framework to assess whether sentiment, macro variables, and COVID conditions help explain market variance dynamics.
- Deliver a reproducible research-and-analysis pipeline for supporting investment and macro-risk decisions.

### Tech Stacks

``Tools and libraries``

This project is built entirely on Python with the following libraries:

- pandas for data cleaning, merging, and time-series preparation
- numpy for numerical transformations and PCA support
- scikit-learn for principal component analysis and sentiment extraction
- scipy for statistical testing, including correlation analysis
- statsmodels for ARDL modeling, residual diagnostics, and stability tests
- arch for GARCH/TARCH conditional volatility estimation
- matplotlib and seaborn for charting and correlation visualizations
- Jupyter notebooks for exploratory analysis and model review

### Scope

``Boundaries``

- The analysis focuses on daily market and COVID data merged on date, with emphasis on pandemic-era risk conditions and sentiment shifts.
- The project includes sector return variables, market breadth, liquidity, momentum, volatility, and pandemic phase dummies derived from COVID case/death series.
- It uses a PCA-driven sentiment index as the primary representation of investor mood and market risk appetite.
- The study is designed for financial and macro-risk interpretation rather than for full-scale production deployment or live forecasting systems.
- The modeling scope is limited to historical panel data and reproducible offline analysis; it does not include real-time streaming or automated trading decisions.

### Key Findings and Insights

``Key findings``

- PCA first principal component proved to be a suitable test for capturing investor sentiment, though it must be carefully researched to include any factors to feed into the model.
- PCA second attempt was to eliminate unnecessary factors between the current day (t) and its previous day (t-1).
- GARCH results revealed that covid phases significantly affected the market variances while volatilties/shocks were not empirically persisted and statistically significant. TGARCH, on the other hand, results in the effects of both covid phases and cases on the market volatilities. Additionally, TGARCH also captured the asymmetric impacts of covid.
- Negative shocks increase volatility more strongly than positive shocks of comparable magnitude. This supports the result that death cases were statistically significant in TGARCH model while that of GARCH was not. The volatilities, in this case, were significant and thus persisted.
- By comparing the results from two models, the question of whether covid played a role in driving market variances have been answered. However, covid alone is not the whole explanation of the market drivers, considering indicators such as liquidity, returns of finance sector, returns of real estate sector, and other control factors like oil, gold, exchange rates, etc.
- Investor sentiment (current day - t) in both models was economically significant thus, suggesting an answer to the question of sentimental behaviors.
- Regaring the ARDL model results, investor sentiment, again, maintained the impact on the market variance, yet covid-related factors showed no statistic effects. This revealed that in short-term/long-term covid was not a major factor bearishing the market. In fact, sentiment was.

``Recommendation``

Use your PCA-based investor sentiment index to identify unusually optimistic or pessimistic market conditions. Combine it with market returns, trading volume, liquidity and other indicators.
- When sentiment deteriorates sharply, reassess portfolio risk and exposure.
- When sentiment becomes extremely optimistic, examine whether prices are running ahead of fundamentals.
- Avoid buying or selling solely because the sentiment index changes direction.

Use GARCH or TGARCH forecasts to monitor conditional market volatility and adjust position sizes accordingly.
- Reduce excessive concentration or leverage when forecast volatility rises.
- Maintain adequate liquidity to meet short-term obligations.
- Set portfolio risk limits before market stress occurs, rather than improvising during a sell-off.

Track whether pandemic-related developments actually affect economic activity, corporate earnings, interest rates, supply chains and investor expectations.
- Treat case counts and death counts as contextual information, not automatic trading signals.
- Examine whether new information changes expected cash flows or the risk premium investors demand.
- Reassess the portfolio when the economic implications change, even if COVID indicators remain stable.

### Project Structure

- `data/raw/` contains the original market and COVID datasets.
- `data/merged/` stores intermediate merged data files.
- `data/cleaned/` stores preprocessed and model-ready datasets, including sentiment and standardized series.
- `python/` contains the analysis scripts for joins, cleaning, standardization, PCA, GARCH/TGARCH, and ARDL modeling.
- `assets/` stores plots and diagnostic visualizations generated during the workflow.
- `notebooks/` contains exploratory analysis notebooks.

### How to Run This Project

``Setup and execution``

- Open a terminal in the project root and create a virtual environment if needed:
  - `python -m venv .venv`
  - `./.venv/Scripts/Activate.ps1`  (Windows PowerShell)
- Install the project dependencies:
  - `python -m pip install --upgrade pip`
  - `python -m pip install -r requirements.txt`
- Run the data preparation and analytics pipeline in sequence:
  - `python python/joins.py`
  - `python python/clean.py`
  - `python python/standardization.py`
  - `python python/pca.py`
  - `python python/GARCH.py`
  - `python python/ARDL.py`
  - `python python/model_benchmark.py`  (optional benchmark OLS + diagnostics run)
- The output files are generated under the `data/cleaned/`, `data/merged/`, and `assets/` directories, including:
  - `data/merged/first_merge.csv`
  - `data/merged/second_merge.csv`
  - `data/merged/third_merge.csv`
  - `data/cleaned/cleaned_data.csv`
  - `data/cleaned/standardized_data.csv`
  - `data/cleaned/IS.csv`
  - `data/cleaned/SIS.csv`
  - `data/cleaned/data.csv`  (benchmark dataset used by `model_benchmark.py`)
  - `assets/IS_correlation_map.png`
  - `assets/SIS_correlation_map.png`
  - `assets/CUSUM_CUSMSQ.png`
- If you want to reproduce the workflow from a clean state, rerun the preprocessing scripts before modeling and review the generated CSV outputs for validation.
