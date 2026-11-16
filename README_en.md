# Corporate ROA forecasting in Kazakhstan

Research internship materials for Arsen Gabidenov's master's thesis at Astana IT University: “Development of an intelligent system for analyzing and forecasting financial and economic indicators of enterprises”.

[Full Russian guide](README.md) contains ten weekly topic folders, actual completion status and execution commands. The repository includes financial data, separate macroeconomic series, Python modelling code, frozen predictions, diagnostics and the internship report.

Annual archive: 196 firm-year records from 15 firms; the primary eligible sample has 165 records from 14 firms and 151 lagged-feature rows. Primary evaluation: 88 cases, 2018–2024. Last-ROA MAE: 4.424 percentage points; original internal GB: 4.515. An exploratory post-diagnostic GB with absolute-error loss reached 4.201.

Quarterly pilot: 72 records, three firms, 2020Q1–2025Q4. Primary test: 24 cases in 2024–2025 with an assumed two-quarter information delay. Seasonal baseline MAE: 0.653 pp; internal GB MAE: 0.505; CatBoost MAE with macro features: 0.596. Annual and quarterly error scales are not directly comparable.

Use Python 3.12 and `pip install -r requirements.txt`. `python reproduce.py` reaggregates metrics; `python reproduce.py --fit` refits recorded choices; `python reproduce.py --nested` repeats annual nested selection. `python reproduce_bootstrap.py` reproduces descriptive paired comparisons. New outputs go to `recomputed/`.

This is retrospective exploratory work, not untouched-holdout or real-time validation. Actual historical report publication dates are not fully reconstructed. No LSTM/GRU, SHAP, calibrated future prediction intervals or deployed decision system is claimed. [Reproducibility notes](docs/REPRODUCIBILITY.md), [data provenance](data/README.md), [rights notice](LICENSING.md).
