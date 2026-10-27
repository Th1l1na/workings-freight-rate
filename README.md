# Freight Rate Prediction

Predicts `posted_rate` for truckload freight. Trained on 1 Jan – 31 Oct 2025 (`data/train_test.csv`),
predicts 1 Nov – 31 Dec 2025 (`data/validation.csv`) and the fixed December lane chart.

**Model:** two-stage. A Ridge regression on `log(rate)` carries the drivers that must extrapolate into
Nov–Dec (distance, equipment, weight, market index, quarter-end ramp, linear trend). An equal-weight
average of LightGBM, XGBoost and Extra Trees, trained on the Ridge residual, adds lane geography and
non-linear effects without using absolute time or city names.

**Validation:** rolling-origin time-series CV. Three folds, each training on all earlier months and
scoring the next two (Jan–Apr→May–Jun, Jan–Jun→Jul–Aug, Jan–Aug→Sep–Oct). Anomaly detection is
re-fitted inside each fold on training months only.

| | MAPE (mean of folds, normal loads) | MAPE incl. anomalous labels |
|---|---|---|
| **Chosen: Ridge + (LightGBM, XGBoost, Extra Trees)** | **1.60%** | 3.98% |
| Ridge + LightGBM | 1.67% | 4.04% |
| LightGBM alone | 2.71% | 5.00% |
| Ridge alone | 2.83% | 5.19% |
| Median $/mile baseline | 4.64% | 6.92% |

21 candidates were compared (`outputs/model_comparison.csv`). The ensemble's gain over Ridge +
LightGBM is significant over all folds (bootstrap over dates) and no worse on the most recent fold.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt

python3 -m src.train          # CV of the main candidates, final fit, both prediction files (~2 min)
python3 score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Full analysis (charts, comparison, checks, report):

```bash
python3 -m src.compare        # 21-model comparison on the same folds (~5 min) + charts
python3 -m src.eda            # EDA charts
python3 -m src.checks         # robustness checks (~10 min); `python -m src.checks 2 6` runs selected ones
python3 -m src.diagnostics    # diagnostics of the chosen model
python3 make_report.py        # report/Freight_Rate_Report.docx
python3 make_figure_book.py   # report/Figure_Book.html (+ .pdf if Playwright is installed)
```

`python -m src.train --skip-cv` refits and predicts without re-running cross-validation.

## Outputs

| File | What it is |
|---|---|
| `validation_predictions.csv` | `load_id,predicted_rate` for all 12,000 validation loads |
| `data/december_chart_inputs.csv` | the fixed December lane with `predicted_rate` filled |
| `scorer_results/candidate_december.png` | chart produced by the provided `score.py` |
| `outputs/cv_results.csv`, `cv_summary.csv` | per-fold and overall metrics: periods, n, MAE, RMSE, MAPE, bias |
| `outputs/cv_predictions.csv.gz` | every out-of-fold prediction |
| `outputs/model_comparison*.csv` | 21-candidate comparison: summary, by fold, out-of-fold predictions |
| `outputs/model_bootstrap_vs_lightgbm_hybrid.csv` | bootstrap confidence intervals for model differences |
| `outputs/figures/{eda,models,checks,diagnostics}/` | every chart, with `captions.json` (shows / finding / decision) |
| `outputs/figures/checks/checks_summary.json` | numeric results of the robustness checks |
| `outputs/summary.json` | data-quality counts, Ridge coefficients, prediction stats |
| `outputs/december_inputs_used.csv` | market inputs used for the December rows |
| `report/Freight_Rate_Report.docx` / `.pdf` | written report |
| `report/Figure_Book.html` / `.pdf` | all charts with explanations |

## Repository layout

```
src/features.py     cleaning + feature engineering (shared by train and inference)
src/model.py        anomaly detector, two-stage ensemble model
src/train.py        time-based CV of main candidates, final fit, prediction files
src/compare.py      21-candidate comparison, ensembles, bootstrap significance, charts
src/eda.py          exploratory charts
src/checks.py       robustness checks (leakage, trend, date effect, unseen cities, anomalies, December)
src/diagnostics.py  error analysis and sanity checks of the chosen model
src/plotting.py     chart style + caption registry
make_report.py      DOCX report (all numbers read from outputs/)
make_figure_book.py all charts in one document
score.py            provided scorer (unchanged)
```

## Key decisions

- **Time-based split, not random.** `validation.csv` is strictly later than the training data. A random
  split would leak same-day market conditions and hide the forecasting problem.
- **Hybrid structure.** Tree models cannot extrapolate a trend: past their last split they predict a
  constant. Used directly they under-predict later months. Every algorithm tested was better on top of
  the Ridge base.
- **No city IDs.** 8 validation cities (1,447 loads) never appear in training. Lanes are described by
  coordinates, distance and circuity. The coordinates are synthetic (not real city locations), so they
  are used only as internal location signals.
- **Anomalous labels** (677 rows, 1.4%, at 0.2–0.5× or 2–5× the expected rate) are excluded from
  training. Features are repaired, never dropped: negative weight → absolute value; missing
  `market_index` → that day's median; missing weight → native NaN handling or the training median.
- **December chart inputs.** The chart file has no coordinates, `market_index` or `quote_signal`.
  Coordinates come from the city lookup. Market inputs are the daily medians of the feature values
  supplied in `validation.csv` (no labels used). A training-only alternative gives nearly the same curve
  (`outputs/figures/checks/06_december_scenarios.png`).

Tested with Python 3.12, pandas 2.2 and 3.0, LightGBM 4.x, XGBoost 3.x, scikit-learn 1.4+.
