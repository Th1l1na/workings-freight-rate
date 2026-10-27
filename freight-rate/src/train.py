"""End-to-end pipeline: clean -> validate (time-based CV) -> fit -> predict.

Run from the repository root:
    python -m src.train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features, clean, daily_market_index, weight_fill_values
from .model import (GBMOnlyModel, TwoStageRateModel, apply_outlier_detector, fit_outlier_detector,
                    flag_label_outliers)

# Rolling-origin folds. Each mimics the real task: train on everything before a
# cut-off, predict the next two calendar months (validation.csv = Nov-Dec).
# eval_start is inclusive, eval_end is EXCLUSIVE: the evaluation rows are
# eval_start <= date < eval_end, so fold 1 evaluates 1 May - 30 Jun.
FOLDS = [
    # (fold, train period label, eval period label, eval_start, eval_end)
    (1, "Jan-Apr", "May-Jun", "2025-05-01", "2025-07-01"),
    (2, "Jan-Jun", "Jul-Aug", "2025-07-01", "2025-09-01"),
    (3, "Jan-Aug", "Sep-Oct", "2025-09-01", "2025-11-01"),
]
N_UNSEEN_CITIES = 8  # validation.csv contains 8 cities never seen in training
SEED = 42


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    err = p - y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(np.abs(err) / y) * 100),
        "Bias%": float(np.mean(err / y) * 100),
    }


class RatePerMileBaseline:
    """Median rate-per-mile by equipment and distance band."""

    BANDS = [0, 250, 500, 1000, 1500, 2000, 1e9]

    def fit(self, frame, log_rate):
        band = pd.cut(frame["distance"], self.BANDS)
        rpm = np.exp(log_rate) / frame["distance"]
        self.table_ = rpm.groupby([frame["equipment"], band], observed=True).median()
        return self

    def predict(self, frame):
        band = pd.cut(frame["distance"], self.BANDS)
        keys = list(zip(frame["equipment"], band))
        return np.array([self.table_.get(k, self.table_.median()) for k in keys]) * frame["distance"].values


def load(data_dir: Path):
    train = pd.read_csv(data_dir / "train_test.csv")
    valid = pd.read_csv(data_dir / "validation.csv")
    daily_mi = daily_market_index(train, valid)
    train = clean(train, daily_mi)
    valid = clean(valid, daily_mi)
    return train, valid, daily_mi


FINAL_MODEL = "Two-stage ensemble (final)"
MODELS = [
    "Baseline: median $/mile",
    "LightGBM only",
    "Linear only (with trend)",
    "Two-stage, no trend",
    "Two-stage, LightGBM residual only",
    FINAL_MODEL,
]


def fit_predict(name: str, train: pd.DataFrame, feats: pd.DataFrame, log_y: pd.Series, tr, te) -> np.ndarray:
    """Fit one candidate on rows `tr`, return price-scale predictions for rows `te`."""
    if name == "Baseline: median $/mile":
        return RatePerMileBaseline().fit(train[tr], log_y[tr]).predict(train[te])
    if name == "LightGBM only":
        return GBMOnlyModel().fit(feats[tr], log_y[tr]).predict(feats[te])
    if name == "Linear only (with trend)":
        return TwoStageRateModel(use_gbm=False).fit(feats[tr], log_y[tr]).predict(feats[te])
    if name == "Two-stage, no trend":
        f_nt = feats.copy()
        f_nt["trend_years"] = 0.0
        return TwoStageRateModel().fit(f_nt[tr], log_y[tr]).predict(f_nt[te])
    if name == "Two-stage, LightGBM residual only":
        return TwoStageRateModel(residual_models=("lightgbm",)).fit(feats[tr], log_y[tr]).predict(feats[te])
    if name == FINAL_MODEL:
        return TwoStageRateModel().fit(feats[tr], log_y[tr]).predict(feats[te])
    raise ValueError(name)


def run_cv(train: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-fold metrics, every out-of-fold prediction).

    Outlier detection is done INSIDE each fold: the detector is fitted on that
    fold's training months only, then applied unchanged to (a) drop corrupt
    training labels and (b) mark which evaluation labels count as "clean".
    No evaluation-period label is used to fit anything. Every model in a fold
    is trained on the same rows and scored on the same rows.
    All metrics are on the original price scale (predictions are exp()'d).
    """
    y = train["posted_rate"].values
    log_y = np.log(train["posted_rate"])
    rows, preds = [], []
    for fold, train_label, eval_label, start, end in FOLDS:
        train_window = (train["date"] < start).values
        te = ((train["date"] >= start) & (train["date"] < end)).values
        detector = fit_outlier_detector(feats[train_window], log_y[train_window])
        flagged = apply_outlier_detector(detector, feats, log_y).values
        tr = train_window & ~flagged
        clean = ~flagged[te]
        eval_dates = train.loc[te, "date"]
        info = {"fold": fold, "train_period": train_label, "eval_period": eval_label,
                "eval_first_date": eval_dates.min().date().isoformat(),
                "eval_last_date": eval_dates.max().date().isoformat(),
                "train_rows": int(tr.sum()), "train_outliers_removed": int((train_window & flagged).sum())}
        for name in MODELS:
            pred = fit_predict(name, train, feats, log_y, tr, te)
            rows.append({**info, "model": name, "subset": "clean labels", "n": int(clean.sum()),
                         **metrics(y[te][clean], pred[clean])})
            rows.append({**info, "model": name, "subset": "all labels", "n": int(te.sum()),
                         **metrics(y[te], pred)})
            preds.append(pd.DataFrame({"fold": fold, "model": name, "load_id": train.loc[te, "load_id"].values,
                                       "date": eval_dates.dt.date.values, "actual": y[te],
                                       "predicted": pred, "clean_label": clean}))
        print(f"fold {fold} (train {train_label}, eval {eval_label}: {info['eval_first_date']} to "
              f"{info['eval_last_date']}, n={int(te.sum())}) done")

    # Unseen-city stress test on fold 3: remove every load touching 8 random
    # cities from training, then score only clean loads that touch them.
    rng = np.random.default_rng(SEED)
    fold, train_label, eval_label, start, end = FOLDS[-1]
    cities = rng.choice(sorted(train["pickup"].unique()), N_UNSEEN_CITIES, replace=False)
    touches = (train["pickup"].isin(cities) | train["delivery"].isin(cities)).values
    train_window = (train["date"] < start).values & ~touches
    detector = fit_outlier_detector(feats[train_window], log_y[train_window])
    flagged = apply_outlier_detector(detector, feats, log_y).values
    tr = train_window & ~flagged
    te = ((train["date"] >= start) & (train["date"] < end)).values & ~flagged & touches
    model = TwoStageRateModel().fit(feats[tr], log_y[tr])
    rows.append({"fold": fold, "train_period": train_label, "eval_period": eval_label,
                 "eval_first_date": train.loc[te, "date"].min().date().isoformat(),
                 "eval_last_date": train.loc[te, "date"].max().date().isoformat(),
                 "train_rows": int(tr.sum()), "train_outliers_removed": int((train_window & flagged).sum()),
                 "model": FINAL_MODEL, "subset": f"unseen-city loads ({', '.join(cities)})",
                 "n": int(te.sum()), **metrics(y[te], model.predict(feats[te]))})
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True)


def aggregate(cv: pd.DataFrame, cv_preds: pd.DataFrame) -> pd.DataFrame:
    """Overall metrics two ways, so the aggregation method is explicit:
    - fold-mean: unweighted average of the three fold-level values
    - pooled:    one metric over all out-of-fold predictions together
    """
    out = []
    for subset, mask_fn in [("clean labels", lambda d: d["clean_label"]), ("all labels", lambda d: d["clean_label"] | True)]:
        fold_rows = cv[(cv["subset"] == subset)]
        for name in MODELS:
            fr = fold_rows[fold_rows["model"] == name]
            p = cv_preds[(cv_preds["model"] == name)]
            p = p[mask_fn(p)]
            pooled = metrics(p["actual"].values, p["predicted"].values)
            out.append({"model": name, "subset": subset, "n_total": int(fr["n"].sum()),
                        **{f"{k} fold-mean": float(fr[k].mean()) for k in ["MAE", "RMSE", "MAPE", "Bias%"]},
                        **{f"{k} pooled": v for k, v in pooled.items()}})
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--predictions-path", default="validation_predictions.csv",
                        help="where to write load_id,predicted_rate (the assessment expects the repo root)")
    parser.add_argument("--december-path", default="december_predictions.csv",
                        help="where to write the filled December chart file")
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train, valid, daily_mi = load(data_dir)
    weight_fill = weight_fill_values(train)
    feats = build_features(train, weight_fill)
    log_y = np.log(train["posted_rate"])
    train["label_outlier"] = flag_label_outliers(feats, log_y).values
    print(f"Flagged {train['label_outlier'].sum()} label outliers ({train['label_outlier'].mean():.2%})")

    summary = {
        "train_rows": len(train),
        "label_outliers": int(train["label_outlier"].sum()),
        "negative_weights_fixed": int(train["weight_was_negative"].sum()),
        "missing_weight": int(train["weight_missing"].sum()),
        "missing_market_index_imputed": int(train["market_index_missing"].sum()),
    }

    if not args.skip_cv:
        cv, cv_preds = run_cv(train, feats)
        agg = aggregate(cv, cv_preds)
        cv.to_csv(out_dir / "cv_results.csv", index=False)
        agg.to_csv(out_dir / "cv_summary.csv", index=False)
        cv_preds.to_csv(out_dir / "cv_predictions.csv.gz", index=False, float_format="%.4f")
        pd.set_option("display.width", 220)
        cols = ["fold", "train_period", "eval_period", "eval_first_date", "eval_last_date", "n",
                "MAE", "RMSE", "MAPE", "Bias%"]
        print(cv[(cv["subset"] == "clean labels") & (cv["model"] == FINAL_MODEL)][cols].round(2).to_string())
        print(agg[agg["subset"] == "clean labels"].drop(columns="subset").round(3).to_string())
        print(cv[cv["subset"].str.startswith("unseen")][cols].round(2).to_string())

    # Final model: all clean labelled data Jan-Oct 2025.
    keep = ~train["label_outlier"].values
    model = TwoStageRateModel().fit(feats[keep], log_y[keep])
    summary["final_train_rows"] = int(keep.sum())
    summary["linear_coefficients"] = {k: round(float(v), 5) for k, v in model.linear_coefficients().items()}

    # Validation predictions in template order.
    template = pd.read_csv(data_dir / "validation_predictions_template.csv")
    vfeats = build_features(valid, weight_fill)
    valid_pred = pd.Series(model.predict(vfeats), index=valid["load_id"])
    template["predicted_rate"] = template["load_id"].map(valid_pred).round(2)
    assert template["predicted_rate"].notna().all() and len(template) == 12_000
    template.to_csv(args.predictions_path, index=False)
    print(f"Wrote {args.predictions_path}")

    # December chart: fill the fixed-lane rows. The chart file only carries
    # lane, equipment, weight and date, so the remaining model inputs are
    # reconstructed: coordinates from the city lookup, market_index from the
    # observed daily market median, quote_signal from the daily median.
    dec_path = data_dir / "december_chart_inputs.csv"  # provided input: read, never required to be writable
    dec = pd.read_csv(dec_path)
    dec_cols = list(dec.columns)
    both = pd.concat([train, valid], ignore_index=True)
    coords = {}
    for side in ["pickup", "delivery"]:
        c = both.groupby(side)[[f"{side}_lat", f"{side}_lon"]].first()
        c.columns = ["lat", "lon"]
        coords.update(c.to_dict("index"))
    daily_qs = both.groupby("date")["quote_signal"].median()
    dx = dec.copy()
    dx["date"] = pd.to_datetime(dx["date"])
    for side in ["pickup", "delivery"]:
        dx[f"{side}_lat"] = dx[side].map(lambda c: coords[c]["lat"])
        dx[f"{side}_lon"] = dx[side].map(lambda c: coords[c]["lon"])
    dx["market_index"] = dx["date"].map(daily_mi)
    dx["quote_signal"] = dx["date"].map(daily_qs)
    dx = clean(dx, daily_mi)
    dec["predicted_rate"] = np.round(model.predict(build_features(dx, weight_fill)), 2)
    dec[dec_cols].to_csv(args.december_path, index=False)
    print(f"Wrote {args.december_path}")
    # The assessment README asks for the predicted_rate column of the provided
    # file to be filled. Do that too, but never fail the run if it is read-only.
    try:
        dec[dec_cols].to_csv(dec_path, index=False)
        print(f"Filled {dec_path} (as the assessment README asks)")
    except PermissionError:
        print(f"NOTE: {dec_path} is read-only, so it was left unchanged. Score with: "
              f"python score.py --predictions {args.predictions_path} --december-predictions {args.december_path}")

    summary["validation_pred_stats"] = template["predicted_rate"].describe().round(2).to_dict()
    summary["december_pred_range"] = [float(dec["predicted_rate"].min()), float(dec["predicted_rate"].max())]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame({"date": dx["date"], "market_index": dx["market_index"], "quote_signal": dx["quote_signal"],
                  "predicted_rate": dec["predicted_rate"]}).to_csv(out_dir / "december_inputs_used.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
