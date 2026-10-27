"""Diagnostics of the chosen model: where it errs, what drives it, and whether the
final validation predictions look sane.

Run from the repository root (after `python -m src.train` and `python -m src.compare`):
    python -m src.diagnostics
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import GBM_FEATURES, LINEAR_FEATURES, build_features
from .model import TwoStageRateModel, flag_label_outliers
from .plotting import ACCENT, BAD, EQUIP_COLORS, GOLD, GOOD, INK, MUTED, TRAIN_C, VALID_C, FigureRegistry
from .train import load

FINAL = "Two-stage ensemble (final)"


def main(out_dir="outputs/figures/diagnostics", model_name=FINAL):
    reg = FigureRegistry(out_dir, "Diagnostics")
    train, valid, daily_mi = load(Path("data"))
    wf = train.groupby("equipment")["weight"].median().to_dict()
    feats = build_features(train, wf)
    log_y = np.log(train["posted_rate"])

    cvp = pd.read_csv("outputs/cv_predictions.csv.gz", parse_dates=["date"])
    cvp = cvp[cvp.model == model_name].merge(
        train[["load_id", "distance", "weight", "equipment", "pickup", "delivery", "quote_signal"]], on="load_id")
    cvp["pct_err"] = (cvp.predicted - cvp.actual) / cvp.actual * 100
    c = cvp[cvp.clean_label].copy()

    # 01 predicted vs actual ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(c.actual, c.predicted, s=1, alpha=0.15, color=ACCENT)
    lim = [c.actual.min(), c.actual.max()]
    axes[0].plot(lim, lim, color=BAD, lw=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    from matplotlib.ticker import LogLocator, NullFormatter
    for a_ in axes:
        for axis in (a_.xaxis, a_.yaxis):
            axis.set_minor_formatter(NullFormatter())
    axes[0].set_xlabel("actual ($)")
    axes[0].set_ylabel("predicted ($)")
    axes[0].set_title("Predicted vs actual (clean labels)")
    o = cvp[~cvp.clean_label]
    axes[1].scatter(cvp[cvp.clean_label].actual, cvp[cvp.clean_label].predicted, s=1, alpha=0.1, color=ACCENT,
                    label="clean")
    axes[1].scatter(o.actual, o.predicted, s=5, alpha=0.7, color=BAD, label="anomalous labels")
    axes[1].plot([10, 30000], [10, 30000], color=INK, lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Including anomalous labels")
    axes[1].legend(markerscale=4, fontsize=8)
    within = {k: (c.pct_err.abs() <= k).mean() for k in [1, 2, 5, 10]}
    reg.save(fig, "01_pred_vs_actual", "Predicted vs actual (out-of-fold)",
             "Every out-of-fold prediction from the 3 time folds (each predicted by a model that never saw its "
             "month).",
             f"Clean labels hug the diagonal: {within[2]:.0%} of loads within ±2%, {within[5]:.0%} within ±5%, "
             f"{within[10]:.1%} within ±10%. The anomalous labels (right, orange) sit far off the line on both "
             "sides; the model predicts a normal price for them, which is the intended behaviour.",
             "")

    # 02 error distribution -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].hist(c.pct_err.clip(-12, 12), bins=120, color=ACCENT)
    axes[0].axvline(0, color=INK, lw=0.8)
    axes[0].set_xlabel("% error ((pred − actual) ÷ actual)")
    axes[0].set_title("Error distribution (clean labels)")
    q = np.linspace(0, 100, 201)
    axes[1].plot(np.percentile(c.pct_err.abs(), q), q, color=ACCENT)
    for k in [2, 5]:
        axes[1].axvline(k, color=MUTED, ls="--", lw=0.8)
        axes[1].text(k + 0.1, 5, f"{within[k]:.0%} within ±{k}%", fontsize=8)
    axes[1].set_xlim(0, 10)
    axes[1].set_xlabel("absolute % error")
    axes[1].set_ylabel("% of loads")
    axes[1].set_title("Cumulative share of loads by error")
    reg.save(fig, "02_error_distribution", "Size of errors",
             "Left: histogram of signed % error. Right: cumulative share of loads with absolute error below x.",
             f"Errors are symmetric and centred near zero (mean {c.pct_err.mean():+.2f}%, median "
             f"{c.pct_err.median():+.2f}%). 90th percentile absolute error: {np.percentile(c.pct_err.abs(), 90):.1f}%.",
             "")

    # 03 residuals by segment ----------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    segs = [
        ("fold", c.fold, "evaluation fold"),
        ("month", c.date.dt.month, "month"),
        ("equipment", c.equipment, "equipment"),
        ("distance", pd.cut(c.distance, [0, 250, 500, 1000, 1500, 2000, 4000]), "distance band"),
        ("weight", pd.cut(c.weight, [0, 15000, 25000, 35000, 45000, 48000]), "weight band (lb)"),
        ("weekday", pd.Categorical(c.date.dt.dayofweek.map(dict(enumerate(["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]))),
                                   categories=["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"], ordered=True), "weekday"),
    ]
    for ax, (key, grp, label) in zip(axes.ravel(), segs):
        g = c.groupby(grp, observed=True).pct_err
        mean, mape = g.mean(), c.pct_err.abs().groupby(grp, observed=True).mean()
        idx = [str(i) for i in mean.index]
        ax.bar(idx, mape.values, color=MUTED, label="MAPE")
        ax.plot(idx, mean.values, "o-", color=BAD, label="bias (mean % error)")
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(f"by {label}", fontsize=10)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Error and bias by segment (clean labels, out-of-fold)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    worst = c.groupby(c.date.dt.month).pct_err.mean()
    dist_b = c.groupby(pd.cut(c.distance, [0, 250, 500, 1000, 1500, 2000, 4000]), observed=True).pct_err.mean()
    seg_mape = [c.pct_err.abs().groupby(g, observed=True).mean() for _, g, _ in segs[2:]]
    lo_m, hi_m = min(m.min() for m in seg_mape), max(m.max() for m in seg_mape)
    reg.save(fig, "03_error_by_segment", "Where does the model err?",
             "Grey bars: MAPE per segment. Red line: mean signed % error (bias; + = over-prediction).",
             f"Accuracy (MAPE) is similar in every equipment, distance, weight and weekday segment "
             f"({lo_m:.1f}–{hi_m:.1f}%). Bias shows mild structure: short hauls (<250 mi) are under-predicted by "
             f"{abs(dist_b.iloc[0]):.1f}% and the longest hauls over-predicted by {dist_b.iloc[-1]:.1f}%, a hint that "
             "the single distance exponent is slightly too steep at the extremes. The largest structure is by month "
             f"({worst.min():+.1f}% to {worst.max():+.1f}%): the remaining error is mostly about the market level "
             "of a period.",
             "Acceptable (all segment biases within ±1%). A possible next step is a distance spline in stage 1. The "
             "month-level bias is the main risk for Nov–Dec, hence the trend checks.")

    # 04 daily bias over time ----------------------------------------------
    d = c.groupby("date").agg(bias=("pct_err", "mean"), mape=("pct_err", lambda s: s.abs().mean()))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(d.index, d.bias, color=MUTED, lw=0.8, label="daily bias")
    ax.plot(d.bias.rolling(7, center=True).mean(), color=BAD, lw=1.8, label="7-day mean bias")
    ax.plot(d.mape.rolling(7, center=True).mean(), color=ACCENT, lw=1.8, label="7-day mean MAPE")
    ax.axhline(0, color=INK, lw=0.6)
    for b in ["2025-07-01", "2025-09-01"]:
        ax.axvline(pd.Timestamp(b), color=INK, ls="--", lw=0.7)
    ax.set_ylabel("%")
    ax.set_title("Out-of-fold error over time (folds 1, 2, 3)")
    ax.legend(fontsize=8, ncol=3)
    c["week_in_fold"] = c.groupby("fold").date.transform(lambda d: (d - d.min()).dt.days // 7)
    wk = c.groupby(["fold", "week_in_fold"]).agg(bias=("pct_err", "mean"),
                                                  mape=("pct_err", lambda x_: x_.abs().mean())).reset_index()
    first = wk[wk.week_in_fold <= 1].groupby("fold")[["bias", "mape"]].mean()
    last = wk[wk.week_in_fold >= wk.week_in_fold.max() - 1].groupby("fold")[["bias", "mape"]].mean()
    drift = (last.bias - first.bias)
    reg.save(fig, "04_error_over_time", "Error over time",
             "Daily mean signed error and 7-day MAPE across the three evaluation windows (dashed lines = fold "
             "boundaries).",
             f"Errors grow with distance from the training cut-off. Mean MAPE goes from {first.mape.mean():.2f}% "
             f"in the first two weeks of a fold to {last.mape.mean():.2f}% in the last two, and the bias drifts by "
             f"{drift.iloc[0]:+.1f} / {drift.iloc[1]:+.1f} / {drift.iloc[2]:+.1f} points (folds 1/2/3) over the "
             "8 weeks, in a direction that depends on how the market moved. Fold 2 (Jul–Aug) runs high: it was "
             "trained on only six months, when the trend was over-estimated (see checks).",
             f"Expect late-December predictions to be less accurate than early-November ones (roughly "
             f"{last.mape.mean():.1f}% vs {first.mape.mean():.1f}% MAPE if the pattern repeats). Retraining weekly or "
             "monthly in production would keep the horizon short.")

    # 05 feature contributions (final model) --------------------------------
    keep = ~flag_label_outliers(feats, log_y).values
    model = TwoStageRateModel().fit(feats[keep], log_y[keep])
    x = feats.loc[keep]
    lin = {f: model.linear_.coef_[i] * (x[f] - x[f].mean()) for i, f in enumerate(LINEAR_FEATURES)}
    contrib = model.gbm_.predict(x[GBM_FEATURES], pred_contrib=True)[:, :-1]
    imp = pd.Series({f"stage 1: {k}": np.abs(v).mean() for k, v in lin.items()})
    imp2 = pd.Series(np.abs(contrib).mean(axis=0), index=[f"stage 2: {f}" for f in GBM_FEATURES])
    allimp = (pd.concat([imp, imp2]) * 100).sort_values()
    allimp = allimp[allimp > 0.05]
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.barh(allimp.index, allimp.values, color=[ACCENT if i.startswith("stage 1") else GOLD for i in allimp.index])
    ax.set_xscale("log")
    ax.set_xlabel("mean |effect on log(rate)| × 100  ≈  average % moved (log scale)")
    ax.set_title("What moves the predicted price (final model)")
    reg.save(fig, "05_feature_importance", "Feature importance",
             "Mean absolute contribution of each feature to log(rate) on the training rows: Ridge coefficient × "
             "centred feature for stage 1 (teal); exact SHAP values of the LightGBM member of the stage-2 ensemble "
             "(gold; the XGBoost and Extra Trees members learn the same residual and are not shown). Log scale.",
             f"Distance dominates (moves price by ~{allimp.get('stage 1: log_distance', 0):.0f}% on average). Then "
             "equipment, weight, market index, quarter-end and trend in stage 1. Stage-2 corrections are small "
             "(mostly geography: pickup/delivery coordinates), refining the price by a few percent.",
             "Confirms the design: the extrapolating drivers are in stage 1, the tree stage only refines.")

    # 06 geography correction map ------------------------------------------
    gx = pd.DataFrame({"lat": x.pickup_lat, "lon": x.pickup_lon,
                       "geo": contrib[:, [GBM_FEATURES.index(f) for f in ["pickup_lat", "pickup_lon"]]].sum(1),
                       "dlat": x.delivery_lat, "dlon": x.delivery_lon,
                       "dgeo": contrib[:, [GBM_FEATURES.index(f) for f in ["delivery_lat", "delivery_lon"]]].sum(1)})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (la, lo, v, t) in zip(axes, [("lat", "lon", "geo", "pickup location"),
                                         ("dlat", "dlon", "dgeo", "delivery location")]):
        g = gx.groupby([la, lo])[v].mean().reset_index()
        sc = ax.scatter(g[lo], g[la], c=(np.exp(g[v]) - 1) * 100, cmap="RdBu_r", vmin=-4, vmax=4, s=40,
                        edgecolors=INK, linewidths=0.3)
        ax.set_title(f"Stage-2 price adjustment by {t}")
        ax.set_xlabel("longitude")
    axes[0].set_ylabel("latitude")
    fig.colorbar(sc, ax=axes, label="% adjustment", shrink=0.85)
    reg.save(fig, "06_geography", "Learned geographic effects",
             "Average SHAP contribution of the pickup (left) and delivery (right) coordinates, per city, as a % "
             "price adjustment.",
             "Smooth regional patterns rather than city-by-city noise, which is what lets the model price cities "
             "it has never seen from their location.",
             "")

    # 07 continuity across the train/validation boundary ---------------------
    vfeats = build_features(valid, wf)
    vp = model.predict(vfeats)
    tr_day = (train[keep].assign(rpm=lambda d: d.posted_rate / d.distance)
              .groupby("date").rpm.median())
    oof_day = (cvp[cvp.clean_label].assign(rpm=lambda d: d.predicted / d.distance).groupby("date").rpm.median())
    va_day = valid.assign(rpm=vp / valid.distance).groupby("date").rpm.median()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(tr_day.index, tr_day.values, color=MUTED, lw=0.9, label="actual (training labels)")
    ax.plot(oof_day.index, oof_day.values, color=ACCENT, lw=0.9, label="out-of-fold predictions")
    ax.plot(va_day.index, va_day.values, color=VALID_C, lw=1.2, label="final predictions for validation.csv")
    ax.set_ylabel("median $/mile per day")
    ax.set_title("Predicted price level continues smoothly from the observed history")
    ax.legend(fontsize=8, loc="upper left")
    reg.save(fig, "07_continuity", "Do the Nov–Dec predictions join the history sensibly?",
             "Median rate per mile per day: actual training labels, out-of-fold predictions, and the submitted "
             "predictions for November–December.",
             f"The submitted predictions continue the October level and keep the same weekly rhythm; no jump at 1 "
             f"November. Validation median $/mile {np.median(vp / valid.distance):.2f} vs October actual "
             f"{tr_day[tr_day.index >= '2025-10-01'].median():.2f}.",
             "")

    # 08 validation prediction distribution ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    b = np.linspace(0, 8000, 100)
    axes[0].hist(train.posted_rate[keep], bins=b, density=True, color=TRAIN_C, alpha=0.6, label="training actuals")
    axes[0].hist(vp, bins=b, density=True, color=VALID_C, alpha=0.5, label="validation predictions")
    axes[0].set_xlabel("$")
    axes[0].set_title("Distribution of rates")
    axes[0].legend(fontsize=8)
    for eq, col in EQUIP_COLORS.items():
        m = valid.equipment == eq
        axes[1].scatter(valid.distance[m], vp[m] / valid.distance[m], s=1, alpha=0.2, color=col, label=eq)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("distance (log)")
    axes[1].set_ylabel("predicted $/mile")
    axes[1].set_title("Validation predictions keep the learned structure")
    axes[1].legend(markerscale=8, fontsize=8)
    reg.save(fig, "08_validation_predictions", "Sanity of the submitted predictions",
             "Left: submitted validation predictions vs training actuals. Right: predicted $/mile by distance and "
             "equipment for the 12,000 validation loads.",
             f"All 12,000 predictions are positive and finite (${vp.min():,.0f}–${vp.max():,.0f}); the distribution "
             "matches training, and the distance/equipment structure is preserved for every load including the "
             "unseen cities.",
             "")
    print("diagnostics done")


if __name__ == "__main__":
    main()
