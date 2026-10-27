"""Robustness checks requested in review. Each produces a chart + recorded verdict.

Run from the repository root (after `python -m src.train`):
    python -m src.checks
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold

from .features import GBM_FEATURES, LINEAR_FEATURES, build_features, clean
from .model import GBM_PARAMS, GBM_ROUNDS, TwoStageRateModel, apply_outlier_detector, fit_outlier_detector
from .plotting import ACCENT, BAD, GOLD, GOOD, INK, MUTED, FigureRegistry
from .train import FOLDS, load, metrics

SEED = 42
TREND_MULTIPLIERS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def fold_rows(train, feats, log_y, start, end, exclude=None):
    window = (train["date"] < start).to_numpy().copy()
    if exclude is not None:
        window &= ~exclude
    te = ((train["date"] >= start) & (train["date"] < end)).values
    det = fit_outlier_detector(feats[window], log_y[window])
    flagged = apply_outlier_detector(det, feats, log_y).values
    return window & ~flagged, te, flagged


class OOFResidualTwoStage(TwoStageRateModel):
    """Variant: stage 2 is trained on OUT-OF-FOLD Ridge residuals (random 5-fold),
    the standard stacking safeguard against a first stage that over-fits its own rows."""

    def _residual_target(self, features, log_rate):
        x = features[LINEAR_FEATURES]
        oof = np.zeros(len(x))
        for a, b in KFold(5, shuffle=True, random_state=SEED).split(x):
            oof[b] = Ridge(self.ridge_alpha).fit(x.iloc[a], log_rate.iloc[a]).predict(x.iloc[b])
        return log_rate - oof


def trend_adjusted(model, feats, k, t_end):
    """Prediction where the trend beyond the last training date grows at k x the fitted slope."""
    b = model.linear_coefficients()["trend_years"]
    beyond = np.clip(feats["trend_years"].values - t_end, 0, None)
    return np.exp(model.predict_log(feats) - (1 - k) * b * beyond)


def main(out_dir="outputs/figures/checks", only=None):
    reg = FigureRegistry(out_dir, "Checks")
    if only:  # keep captions of checks that are not re-run
        reg.items = json.loads((Path(out_dir) / "captions.json").read_text())
    summary_path = Path(out_dir) / "checks_summary.json"
    results = json.loads(summary_path.read_text()) if (only and summary_path.exists()) else {}
    run = (lambda n: True) if not only else (lambda n: n in only)
    train, valid, daily_mi = load(Path("data"))
    wf = train.groupby("equipment")["weight"].median().to_dict()
    feats = build_features(train, wf)
    log_y = np.log(train["posted_rate"])
    y = train["posted_rate"].values

    # ------------------------------------------------------------------ 1 residual leakage
    fold_models = {}
    if run(1) or run(2):
        for fold, trl, evl, start, end in FOLDS:
            tr, te, flagged = fold_rows(train, feats, log_y, start, end)
            fold_models[fold] = (TwoStageRateModel().fit(feats[tr], log_y[tr]), tr, te, ~flagged[te])
    if run(1):
        print("check 1: in-sample vs out-of-fold residuals")
        rows = []
        for fold, (m, tr, te, cl) in fold_models.items():
            m_oof = OOFResidualTwoStage().fit(feats[tr], log_y[tr])
            rows.append({"fold": fold, "variant": "in-sample Ridge residuals (used)",
                         **metrics(y[te][cl], m.predict(feats[te])[cl])})
            rows.append({"fold": fold, "variant": "out-of-fold Ridge residuals (random 5-fold)",
                         **metrics(y[te][cl], m_oof.predict(feats[te])[cl])})
        r1 = pd.DataFrame(rows)
        piv = r1.pivot(index="fold", columns="variant", values="MAPE")
        # residual agreement on the final training set
        keep_all = ~apply_outlier_detector(fit_outlier_detector(feats, log_y), feats, log_y).values
        xa, ya = feats.loc[keep_all, LINEAR_FEATURES], log_y[keep_all]
        res_in = ya - Ridge(1e-3).fit(xa, ya).predict(xa)
        res_oof = OOFResidualTwoStage()._residual_target(feats[keep_all], ya)
        corr = float(np.corrcoef(res_in, res_oof)[0, 1])
        maxdiff = float(np.abs(res_in - res_oof).max() * 100)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        x = np.arange(3)
        for i, (col, c) in enumerate(zip(piv.columns, [ACCENT, GOLD])):
            axes[0].bar(x + (i - 0.5) * 0.38, piv[col], 0.38, color=c, label=col)
            for xi, v in zip(x, piv[col]):
                axes[0].text(xi + (i - 0.5) * 0.38, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
        axes[0].set_xticks(x, [f"Fold {f}" for f in piv.index])
        axes[0].set_ylabel("MAPE (clean labels)")
        axes[0].set_ylim(0, piv.values.max() * 1.35)
        axes[0].set_title("Fold MAPE")
        axes[0].legend(loc="upper left", fontsize=7.5)
        idx = np.random.default_rng(0).choice(len(res_in), 5000, replace=False)
        axes[1].scatter(res_in.values[idx] * 100, res_oof.values[idx] * 100, s=2, alpha=0.3, color=ACCENT)
        axes[1].plot([-15, 15], [-15, 15], color=BAD, lw=0.8)
        axes[1].set_xlabel("in-sample residual (%)")
        axes[1].set_ylabel("out-of-fold residual (%)")
        axes[1].set_title(f"Residuals agree (r = {corr:.5f})")
        diff = (piv.iloc[:, 1] - piv.iloc[:, 0]).abs().max()
        results["residual_leakage"] = {"max_fold_MAPE_diff_pp": round(float(diff), 3), "residual_corr": corr,
                                       "max_residual_diff_pct": maxdiff}
        reg.save(fig, "01_residual_leakage", "Is stage 2 trained on leaky residuals?",
                 "Left: fold MAPE when the stage-2 models learn in-sample Ridge residuals (submitted) vs out-of-fold "
                 "Ridge residuals (each row's residual from a Ridge fitted on the other 80% of rows). Right: the "
                 "two kinds of residual for the same rows (5,000-row sample, final training set).",
                 f"Largest fold difference {diff:.3f} points. The residuals are practically identical (correlation "
                 f"{corr:.5f}, largest difference {maxdiff:.2f}% of price): Ridge has 8 coefficients fitted on "
                 "15,000–47,000 rows, so it cannot memorise its own rows. (A time-blocked out-of-fold version, "
                 "tried first, was worse — 1.9% on fold 1 — because holding out a whole period makes the "
                 "residuals contain period-level errors, which is a different problem from leakage.) Both variants "
                 "are always scored on later months.",
                 "Keep the simpler in-sample residuals; document the check.")

    # ------------------------------------------------------------------ 2 trend
    if run(2):
        print("check 2: trend sensitivity and stability")
        rows = []
        for fold, (m, tr, te, cl) in fold_models.items():
            t_end = feats.loc[tr, "trend_years"].max()
            for k in TREND_MULTIPLIERS:
                p = trend_adjusted(m, feats[te], k, t_end)
                rows.append({"fold": fold, "k": k, **metrics(y[te][cl], p[cl])})
        r2 = pd.DataFrame(rows)
        mean_k = r2.groupby("k")[["MAPE", "Bias%"]].mean()
        cvres = pd.read_csv("outputs/cv_results.csv")
        no_trend = cvres[(cvres.model == "Two-stage, no trend") & (cvres.subset == "clean labels")].MAPE.mean()
        slopes = []
        for label, end in [("Jan–Apr", "2025-05-01"), ("Jan–Jun", "2025-07-01"), ("Jan–Aug", "2025-09-01"),
                           ("Jan–Oct (final)", "2025-11-01")]:
            tr, _, _ = fold_rows(train, feats, log_y, end, end)
            bb = Ridge(1e-3).fit(feats.loc[tr, LINEAR_FEATURES], log_y[tr]).coef_[LINEAR_FEATURES.index("trend_years")]
            slopes.append((label, (np.exp(bb) - 1) * 100))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for fold, c in zip([1, 2, 3], [MUTED, GOLD, ACCENT]):
            d = r2[r2.fold == fold]
            axes[0].plot(d.k, d.MAPE, "o-", color=c, label=f"Fold {fold}")
        axes[0].plot(mean_k.index, mean_k.MAPE, "s--", color=INK, label="mean of folds")
        axes[0].axhline(no_trend, color=BAD, ls=":", lw=1.2)
        axes[0].text(0, no_trend - 0.09, f"trend removed from the model entirely: {no_trend:.2f}%", fontsize=7.5,
                     color=BAD)
        axes[0].set_xlabel("after training ends, trend continues at k × fitted slope")
        axes[0].set_ylabel("MAPE (clean labels)")
        axes[0].set_title("Extrapolation matters little; the trend term matters a lot")
        axes[0].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=4)
        axes[1].bar([s_[0] for s_ in slopes], [s_[1] for s_ in slopes], color=ACCENT)
        for i, s_ in enumerate(slopes):
            axes[1].text(i, s_[1] + 0.2, f"{s_[1]:.1f}%", ha="center", fontsize=8.5)
        axes[1].set_ylabel("fitted drift, % per year")
        axes[1].set_title("Fitted trend by training window")
        best_k = float(mean_k.MAPE.idxmin())
        f2 = r2[r2.fold == 2].set_index("k").MAPE
        f3 = r2[r2.fold == 3].set_index("k").MAPE
        results["trend"] = {"mean_MAPE_by_k": mean_k.MAPE.round(3).to_dict(), "slopes_pct_per_year":
                            {s_[0]: round(s_[1], 2) for s_ in slopes}, "best_k": best_k, "no_trend_MAPE": no_trend}
        reg.save(fig, "02_trend", "The time trend: what it does and how far to push it",
                 "Left: CV MAPE when, after the last training date, the trend continues at k × its fitted slope "
                 "(k = 0 freezes the level at the training end; k = 1 is the submitted model). Dotted line: the same "
                 "model trained without any trend term. Right: fitted drift for different training windows.",
                 f"Two separate effects. (1) Having a trend term at all is what matters: without it MAPE is "
                 f"{no_trend:.2f}% vs {mean_k.MAPE.loc[1.0]:.2f}%, because the model otherwise prices every month at "
                 "the training-period average level instead of the latest level. (2) How far to extend it beyond the "
                 f"training end matters little and depends on the period: across k = 0–1.5 the mean moves only "
                 f"{mean_k.MAPE.min():.2f}–{mean_k.MAPE.max():.2f}%. Extending helps in fold 3 ({f3.loc[0.0]:.2f}% → "
                 f"{f3.loc[1.0]:.2f}%) and hurts in fold 2 ({f2.loc[0.0]:.2f}% → {f2.loc[1.0]:.2f}%), where the "
                 f"slope was estimated from Jan–Jun only ({slopes[1][1]:.1f}%/yr, later revised to "
                 f"{slopes[3][1]:.1f}%/yr).",
                 f"Keep k = 1. The mean-optimal k = {best_k:g} is chosen from only three folds and differs from "
                 "k = 1 by a few hundredths of a point, so tuning it would be over-fitting. The December chart is "
                 "shown under k = 0, 0.5 and 1 (check 6) so the reader can see the assumption.")

    # ------------------------------------------------------------------ 3 date-effect
    if run(3):
        print("check 3: date-effect decomposition")
        keep = ~fold_rows(train, feats, log_y, "2025-11-01", "2025-11-01")[2] & train["weight"].notna().values
        cols = ["log_distance", "eq_reefer", "eq_flatbed", "log_weight", "quote_signal", "market_index_dev"]
        X = feats.loc[keep, cols].reset_index(drop=True)
        D = pd.get_dummies(train.loc[keep, "date"].dt.strftime("%Y-%m-%d")).astype(float).reset_index(drop=True)
        fe_reg = LinearRegression(fit_intercept=False).fit(pd.concat([X, D], axis=1), log_y[keep].values)
        eff = pd.Series(fe_reg.coef_[len(cols):], index=pd.to_datetime(D.columns))
        eff -= eff.mean()
        lmi = np.log(daily_mi.reindex(eff.index)).values
        tt = ((eff.index - pd.Timestamp("2025-01-01")).days / 365).values
        dq = ((eff.index + pd.offsets.QuarterEnd(0)) - eff.index).days.values
        ramp = np.clip(1 - dq / 28, 0, 1)
        designs = {"market index only": np.c_[lmi], "+ quarter-end ramp": np.c_[lmi, ramp],
                   "+ linear trend": np.c_[lmi, ramp, tt]}
        is_tr = eff.index < "2025-09-01"
        r2_in, rmse_out, preds = {}, {}, {}
        for name, Xd in designs.items():
            r2_in[name] = LinearRegression().fit(Xd, eff.values).score(Xd, eff.values)
            p = LinearRegression().fit(Xd[is_tr], eff.values[is_tr]).predict(Xd[~is_tr])
            preds[name] = p
            rmse_out[name] = float(np.sqrt(np.mean((p - eff.values[~is_tr]) ** 2)) * 100)
        naive = eff.values[is_tr][-14:].mean()
        rmse_out["naive: last 2 weeks of Aug carried forward"] = float(
            np.sqrt(np.mean((naive - eff.values[~is_tr]) ** 2)) * 100)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1, 1.1, 1.6], "wspace": 0.45})
        xx = np.arange(3)
        axes[0].bar(xx, list(r2_in.values()), color=ACCENT)
        for i, v in enumerate(r2_in.values()):
            axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8.5)
        axes[0].set_xticks(xx, ["market\nindex", "+ quarter-\nend ramp", "+ linear\ntrend"], fontsize=8)
        axes[0].set_ylim(0, 1.1)
        axes[0].set_title("In-sample R² (304 dates)", fontsize=10)
        names = list(rmse_out)
        cols_ = [MUTED, MUTED, ACCENT, BAD]
        axes[1].barh(range(4), [rmse_out[n] for n in names], color=cols_)
        for i, n in enumerate(names):
            axes[1].text(rmse_out[n] + 0.08, i, f"{rmse_out[n]:.2f}%", va="center", fontsize=8)
        axes[1].set_yticks(range(4), ["market index", "+ ramp", "+ trend", "naive"], fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, max(rmse_out.values()) * 1.3)
        axes[1].set_title("Out-of-sample error, Sep–Oct\n(RMSE, % of price)", fontsize=10)
        dts = eff.index[~is_tr]
        axes[2].plot(dts, eff.values[~is_tr] * 100, color=INK, lw=1.4, label="actual date effect")
        axes[2].plot(dts, preds["+ linear trend"] * 100, color=ACCENT, lw=1.4, label="3-term model (fit Jan–Aug)")
        axes[2].plot(dts, preds["market index only"] * 100, color=MUTED, lw=1, label="market index only")
        axes[2].axhline(naive * 100, color=BAD, ls="--", lw=1, label="naive")
        axes[2].xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(bymonthday=[1, 15]))
        axes[2].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
        axes[2].set_ylabel("% vs average day")
        axes[2].set_title("Predicting unseen Sep–Oct price levels", fontsize=10)
        axes[2].legend(fontsize=7, loc="lower left")
        results["date_effect"] = {"in_sample_r2": {k: round(v, 3) for k, v in r2_in.items()},
                                  "out_of_sample_rmse_pct_SepOct": {k: round(v, 3) for k, v in rmse_out.items()}}
        reg.save(fig, "03_date_effect", "The date effect: how the 0.97 is defined and whether it carries forward",
                 "Step 1: regress log(rate) on load features (log distance, equipment, log weight, quote_signal, the "
                 "load's market-index deviation) plus one dummy per date (304 dates, clean labels). The dummy "
                 "coefficients are the per-date price levels. Step 2: regress those levels on daily "
                 "log(market_index), the quarter-end ramp and a linear trend. Left: ordinary in-sample R² of step 2. "
                 "Middle/right: step 2 fitted on Jan–Aug dates only, then used to predict the Sep–Oct levels.",
                 f"In-sample R² rises {r2_in['market index only']:.2f} → {r2_in['+ quarter-end ramp']:.2f} → "
                 f"{r2_in['+ linear trend']:.2f}. Out of sample, the 3-term model predicts the daily price level "
                 f"of unseen Sep–Oct within {rmse_out['+ linear trend']:.2f}% RMSE, vs "
                 f"{rmse_out['naive: last 2 weeks of Aug carried forward']:.2f}% for carrying the last level "
                 f"forward and {rmse_out['market index only']:.2f}% for market index alone (which gets the weekly "
                 "wiggle but not the level). R² is not used out of sample because a 2-month window has little "
                 "variance, which makes R² unstable.",
                 "Describe 0.97 as an in-sample, descriptive figure; use the out-of-sample error as the evidence "
                 "that the structure carries forward two months. It does not prove it holds in Nov–Dec.")

    # ------------------------------------------------------------------ 4 unseen cities
    if run(4):
        print("check 4: unseen-city stress test (5 draws)")
        fold, trl, evl, start, end = FOLDS[-1]
        rng = np.random.default_rng(SEED)
        all_cities = sorted(train["pickup"].unique())
        rows = []
        for draw in range(5):
            cities = rng.choice(all_cities, 8, replace=False)
            touches = (train["pickup"].isin(cities) | train["delivery"].isin(cities)).values
            tr_with, te, flagged = fold_rows(train, feats, log_y, start, end)
            tr_wo, _, flagged_wo = fold_rows(train, feats, log_y, start, end, exclude=touches)
            sel = te & touches & ~flagged_wo
            p_with = TwoStageRateModel().fit(feats[tr_with], log_y[tr_with]).predict(feats[sel])
            p_wo = TwoStageRateModel().fit(feats[tr_wo], log_y[tr_wo]).predict(feats[sel])
            rows.append({"draw": draw + 1, "cities": ", ".join(cities), "n": int(sel.sum()),
                         "MAPE cities seen in training": metrics(y[sel], p_with)["MAPE"],
                         "MAPE cities removed from training": metrics(y[sel], p_wo)["MAPE"]})
            print(f"   draw {draw + 1}: {rows[-1]['MAPE cities seen in training']:.2f} vs "
                  f"{rows[-1]['MAPE cities removed from training']:.2f}")
        r4 = pd.DataFrame(rows)
        r4.to_csv(Path(out_dir) / "unseen_city_draws.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        x = np.arange(len(r4))
        ax.bar(x - 0.2, r4["MAPE cities seen in training"], 0.4, color=ACCENT, label="cities kept in training")
        ax.bar(x + 0.2, r4["MAPE cities removed from training"], 0.4, color=BAD, label="cities removed from training")
        ax.set_xticks(x, [f"draw {d}\n(n={n:,})" for d, n in zip(r4.draw, r4.n)])
        ax.set_ylabel("MAPE on loads touching the 8 cities")
        ax.set_title("Cost of never having seen a city: 5 random draws of 8 cities (fold 3)")
        ax.set_ylim(0, r4[["MAPE cities seen in training", "MAPE cities removed from training"]].values.max() * 1.3)
        for xi, (a_, b_) in enumerate(zip(r4["MAPE cities seen in training"], r4["MAPE cities removed from training"])):
            ax.text(xi - 0.2, a_ + 0.02, f"{a_:.2f}", ha="center", fontsize=7.5)
            ax.text(xi + 0.2, b_ + 0.02, f"{b_:.2f}", ha="center", fontsize=7.5)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        gap = (r4["MAPE cities removed from training"] - r4["MAPE cities seen in training"])
        results["unseen_cities"] = r4.round(3).to_dict("records")
        reg.save(fig, "04_unseen_cities", "Generalisation to unseen cities",
                 "For each draw, 8 random cities are chosen. The same fold-3 loads touching them are predicted by a "
                 "model trained with those cities and one trained with every load touching them removed.",
                 f"Removing the cities raises MAPE by {gap.mean():.2f} points on average (range {gap.min():.2f} to "
                 f"{gap.max():.2f}); MAPE stays at {r4['MAPE cities removed from training'].min():.2f}–"
                 f"{r4['MAPE cities removed from training'].max():.2f}%. The random cities are not the 8 real "
                 "unseen validation cities (those have no labels), so this tests unseen cities in general.",
                 "Evidence (not proof) that predictions for the 1,447 validation loads touching new cities should be "
                 "close in quality to the rest.")

    # ------------------------------------------------------------------ 5 outlier handling
    if run(5):
        print("check 5: effect of removing anomalous labels")
        rows = []
        for fold, trl, evl, start, end in FOLDS:
            tr, te, flagged = fold_rows(train, feats, log_y, start, end)
            window = (train["date"] < start).values
            cl = ~flagged[te]
            for name, rows_used in [("anomalies removed (used)", tr), ("anomalies kept in training", window)]:
                p = TwoStageRateModel().fit(feats[rows_used], log_y[rows_used]).predict(feats[te])
                rows.append({"fold": fold, "variant": name, "subset": "clean labels", **metrics(y[te][cl], p[cl])})
                rows.append({"fold": fold, "variant": name, "subset": "all labels", **metrics(y[te], p)})
        r5 = pd.DataFrame(rows).groupby(["variant", "subset"])[["MAPE", "MAE", "RMSE"]].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        for ax, sub in zip(axes, ["clean labels", "all labels"]):
            d = r5[r5.subset == sub]
            ax.bar(d.variant, d.MAPE, color=[MUTED, ACCENT][:len(d)])
            for i, v in enumerate(d.MAPE):
                ax.text(i, v + 0.03, f"{v:.2f}%", ha="center", fontsize=8.5)
            ax.set_title(f"Scored on {sub}")
            ax.set_ylabel("mean MAPE")
            ax.tick_params(axis="x", labelsize=8)
        results["outlier_handling"] = r5.round(3).to_dict("records")
        reg.save(fig, "05_outlier_impact", "Does removing anomalous labels help?",
                 "Two-stage model trained with vs without the flagged labels (detector fitted per fold on training "
                 "months), scored on clean labels and on all labels.",
                 "Removing them improves accuracy on normal loads AND on all loads (anomalies included): a few "
                 "labels at 3–5× or 0.2–0.4× pull a log-scale model's fit away from the bulk of the data, so "
                 "keeping them makes every prediction slightly worse. The all-label error stays high either way "
                 "because no model can anticipate a 4× label.",
                 "Exclude them from training; report both metrics.")

    # ------------------------------------------------------------------ 6 December scenarios
    if run(6):
        print("check 6: December input scenarios")
        keep_final = ~apply_outlier_detector(fit_outlier_detector(feats, log_y), feats, log_y).values
        final = TwoStageRateModel().fit(feats[keep_final], log_y[keep_final])
        dec = pd.read_csv("data/december_chart_inputs.csv")
        dec["date"] = pd.to_datetime(dec["date"])
        both = pd.concat([train, valid])
        coords = {}
        for side in ["pickup", "delivery"]:
            cc = both.groupby(side)[[f"{side}_lat", f"{side}_lon"]].first()
            cc.columns = ["lat", "lon"]
            coords.update(cc.to_dict("index"))
        for side in ["pickup", "delivery"]:
            dec[f"{side}_lat"] = dec[side].map(lambda c: coords[c]["lat"])
            dec[f"{side}_lon"] = dec[side].map(lambda c: coords[c]["lon"])
        oct_mi = daily_mi[(daily_mi.index >= "2025-10-01") & (daily_mi.index < "2025-11-01")]
        weekday_prof = oct_mi.groupby(oct_mi.index.dayofweek).mean()
        scen_mi = {
            "A: observed daily market index (validation.csv) — submitted": dec.date.map(daily_mi).values,
            "B: training-only forecast (October weekday profile)": dec.date.dt.dayofweek.map(weekday_prof).values,
            "C: market index held at October average": np.full(len(dec), oct_mi.mean()),
        }
        daily_qs = pd.concat([train, valid]).groupby("date")["quote_signal"].median()
        qs = dec.date.map(daily_qs).values  # same quote_signal input as the submitted file
        t_end = feats.loc[keep_final, "trend_years"].max()
        curves = {}
        for name, m_i in scen_mi.items():
            d = dec.copy()
            d["market_index"], d["quote_signal"] = m_i, qs
            f = build_features(clean(d, daily_mi), wf)
            curves[name] = final.predict(f)
        d = dec.copy()
        d["market_index"], d["quote_signal"] = scen_mi[list(scen_mi)[0]], qs
        fA = build_features(clean(d, daily_mi), wf)
        for k in [0.0, 0.5]:
            curves[f"A with trend frozen at Oct 31 (k = {k:g})" if k == 0 else f"A with half-speed trend (k = {k:g})"] = \
                trend_adjusted(final, fA, k, t_end)
        fig, ax = plt.subplots(figsize=(9, 3.8))
        styles = [(ACCENT, "-", 2.4), (GOLD, "-", 1.6), (MUTED, "-", 1.6), (BAD, ":", 1.4), (BAD, "--", 1.2)]
        for (name, cv), (c, ls, lw) in zip(curves.items(), styles):
            ax.plot(dec.date, cv, ls, color=c, lw=lw, label=f"{name}  (${cv.min():.0f}–${cv.max():.0f})")
        ax.set_ylabel("predicted rate ($)")
        ax.set_title("December lane under different input assumptions")
        ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=1)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
        spread = max(v.max() for v in curves.values()) - min(v.min() for v in curves.values())
        results["december_scenarios"] = {k: [round(float(v.min()), 2), round(float(v.max()), 2), round(float(v.mean()), 2)]
                                         for k, v in curves.items()}
        reg.save(fig, "06_december_scenarios", "December chart: sensitivity to assumptions",
                 "The fixed Lexington → Fort Wayne dry-van load priced for every December day under alternative "
                 "inputs. A uses the market_index values actually supplied for those dates in validation.csv "
                 "(features only, never labels). B and C use training data only. The last two vary the trend.",
                 f"All scenarios lie within a ${spread:.0f} band (~{spread / curves[list(curves)[0]].mean() * 100:.0f}% "
                 "of the price). The market index mainly shapes the 7-day wiggle (A vs B is nearly identical because "
                 "the weekly pattern is so regular); the trend assumption shifts the level by a few percent.",
                 "Submit A: the assessment supplies market_index for these dates, and using a provided input feature "
                 "is not leakage. B shows the result barely changes if those values were not available.")

    (Path(out_dir) / "checks_summary.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    import sys
    sel = {int(a) for a in sys.argv[1:] if a.isdigit()}
    main(only=sel or None)
