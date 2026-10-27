"""Model comparison: many algorithm families, same time folds, same metrics.

Every candidate is evaluated two ways:
  direct  - the algorithm predicts log(rate) from all features (incl. a time trend
            feature, so it is not handicapped by missing information)
  hybrid  - a Ridge base model carries the extrapolating effects (distance,
            equipment, weight, market index, quarter-end, trend) and the algorithm
            learns only the Ridge residual from features WITHOUT absolute time.
Then ensembles are built from the hybrid models' out-of-fold predictions.

Run from the repository root (takes ~10-15 min on 2 cores):
    python -m src.compare
"""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import nnls
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import GammaRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import GBM_FEATURES, LINEAR_FEATURES, build_features
from .model import GBM_PARAMS, GBM_ROUNDS, apply_outlier_detector, fit_outlier_detector
from .train import FOLDS, RatePerMileBaseline, load, metrics

SEED = 42
# Direct models see everything the hybrid sees, plus the extrapolating terms.
DIRECT_FEATURES = GBM_FEATURES + ["log_weight", "quarter_end_ramp", "trend_years"]


class Imputed:
    """Fill NaN (only weight can be missing) with TRAINING medians, then fit an sklearn model."""

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.medians_ = X.median()
        self.model.fit(X.fillna(self.medians_), y)
        return self

    def predict(self, X):
        return self.model.predict(X.fillna(self.medians_))


# --- learners: each is (fit(X, y) -> model, predict(model, X) -> y) --------------
def lgbm():
    return (lambda X, y: lgb.train(GBM_PARAMS, lgb.Dataset(X, y), GBM_ROUNDS),
            lambda m, X: m.predict(X))


def xgboost():
    params = dict(n_estimators=800, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, min_child_weight=10, tree_method="hist", random_state=SEED, n_jobs=2)
    return (lambda X, y: xgb.XGBRegressor(**params).fit(X, y), lambda m, X: m.predict(X))


def hgb():
    return (lambda X, y: HistGradientBoostingRegressor(max_iter=800, learning_rate=0.03, max_leaf_nodes=31,
                                                       min_samples_leaf=50, l2_regularization=1.0,
                                                       random_state=SEED).fit(X, y),
            lambda m, X: m.predict(X))


def random_forest():
    return (lambda X, y: Imputed(RandomForestRegressor(n_estimators=150, min_samples_leaf=3, max_features=0.5,
                                                       n_jobs=2, random_state=SEED)).fit(X, y),
            lambda m, X: m.predict(X))


def extra_trees():
    return (lambda X, y: Imputed(ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.7,
                                                     n_jobs=2, random_state=SEED)).fit(X, y),
            lambda m, X: m.predict(X))


def knn():
    return (lambda X, y: Imputed(make_pipeline(StandardScaler(),
                                               KNeighborsRegressor(n_neighbors=15, weights="distance"))).fit(X, y),
            lambda m, X: m.predict(X))


def mlp():
    return (lambda X, y: Imputed(make_pipeline(StandardScaler(),
                                               MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                                            learning_rate_init=1e-3, max_iter=200,
                                                            early_stopping=True, random_state=SEED))).fit(X, y),
            lambda m, X: m.predict(X))


LEARNERS = {
    "LightGBM": lgbm, "XGBoost": xgboost, "HistGradientBoosting": hgb, "Random Forest": random_forest,
    "Extra Trees": extra_trees, "k-Nearest Neighbours": knn, "Neural net (MLP)": mlp,
}
FAMILY = {"LightGBM": "Gradient boosting", "XGBoost": "Gradient boosting", "HistGradientBoosting": "Gradient boosting",
          "Random Forest": "Bagged trees", "Extra Trees": "Bagged trees", "k-Nearest Neighbours": "Instance-based",
          "Neural net (MLP)": "Neural network"}


def evaluate_fold(train, feats, log_y, tr, te):
    """Return {model name: log-scale predictions on te} and fit times."""
    preds, times = {}, {}

    def timed(name, fn):
        t0 = time.time()
        preds[name] = fn()
        times[name] = time.time() - t0
        print(f"    {name:<45s} {times[name]:6.1f}s")

    timed("Baseline: median $/mile",
          lambda: np.log(RatePerMileBaseline().fit(train[tr], log_y[tr]).predict(train[te])))
    timed("Ridge (log-linear)",
          lambda: Ridge(1e-3).fit(feats.loc[tr, LINEAR_FEATURES], log_y[tr]).predict(feats.loc[te, LINEAR_FEATURES]))
    timed("Gamma GLM (log link)",
          lambda: np.log(make_pipeline(StandardScaler(), GammaRegressor(alpha=1e-4, max_iter=1000))
                         .fit(feats.loc[tr, LINEAR_FEATURES], np.exp(log_y[tr]))
                         .predict(feats.loc[te, LINEAR_FEATURES])))

    ridge = Ridge(1e-3).fit(feats.loc[tr, LINEAR_FEATURES], log_y[tr])
    base_tr = ridge.predict(feats.loc[tr, LINEAR_FEATURES])
    base_te = ridge.predict(feats.loc[te, LINEAR_FEATURES])
    for name, factory in LEARNERS.items():
        fit, pred = factory()
        timed(f"{name} — direct",
              lambda: pred(fit(feats.loc[tr, DIRECT_FEATURES], log_y[tr]), feats.loc[te, DIRECT_FEATURES]))
        fit, pred = factory()
        timed(f"{name} — hybrid (Ridge + {name})",
              lambda: base_te + pred(fit(feats.loc[tr, GBM_FEATURES], log_y[tr] - base_tr),
                                     feats.loc[te, GBM_FEATURES]))
    return preds, times


def add_ensembles(oof: pd.DataFrame) -> pd.DataFrame:
    """Ensembles from hybrid models. Blend weights are learned only on EARLIER folds
    (walk-forward), so no fold's own labels choose its weights."""
    hybrids = [c for c in oof.columns if c.endswith(")") and "hybrid" in c]
    oof["Ensemble: mean of all hybrids"] = oof[hybrids].mean(axis=1)
    top3 = ["LightGBM — hybrid (Ridge + LightGBM)", "XGBoost — hybrid (Ridge + XGBoost)",
            "Extra Trees — hybrid (Ridge + Extra Trees)"]
    oof["Ensemble: LightGBM + XGBoost + Extra Trees (hybrids)"] = oof[top3].mean(axis=1)
    oof["Ensemble: LightGBM + XGBoost (hybrids)"] = oof[top3[:2]].mean(axis=1)
    # Stacking with non-negative weights summing to 1, walk-forward.
    stacked = pd.Series(np.nan, index=oof.index)
    weights = {}
    for fold in sorted(oof["fold"].unique()):
        past = (oof["fold"] < fold) & oof["clean"]
        if past.sum() == 0:
            continue  # no earlier fold to learn weights from
        A = oof.loc[past, hybrids].values
        w, _ = nnls(A, oof.loc[past, "log_actual"].values)
        w = w / w.sum()
        weights[fold] = dict(zip(hybrids, np.round(w, 3)))
        cur = oof["fold"] == fold
        stacked[cur] = oof.loc[cur, hybrids].values @ w
    oof["Ensemble: stacked (NNLS weights from earlier folds)"] = stacked
    return oof, weights


def main(out_dir="outputs"):
    out = Path(out_dir)
    train, valid, daily_mi = load(Path("data"))
    feats = build_features(train, train.groupby("equipment")["weight"].median().to_dict())
    log_y = np.log(train["posted_rate"])
    frames, time_rows = [], []
    for fold, train_label, eval_label, start, end in FOLDS:
        print(f"fold {fold}: train {train_label} -> eval {eval_label}")
        window = (train["date"] < start).values
        te = ((train["date"] >= start) & (train["date"] < end)).values
        det = fit_outlier_detector(feats[window], log_y[window])
        flagged = apply_outlier_detector(det, feats, log_y).values
        tr = window & ~flagged
        preds, times = evaluate_fold(train, feats, log_y, tr, te)
        f = pd.DataFrame(preds)
        f["fold"], f["clean"] = fold, ~flagged[te]
        f["log_actual"] = log_y[te].values
        f["load_id"] = train.loc[te, "load_id"].values
        f["date"] = train.loc[te, "date"].values
        frames.append(f)
        time_rows += [{"fold": fold, "model": k, "fit_seconds": v} for k, v in times.items()]
    oof = pd.concat(frames, ignore_index=True)
    oof, weights = add_ensembles(oof)

    meta = {"fold", "clean", "log_actual", "load_id", "date"}
    models = [c for c in oof.columns if c not in meta]
    rows = []
    for m in models:
        for subset in ["clean labels", "all labels"]:
            for fold in [1, 2, 3]:
                d = oof[(oof.fold == fold) & (oof.clean if subset == "clean labels" else True)].dropna(subset=[m])
                if len(d) == 0:
                    continue
                rows.append({"model": m, "subset": subset, "fold": fold, "n": len(d),
                             **metrics(np.exp(d.log_actual.values), np.exp(d[m].values))})
    res = pd.DataFrame(rows)
    res.to_csv(out / "model_comparison_by_fold.csv", index=False)

    tm = pd.DataFrame(time_rows).groupby("model").fit_seconds.mean()
    summ = []
    for m in models:
        r = res[res.model == m]
        c, a = r[r.subset == "clean labels"], r[r.subset == "all labels"]
        summ.append({"model": m, "folds_scored": len(c),
                     "MAPE": c.MAPE.mean(), "MAE": c.MAE.mean(), "RMSE": c.RMSE.mean(), "Bias%": c["Bias%"].mean(),
                     "MAPE fold 3": c[c.fold == 3].MAPE.iloc[0], "MAPE fold 3 all labels": a[a.fold == 3].MAPE.iloc[0],
                     "MAPE all labels": a.MAPE.mean(), "worst fold MAPE": c.MAPE.max(),
                     "fit_seconds": tm.get(m, np.nan)})
    summ = pd.DataFrame(summ).sort_values("MAPE fold 3")
    summ.to_csv(out / "model_comparison.csv", index=False)
    oof.to_csv(out / "model_comparison_oof.csv.gz", index=False, float_format="%.5f")
    pd.Series({str(k): v for k, v in weights.items()}).to_json(out / "stacking_weights.json", indent=2)
    pd.set_option("display.width", 250)
    print(summ.round(3).to_string())
    print(weights)




# ============================================================================ charts
def bootstrap_vs(oof: pd.DataFrame, ref: str, other: str, mask=None, n_boot=2000, seed=0):
    """Difference in MAPE (other - ref), with a 95% CI from resampling whole DATES
    (loads on the same day share market conditions, so they are not independent)."""
    d = oof[oof.clean] if mask is None else oof[oof.clean & mask]
    y = np.exp(d.log_actual)
    diff = (np.abs(np.exp(d[other]) - y) - np.abs(np.exp(d[ref]) - y)) / y * 100
    g = diff.groupby(d.date).agg(["sum", "size"])
    rng = np.random.default_rng(seed)
    s, n = g["sum"].values, g["size"].values
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(g), len(g))
        boots.append(s[i].sum() / n[i].sum())
    return s.sum() / n.sum(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), float((g["sum"] < 0).mean())


def plots(out_dir="outputs/figures/models"):
    import matplotlib.pyplot as plt
    from .plotting import ACCENT, BAD, GOLD, INK, MUTED, FigureRegistry
    reg = FigureRegistry(out_dir, "Models")
    summ = pd.read_csv("outputs/model_comparison.csv")
    byf = pd.read_csv("outputs/model_comparison_by_fold.csv")
    oof = pd.read_csv("outputs/model_comparison_oof.csv.gz", parse_dates=["date"])
    s = summ.set_index("model")

    # 01 direct vs hybrid --------------------------------------------------
    names = list(LEARNERS)
    direct = [s.loc[f"{n} — direct", "MAPE"] for n in names]
    hybrid = [s.loc[f"{n} — hybrid (Ridge + {n})", "MAPE"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(names))
    ax.bar(x - 0.2, np.minimum(direct, 6), 0.4, color=MUTED, label="direct: algorithm alone on log(rate)")
    ax.bar(x + 0.2, hybrid, 0.4, color=ACCENT, label="hybrid: Ridge base + algorithm on residual")
    for xi, d, h in zip(x, direct, hybrid):
        ax.text(xi - 0.2, min(d, 6) + 0.08, f"{d:.1f}" + ("↑" if d > 6 else ""), ha="center", fontsize=7.5)
        ax.text(xi + 0.2, h + 0.08, f"{h:.2f}", ha="center", fontsize=7.5)
    for ref, lab in [("Ridge (log-linear)", "Ridge alone"), ("Baseline: median $/mile", "$/mile baseline")]:
        ax.axhline(s.loc[ref, "MAPE"], color=BAD if "Baseline" in ref else GOLD, ls="--", lw=0.9)
        ax.text(len(names) - 0.5, s.loc[ref, "MAPE"] + 0.05, f"{lab} {s.loc[ref, 'MAPE']:.2f}%", ha="right",
                fontsize=7.5)
    ax.set_xticks(x, [n.replace(" (", "\n(").replace("HistGradientBoosting", "HistGradient-\nBoosting")
                      .replace("k-Nearest Neighbours", "k-Nearest\nNeighbours").replace("Random Forest", "Random\nForest")
                      .replace("Extra Trees", "Extra\nTrees") for n in names], fontsize=8)
    ax.set_ylim(0, 6.3)
    ax.set_ylabel("MAPE, mean of 3 folds (clean labels)")
    ax.set_title("Every algorithm is better on top of the Ridge base than on its own")
    ax.legend(fontsize=8, loc="upper left")
    reg.save(fig, "01_direct_vs_hybrid", "Direct vs hybrid, for 7 algorithm families",
             "Each algorithm trained (a) directly on log(rate) with all features including a time-trend feature, "
             "(b) on the residual of the Ridge base. Same folds, same rows, same metric. Bars clipped at 6% "
             "(k-NN and MLP direct are far worse).",
             "The hybrid wins for every family, by 0.5–1 point for tree models. Reason: trees cannot extrapolate. "
             "Given a trend feature, a tree predicts the same value for every date after its last split, so direct "
             "tree models under-predict later months (negative bias in folds 1 and 3, next chart). The Ridge base "
             "carries the trend forward; the tree only learns patterns that repeat (geography, interactions). "
             "k-NN and MLP direct fail badly: unscaled time extrapolation and no built-in log-linear structure.",
             "Use the hybrid structure for the final model.")

    # 02 bias by fold ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.4))
    cl = byf[byf.subset == "clean labels"]
    for n, col in [("LightGBM", ACCENT), ("XGBoost", GOLD), ("Random Forest", BAD)]:
        for kind, ls in [("direct", "--"), ("hybrid", "-")]:
            m = f"{n} — direct" if kind == "direct" else f"{n} — hybrid (Ridge + {n})"
            d = cl[cl.model == m]
            ax.plot(d.fold, d["Bias%"], ls, marker="o", color=col, label=f"{n} {kind}")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks([1, 2, 3], ["Fold 1\nMay–Jun", "Fold 2\nJul–Aug", "Fold 3\nSep–Oct"])
    ax.set_ylabel("bias (mean % error)")
    ax.set_title("Direct tree models under-predict when prices are rising (folds 1 and 3)")
    ax.legend(fontsize=7.5, ncol=3, loc="lower left")
    reg.save(fig, "02_bias_by_fold", "Why direct trees lose: bias",
             "Mean signed % error per fold (+ = over-prediction) for three tree learners, direct (dashed) vs "
             "hybrid (solid).",
             "Direct models are 1.2–2.3% too low in folds 1 and 3: they freeze the price level at the end of "
             "training. Hybrids stay within ±0.6% there. In fold 2 every model over-predicts by ~1.5% because the trend "
             "estimated from Jan–Jun alone was too steep (see checks).",
             "")

    # 03 full leaderboard --------------------------------------------------
    s2 = summ[summ.MAPE < 6].sort_values("MAPE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    yy = np.arange(len(s2))
    colors = [ACCENT if m.startswith("Ensemble: LightGBM + XGBoost + Extra") else
              (GOLD if m.startswith("Ensemble") else (MUTED if "direct" in m or "Baseline" in m or "Ridge (" in m
                                                       or "GLM" in m else "#5E8C96")) for m in s2.model]
    ax.barh(yy, s2.MAPE, color=colors)
    for fold, mk in [(1, "o"), (2, "s"), (3, "D")]:
        v = [cl[(cl.model == m) & (cl.fold == fold)].MAPE.values for m in s2.model]
        ax.scatter([vv[0] if len(vv) else np.nan for vv in v], yy, marker=mk, s=14, color=INK, zorder=3,
                   label=f"fold {fold}")
    ax.set_yticks(yy, s2.model, fontsize=7.5)
    ax.set_xlabel("MAPE (bar = mean of folds; markers = individual folds)")
    ax.set_title("All candidates, ranked")
    ax.legend(fontsize=8, loc="lower right")
    excluded = summ[summ.MAPE >= 6].model.tolist()
    reg.save(fig, "03_leaderboard", "Leaderboard of all 21 candidates",
             "Mean clean-label MAPE over the 3 time folds (bars) and each fold separately (markers). Teal = chosen "
             f"ensemble, gold = other ensembles. The stacked ensemble has scores for folds 2–3 only (its weights "
             f"need an earlier fold), so its bar is not directly comparable. Not shown (MAPE > 6%): {', '.join(excluded)}.",
             "Three tiers: (1) hybrids and their ensembles at 1.6–1.9%; (2) direct tree models at 2.2–2.7%; "
             "(3) linear/GLM at 2.8% and the $/mile baseline at 4.6%. Ridge and the Gamma GLM are "
             "indistinguishable, so the log-normal Ridge is not losing anything versus a GLM made for positive "
             "skewed targets.",
             "")

    # 04 ensemble significance --------------------------------------------
    ref = "LightGBM — hybrid (Ridge + LightGBM)"
    cands = ["Ensemble: LightGBM + XGBoost + Extra Trees (hybrids)", "Ensemble: mean of all hybrids",
             "Ensemble: LightGBM + XGBoost (hybrids)", "XGBoost — hybrid (Ridge + XGBoost)",
             "Extra Trees — hybrid (Ridge + Extra Trees)", "Random Forest — hybrid (Ridge + Random Forest)",
             "HistGradientBoosting — hybrid (Ridge + HistGradientBoosting)"]
    rows = []
    for c_ in cands:
        for lab, mask in [("all 3 folds", None), ("fold 3 only (most recent)", oof.fold == 3)]:
            est, lo, hi, share = bootstrap_vs(oof, ref, c_, mask)
            rows.append({"model": c_, "window": lab, "diff": est, "lo": lo, "hi": hi, "share_dates_better": share})
    bs = pd.DataFrame(rows)
    bs.to_csv("outputs/model_bootstrap_vs_lightgbm_hybrid.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, (lab, col, off) in enumerate([("all 3 folds", ACCENT, -0.15), ("fold 3 only (most recent)", GOLD, 0.15)]):
        d = bs[bs.window == lab]
        yy = np.arange(len(d)) + off
        ax.errorbar(d["diff"], yy, xerr=[d["diff"] - d.lo, d.hi - d["diff"]], fmt="o", color=col, capsize=3,
                    label=lab)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(np.arange(len(cands)), [c_.replace(" (hybrids)", "").replace(" — hybrid", "")
                                          .replace("(Ridge + ", "(R+") for c_ in cands], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("MAPE difference vs Ridge + LightGBM (points); left of 0 = better")
    ax.set_title("Only the ensembles improve reliably; 95% CIs from resampling dates")
    ax.legend(fontsize=8, loc="lower right")
    e = bs[(bs.model == cands[0])]
    reg.save(fig, "04_ensemble_significance", "Is the ensemble's gain real?",
             "Difference in MAPE between each candidate and the original Ridge + LightGBM, with 95% confidence "
             "intervals from a bootstrap over dates (2,000 resamples; whole days are resampled because loads on "
             "the same day share market conditions).",
             f"LightGBM + XGBoost + Extra Trees: {e.iloc[0]['diff']:+.3f} points over all folds "
             f"(CI {e.iloc[0].lo:+.3f} to {e.iloc[0].hi:+.3f}, better on {e.iloc[0].share_dates_better:.0%} of "
             f"dates) and {e.iloc[1]['diff']:+.3f} on fold 3 (CI {e.iloc[1].lo:+.3f} to {e.iloc[1].hi:+.3f}: no "
             "loss). Single learners are within ±0.02 points of LightGBM (XGBoost, HistGradientBoosting) or "
             "clearly worse on the most recent fold (Random Forest / Extra Trees alone lose 0.13–0.15 points). Combining tree types that make "
             "different errors (boosting vs randomised bagging) is what helps.",
             "Chosen model: Ridge base + equal-weight average of LightGBM, XGBoost and Extra Trees residual "
             "models. Rule applied: best mean across folds, provided it is not significantly worse on the most "
             "recent fold. Gain is modest (~$2 MAE, 4% relative) at the cost of ~3× training time (~10 s).")

    # 05 stacking ----------------------------------------------------------
    w = pd.read_json("outputs/stacking_weights.json", typ="series")
    W = pd.DataFrame({k: v for k, v in w.items()}).T
    W.columns = [c_.split(" — ")[0] for c_ in W.columns]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    W.T.plot.barh(ax=axes[0], color=[MUTED, ACCENT])
    axes[0].set_title("Stacking weights learned on earlier folds")
    axes[0].legend(["learned on fold 1 → used in fold 2", "learned on folds 1–2 → used in fold 3"], fontsize=7.5)
    st = cl[cl.model.str.startswith("Ensemble")]
    piv = st.pivot(index="model", columns="fold", values="MAPE")
    piv.index = [i.replace("Ensemble: ", "").replace(" (hybrids)", "") for i in piv.index]
    piv.index = [i.replace(" + ", "+\n").replace("stacked (NNLS weights from earlier folds)", "stacked\n(NNLS)")
                 .replace("mean of all hybrids", "mean of all\n7 hybrids") for i in piv.index]
    piv.plot.bar(ax=axes[1], color=[MUTED, GOLD, ACCENT], rot=0)
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].set_title("Ensemble MAPE by fold")
    axes[1].legend(title="fold", fontsize=7.5)
    reg.save(fig, "05_stacking", "Simple averaging vs learned stacking",
             "Left: non-negative stacking weights learned walk-forward (only earlier folds' predictions). Right: "
             "fold MAPE of each ensemble (stacking has no fold-1 score because no earlier fold exists).",
             "Learned weights jump around between folds (e.g. MLP gets 38% from fold 1, 0% from folds 1–2) and "
             "the stacked ensemble is the worst ensemble. With only 2–3 folds the weights over-fit the specific "
             "market of the earlier months.",
             "Use fixed equal weights: nothing to over-fit.")

    # 06 accuracy vs cost --------------------------------------------------
    sel = summ[summ.fit_seconds.notna() & (summ.MAPE < 6)].copy()
    ens_t = sum(sel.set_index("model").loc[m, "fit_seconds"] for m in
                ["LightGBM — hybrid (Ridge + LightGBM)", "XGBoost — hybrid (Ridge + XGBoost)",
                 "Extra Trees — hybrid (Ridge + Extra Trees)"])
    ens = summ[summ.model == "Ensemble: LightGBM + XGBoost + Extra Trees (hybrids)"].assign(fit_seconds=ens_t)
    sel = pd.concat([sel, ens]).sort_values("MAPE", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    yy = np.arange(len(sel))
    ax.barh(yy, sel.fit_seconds, color=[BAD if m.startswith("Ensemble") else (ACCENT if "hybrid" in m else MUTED)
                                        for m in sel.model])
    for yi, (_, r) in zip(yy, sel.iterrows()):
        ax.text(r.fit_seconds * 1.08, yi, f"MAPE {r.MAPE:.2f}%", va="center", fontsize=7.5)
    ax.set_yticks(yy, sel.model.str.replace(" (hybrids)", "").str.replace(r" \(Ridge \+ .*\)", "", regex=True),
                  fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlim(right=sel.fit_seconds.max() * 4)
    ax.set_xlabel("average fit time per fold (seconds, log); rows ordered by MAPE, best at bottom")
    ax.set_title("Training cost of each candidate")
    reg.save(fig, "06_accuracy_vs_cost", "Accuracy vs training cost",
             "Mean MAPE against fit time on a 2-core machine. (h) = hybrid, (d) = direct.",
             f"The chosen ensemble trains in ~{ens_t:.0f} s per fold; still trivial for daily retraining.",
             "")
    print("model charts done")


if __name__ == "__main__":
    import sys
    if "--plots-only" not in sys.argv:
        main()
    plots()
