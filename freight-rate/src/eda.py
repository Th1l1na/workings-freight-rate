"""Exploratory data analysis: one chart per question, each with a recorded finding.

Run from the repository root:
    python -m src.eda
Figures -> outputs/figures/eda/, captions -> outputs/figures/eda/captions.json
"""
from __future__ import annotations

import matplotlib.dates
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from .features import build_features, clean, daily_market_index, haversine_miles
from .model import LINEAR_FEATURES, fit_outlier_detector
from .plotting import (ACCENT, BAD, EQUIP_COLORS, GOLD, GOOD, INK, MUTED, TRAIN_C, VALID_C,
                       FigureRegistry)

# Approximate real-world coordinates of well-known cities, used only to show
# that the dataset's coordinates are synthetic (not used by the model).
REAL_COORDS = {
    "Phoenix": (33.45, -112.07), "Los Angeles": (34.05, -118.24), "Dallas": (32.78, -96.80),
    "Atlanta": (33.75, -84.39), "Houston": (29.76, -95.37), "New York": (40.71, -74.01),
    "Boston": (42.36, -71.06), "Philadelphia": (39.95, -75.17), "Kansas City": (39.10, -94.58),
    "San Francisco": (37.77, -122.42), "Lexington": (38.04, -84.50), "Fort Wayne": (41.08, -85.14),
    "Green Bay": (44.51, -88.01), "Salt Lake City": (40.76, -111.89), "Tampa": (27.95, -82.46),
    "Detroit": (42.33, -83.05), "Las Vegas": (36.17, -115.14), "Memphis": (35.15, -90.05),
}


def load():
    tr = pd.read_csv("data/train_test.csv")
    va = pd.read_csv("data/validation.csv")
    raw_tr, raw_va = tr.copy(), va.copy()
    mi = daily_market_index(tr, va)
    tr, va = clean(tr, mi), clean(va, mi)
    for df in (raw_tr, raw_va):
        df["date"] = pd.to_datetime(df["date"])
    return raw_tr, raw_va, tr, va, mi


def main(out_dir="outputs/figures/eda"):
    reg = FigureRegistry(out_dir, "EDA")
    raw_tr, raw_va, tr, va, mi = load()
    tr["rpm"] = tr["posted_rate"] / tr["distance"]
    feats = build_features(tr, tr.groupby("equipment")["weight"].median().to_dict())
    log_y = np.log(tr["posted_rate"])
    det = fit_outlier_detector(feats, log_y)
    ratio = np.exp(log_y - det.predict(feats[LINEAR_FEATURES]))
    out = (ratio > 1.6) | (ratio < 1 / 1.6)
    c = tr[~out]

    # 01 timeline -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(tr.groupby("date").size(), color=TRAIN_C, lw=1, label=f"train_test.csv  ({len(tr):,} loads)")
    ax.plot(va.groupby("date").size(), color=VALID_C, lw=1, label=f"validation.csv  ({len(va):,} loads)")
    ax.set_ylabel("Loads per day")
    ax.set_ylim(0)
    ax.set_title("Training and validation data do not overlap in time")
    ax.legend(loc="lower left")
    reg.save(fig, "01_timeline", "Data coverage over time",
             "Number of loads per day in each file.",
             f"Training covers {tr.date.min():%d %b} – {tr.date.max():%d %b %Y}; validation covers "
             f"{va.date.min():%d %b} – {va.date.max():%d %b %Y}. No overlap; volumes are steady (~160/day "
             "train, ~200/day validation).",
             "Validation must be time-based (train on the past, score the future). A random split would leak "
             "same-day market conditions into training.")

    # 02 missingness --------------------------------------------------------
    cols = [c_ for c_ in raw_tr.columns if c_ not in ("load_id", "posted_rate")]
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(cols))
    ax.bar(x - 0.2, raw_tr[cols].isna().mean() * 100, 0.4, color=TRAIN_C, label="train")
    ax.bar(x + 0.2, raw_va[cols].isna().mean() * 100, 0.4, color=VALID_C, label="validation")
    ax.set_xticks(x, cols, rotation=40, ha="right")
    ax.set_ylabel("% missing")
    ax.set_title("Only weight and market_index have missing values (<2.1%)")
    ax.legend()
    reg.save(fig, "02_missing", "Missing values by column",
             "Share of missing values per column in each file.",
             f"weight: {raw_tr.weight.isna().sum()} train / {raw_va.weight.isna().sum()} validation missing; "
             f"market_index: {raw_tr.market_index.isna().sum()} / {raw_va.market_index.isna().sum()}. "
             "Every other column is complete, including the target.",
             "No rows dropped for missing features. market_index -> that day's median (it is a market-wide "
             "signal); weight -> left as NaN for LightGBM (native handling) plus a missing flag, and equipment "
             "median for the linear stage.")

    # 03 target distribution -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(tr.posted_rate.clip(upper=8000), bins=120, color=ACCENT)
    axes[0].text(0.98, 0.9, f"{(tr.posted_rate > 8000).sum()} loads > $8,000\nshown at the edge",
                 transform=axes[0].transAxes, ha="right", fontsize=7.5)
    axes[0].set_title("posted_rate ($): right-skewed")
    axes[0].set_xlabel("$")
    axes[1].hist(np.log(tr.posted_rate), bins=120, color=ACCENT)
    axes[1].set_title("log(posted_rate)")
    axes[1].set_xlabel("log $")
    reg.save(fig, "03_target", "Target distribution",
             "Histogram of posted_rate on the dollar scale and on the log scale.",
             f"Range ${tr.posted_rate.min():,.0f} – ${tr.posted_rate.max():,.0f}, median "
             f"${tr.posted_rate.median():,.0f}. Strongly right-skewed with a long tail of extreme values.",
             "Model log(rate): errors become relative (a $50 miss matters more on a $300 load than a $3,000 "
             "one), multiplicative effects (equipment premium, market index) become additive, and the tail "
             "has less leverage.")

    # 04 rate vs distance ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    for eq, col in EQUIP_COLORS.items():
        m = (c.equipment == eq)
        ax.scatter(c.distance[m], c.posted_rate[m], s=1.5, alpha=0.15, color=col, label=eq)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance (miles, log)")
    ax.set_ylabel("Posted rate ($, log)")
    ax.set_title("Rate is almost a straight line in distance on log–log axes")
    ax.legend(markerscale=8)
    slope = np.polyfit(np.log(c.distance), np.log(c.posted_rate), 1)[0]
    reg.save(fig, "04_rate_vs_distance", "Rate vs distance (log–log)",
             "Each point is one load (flagged anomalous labels removed), coloured by equipment.",
             f"A straight line on log–log axes means a power law: rate ∝ distance^{slope:.2f}. Distance alone "
             f"explains {np.corrcoef(np.log(c.distance), np.log(c.posted_rate))[0, 1] ** 2:.0%} of the "
             "variance in log(rate).",
             "log(distance) is the primary feature. A power below 1 means rate per mile falls with distance.")

    # 05 rate per mile ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.array([70, 150, 250, 400, 600, 900, 1300, 1800, 2400, 3500])
    mid = np.sqrt(bins[:-1] * bins[1:])
    for eq, col in EQUIP_COLORS.items():
        s = c[c.equipment == eq]
        g = s.groupby(pd.cut(s.distance, bins), observed=True).rpm
        ax.plot(mid, g.median().values, "o-", color=col, label=eq)
        ax.fill_between(mid, g.quantile(.1).values, g.quantile(.9).values, color=col, alpha=0.08)
    ax.set_xscale("log")
    ax.set_xlabel("Distance (miles, log)")
    ax.set_ylabel("$ per mile")
    ax.set_title("Rate per mile falls with distance; Reefer > Flatbed > Dry Van at every distance")
    ax.legend()
    med = c.groupby("equipment").rpm.median()
    reg.save(fig, "05_rate_per_mile", "Rate per mile by distance and equipment",
             "Median $/mile per distance band (line) with the 10th–90th percentile band (shaded).",
             f"Median $/mile: Dry Van {med['Dry Van']:.2f}, Flatbed {med['Flatbed']:.2f}, Reefer "
             f"{med['Reefer']:.2f}. The ordering holds at every distance, so equipment acts as a roughly "
             "constant % premium.",
             "Equipment enters as a multiplicative premium (dummy variables on log scale).")

    # 06 equipment mix ------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    eqs = list(EQUIP_COLORS)
    x = np.arange(3)
    axes[0].bar(x - 0.2, tr.equipment.value_counts(normalize=True)[eqs] * 100, 0.4, color=TRAIN_C, label="train")
    axes[0].bar(x + 0.2, va.equipment.value_counts(normalize=True)[eqs] * 100, 0.4, color=VALID_C, label="validation")
    axes[0].set_xticks(x, eqs)
    axes[0].set_ylabel("% of loads")
    axes[0].set_title("Equipment mix is identical")
    axes[0].legend()
    axes[1].boxplot([c.rpm[c.equipment == e] for e in eqs], tick_labels=eqs, showfliers=False,
                    medianprops={"color": BAD})
    axes[1].set_title("$/mile by equipment")
    reg.save(fig, "06_equipment", "Equipment mix and price level",
             "Left: share of each equipment type per file. Right: $/mile distribution (outlier labels removed).",
             "Mix is ~57% Dry Van, 25% Reefer, 18% Flatbed in both files; no category appears only in validation.",
             "No re-weighting needed for equipment.")

    # 07 weight -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(raw_tr.weight.dropna(), bins=120, color=TRAIN_C)
    axes[0].axvline(0, color=INK, lw=0.8)
    axes[0].set_title("Raw weight: negative values exist")
    axes[0].set_xlabel("lb")
    axes[1].hist(tr.weight.dropna(), bins=100, color=TRAIN_C, alpha=0.7, density=True, label="train")
    axes[1].hist(va.weight.dropna(), bins=100, color=VALID_C, alpha=0.5, density=True, label="validation")
    axes[1].set_title("After abs(): one population, capped at 47,500")
    axes[1].set_xlabel("lb")
    axes[1].legend()
    neg = raw_tr.weight < 0
    reg.save(fig, "07_weight", "Weight: sign errors and a cap",
             "Left: raw weight including negatives. Right: weight after taking the absolute value.",
             f"{neg.sum()} train and {(raw_va.weight < 0).sum()} validation weights are negative, with magnitudes "
             f"(median {(-raw_tr.weight[neg]).median():,.0f} lb) matching normal loads ⇒ sign errors, not a "
             f"separate population. {(tr.weight == 47500).mean():.1%} of loads sit exactly at 47,500 lb (a cap).",
             "Take abs(weight) and keep a 'was negative' flag. Keep the capped values (a legal-limit cap is "
             "plausible and the model sees it as 'heaviest').")

    # 08 weight effect -----------------------------------------------------
    resid = np.log(c.posted_rate) - np.polyval(np.polyfit(np.log(c.distance), np.log(c.posted_rate), 1),
                                               np.log(c.distance))
    fig, ax = plt.subplots(figsize=(8, 3))
    wb = pd.cut(c.weight, np.arange(0, 50001, 2500))
    g = resid.groupby(wb, observed=True).mean()
    ax.plot([i.mid for i in g.index], (np.exp(g.values) - 1) * 100, "o-", color=ACCENT)
    ax.set_xlabel("Weight (lb)")
    ax.set_ylabel("% above/below distance-only price")
    ax.set_title("Heavier loads cost more: about −5% at 10k lb to +3% at 45k lb")
    reg.save(fig, "08_weight_effect", "Effect of weight on price",
             "Mean residual of a distance-only fit, by weight band (shown as % difference).",
             "Smooth, monotonic, roughly logarithmic relationship. Weight is a real but modest driver.",
             "Use log(weight) in the linear stage and raw weight in the tree stage.")

    # 09 numeric train vs validation ----------------------------------------
    num = ["distance", "weight", "market_index", "quote_signal"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, col in zip(axes, num):
        lo, hi = np.nanpercentile(pd.concat([tr[col], va[col]]), [0.2, 99.8])
        b = np.linspace(lo, hi, 50)
        ax.hist(tr[col].dropna(), bins=b, density=True, color=TRAIN_C, alpha=0.6, label="train")
        ax.hist(va[col].dropna(), bins=b, density=True, color=VALID_C, alpha=0.5, label="validation")
        ax.set_title(col, fontsize=10)
        ax.set_yticks([])
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Train vs validation feature distributions", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    reg.save(fig, "09_shift", "Train vs validation: covariate shift",
             "Density of each numeric feature in train_test.csv and validation.csv.",
             f"distance, weight and quote_signal match closely. market_index is the exception: validation sits "
             f"in a narrow low band ({va.market_index.min():.2f}–{va.market_index.max():.2f}) vs "
             f"{tr.market_index.min():.2f}–{tr.market_index.max():.2f} in training — but fully inside the "
             "training range.",
             "Validation inputs are covered by training data, so no feature has to be extrapolated except time "
             "itself.")

    # 10 market index time series --------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [3, 1]})
    axes[0].plot(mi.index, mi.values, color=MUTED, lw=0.8)
    axes[0].plot(mi.rolling(7, center=True).mean(), color=ACCENT, lw=2, label="7-day mean")
    axes[0].axvline(pd.Timestamp("2025-11-01"), color=INK, ls="--", lw=0.8)
    axes[0].text(pd.Timestamp("2025-11-04"), 1.38, "validation", fontsize=8)
    axes[0].set_title("Daily market_index: weekly cycle around a slow regime")
    axes[0].legend(loc="lower left")
    dow = pd.DataFrame({"mi": mi, "dow": mi.index.dayofweek, "set": np.where(mi.index >= "2025-11-01", "val", "train")})
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for s, col in [("train", TRAIN_C), ("val", VALID_C)]:
        d = dow[dow.set == s]
        prof = d.groupby("dow").mi.mean() / d.mi.mean()
        axes[1].plot(days, prof.values, "o-", color=col, label=s)
    axes[1].set_title("Weekday profile (÷ mean)")
    axes[1].legend()
    reg.save(fig, "10_market_index", "Market index over time",
             "Left: daily median market_index (grey) and 7-day mean. Right: average by weekday relative to the "
             "period mean.",
             "Two components: (1) a slow regime — rises Jan–May, plateaus, falls Aug–Sep, flat Oct–Dec; (2) a "
             "very regular 7-day cycle peaking Wed–Thu (±12%) that is identical in train and validation. "
             f"Within a single day loads differ by only ~{tr.groupby('date').market_index.std().mean():.3f}.",
             "Treat market_index as a market-wide daily signal: impute by daily median and also give the model "
             "each load's deviation from it.")

    # 11 quote signal -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(tr.quote_signal, bins=100, color=ACCENT)
    axes[0].set_title("quote_signal distribution")
    qb = pd.qcut(c.quote_signal, 20)
    g = resid.groupby(qb, observed=True).mean()
    axes[1].plot([i.mid for i in g.index], (np.exp(g.values) - 1) * 100, "o-", color=ACCENT)
    axes[1].set_title("Price effect (% vs distance-only)")
    axes[1].set_xlabel("quote_signal")
    reg.save(fig, "11_quote_signal", "quote_signal",
             "Left: distribution. Right: mean residual of a distance-only fit by quote_signal vigintile.",
             f"Correlation with $/mile is only {np.corrcoef(c.quote_signal, c.rpm)[0, 1]:.2f}. The effect is "
             "U-shaped: almost flat (about −2% to +1%) for the middle 80% of loads, rising to +4–8% only in the "
             "thin tails (quote_signal < 1.6 or > 2.5). A linear term cannot capture a U-shape.",
             "Keep it: its linear coefficient is near zero, but the tree stage can learn the U-shape.")

    # 12 correlation heatmap -----------------------------------------------
    m = pd.DataFrame({"log_rate": np.log(c.posted_rate), "log_distance": np.log(c.distance),
                      "rate_per_mile": c.rpm, "weight": c.weight, "market_index": c.market_index,
                      "quote_signal": c.quote_signal, "reefer": (c.equipment == "Reefer") * 1.0,
                      "flatbed": (c.equipment == "Flatbed") * 1.0, "pickup_lon": c.pickup_lon,
                      "delivery_lon": c.delivery_lon, "day_of_year": c.date.dt.dayofyear})
    corr = m.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=50, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else INK)
    fig.colorbar(im, shrink=0.7)
    ax.set_title("Correlation matrix (clean labels)")
    reg.save(fig, "12_correlations", "Correlations between features and target",
             "Pearson correlation between numeric features, log(rate) and rate per mile.",
             f"log_distance dominates log_rate (r = {corr.loc['log_rate', 'log_distance']:.2f}). "
             " Among the rest, market_index, weight and equipment "
             "correlate with $/mile; features are not strongly correlated with each other, so multicollinearity "
             "is not a concern for the linear stage.",
             "")

    # 13 city map -----------------------------------------------------------
    both = pd.concat([tr.assign(s="train"), va.assign(s="val")])
    cities = (both.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
              .rename(columns={"pickup_lat": "lat", "pickup_lon": "lon"}))
    unseen = sorted(set(va.pickup) - set(tr.pickup))
    fig, ax = plt.subplots(figsize=(9, 5))
    seen = cities.drop(unseen)
    ax.scatter(seen.lon, seen.lat, s=22, color=ACCENT, label="in training")
    ax.scatter(cities.loc[unseen].lon, cities.loc[unseen].lat, s=60, color=BAD, marker="*",
               label="only in validation (unseen)")
    for name, r in cities.iterrows():
        ax.annotate(name, (r.lon, r.lat), fontsize=6.3, xytext=(3, 2), textcoords="offset points",
                    color=BAD if name in unseen else INK)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Dataset city coordinates; 8 cities appear only in validation")
    ax.legend(loc="lower left")
    n_unseen_loads = int((va.pickup.isin(unseen) | va.delivery.isin(unseen)).sum())
    reg.save(fig, "13_city_map", "City locations and unseen cities",
             "Each city's coordinates as given in the data. Stars = cities that never appear in training.",
             f"{len(cities)} cities. Unseen in training: {', '.join(unseen)}. {n_unseen_loads:,} validation loads "
             f"({n_unseen_loads / len(va):.0%}) touch at least one of them.",
             "City name cannot be a model feature (no training rows for 8 of them). Describe lanes by "
             "coordinates, distance and circuity instead, which exist for every city.")

    # 14 real vs dataset coordinates ---------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    offs = []
    for name, (lat, lon) in REAL_COORDS.items():
        if name not in cities.index:
            continue
        dl = cities.loc[name]
        ax.annotate("", xy=(dl.lon, dl.lat), xytext=(lon, lat),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
        ax.scatter(lon, lat, s=14, color=INK, zorder=3)
        ax.scatter(dl.lon, dl.lat, s=14, color=BAD, zorder=3)
        ax.annotate(name, (lon, lat), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
        offs.append(haversine_miles(lat, lon, dl.lat, dl.lon))
    ax.scatter([], [], color=INK, s=14, label="real location (approx.)")
    ax.scatter([], [], color=BAD, s=14, label="location in dataset")
    ax.legend(loc="lower left")
    ax.set_title("Dataset coordinates are displaced from the real cities")
    circ = tr.distance / haversine_miles(tr.pickup_lat, tr.pickup_lon, tr.delivery_lat, tr.delivery_lon)
    reg.save(fig, "14_coordinates_vs_real", "Are the coordinates real?",
             "Arrows from each city's approximate real location to where the dataset puts it (18 well-known "
             "cities).",
             f"Displacements are large (median {np.median(offs):.0f} miles; Phoenix is moved "
             f"~{offs[0]:.0f} miles). So coordinates are synthetic/perturbed, not real geography. However each "
             f"city has exactly one coordinate pair, and the distance column is consistent with them (median "
             f"distance ÷ great-circle = {circ.median():.2f}, a typical road-circuity factor).",
             "Treat coordinates as dataset-internal location signals, not real geography. Do not join external "
             "data (e.g. real regional rate indices) on them. This corrects the earlier report wording.")

    # 15 lane coverage ------------------------------------------------------
    lanes_tr = tr.groupby(["pickup", "delivery"]).size()
    va_seen = va.apply(lambda r: (r.pickup, r.delivery) in lanes_tr.index, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].hist(lanes_tr.values, bins=range(0, 42), color=ACCENT)
    axes[0].set_title(f"Loads per lane in training ({len(lanes_tr):,} lanes)")
    axes[0].set_xlabel("loads on the lane")
    counts = va_seen.map({True: "lane seen in training", False: "lane never seen"}).value_counts()
    axes[1].barh(counts.index, counts.values, color=[ACCENT, BAD][:len(counts)])
    axes[1].set_title("Validation loads by lane history")
    reg.save(fig, "15_lanes", "Lane coverage",
             "Left: how many training loads each pickup→delivery lane has. Right: validation loads on lanes "
             "with/without training history.",
             f"Median lane has only {int(lanes_tr.median())} loads, and {1 - va_seen.mean():.0%} of validation "
             "loads are on lanes never seen in training.",
             "Lane-level target encoding would be noisy (few loads per lane) and unavailable for 12% of "
             "validation. Use smooth geographic features that generalise between lanes.")

    # 16 circuity -----------------------------------------------------------
    gc = haversine_miles(tr.pickup_lat, tr.pickup_lon, tr.delivery_lat, tr.delivery_lon)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.scatter(gc, circ, s=1, alpha=0.1, color=ACCENT)
    ax.set_xscale("log")
    ax.set_ylim(1, 2)
    ax.set_xlabel("great-circle distance (miles, log)")
    ax.set_ylabel("distance ÷ great-circle")
    ax.set_title("Road distance is ~1.18× straight-line; short lanes are less direct")
    reg.save(fig, "16_circuity", "Circuity (road vs straight-line distance)",
             "Ratio of the distance column to the straight-line distance between the coordinates.",
             f"Median {circ.median():.2f}; rises above 1.4 only for short lanes (<150 miles straight-line). "
             "No impossible values (ratio < 1).",
             "The distance column is trustworthy. Circuity is added as a feature for the tree stage.")

    # 17 weekday & month ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    rd = pd.Series(resid.values, index=c.index)
    wd = rd.groupby(c.date.dt.dayofweek).mean()
    axes[0].bar(days, (np.exp(wd.values) - 1) * 100, color=ACCENT)
    axes[0].set_title("Price by weekday (% vs distance-only)")
    mo = rd.groupby(c.date.dt.month).mean()
    axes[1].bar(mo.index, (np.exp(mo.values) - 1) * 100, color=ACCENT)
    axes[1].set_xticks(range(1, 11), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O"])
    axes[1].set_title("Price by month (% vs distance-only)")
    reg.save(fig, "17_weekday_month", "Calendar patterns in price",
             "Mean residual of a distance-only fit by weekday and by month.",
             "Weekday pattern mirrors the market_index weekly cycle (Wed–Thu highest). Month pattern rises "
             "through spring, peaks in June, dips in Aug–Sep and recovers in Oct.",
             "Both are largely explained by market_index; the remainder is studied in the date-effect "
             "analysis (validation section).")

    # 18 outliers -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].hist(np.log2(ratio), bins=200, color=ACCENT)
    axes[0].set_yscale("log")
    for th in [1.6, 1 / 1.6]:
        axes[0].axvline(np.log2(th), color=BAD, ls="--", lw=0.8)
    axes[0].set_xticks([-3, -2, -1, 0, 1, 2, 3], ["⅛×", "¼×", "½×", "1×", "2×", "4×", "8×"])
    axes[0].set_xlabel("actual ÷ expected rate (log scale)")
    axes[0].set_ylabel("loads (log)")
    axes[0].set_title("Empty gaps separate normal and anomalous labels")
    ths = np.linspace(1.05, 4, 120)
    n_flag = [int(((ratio > t) | (ratio < 1 / t)).sum()) for t in ths]
    axes[1].plot(ths, n_flag, color=ACCENT)
    axes[1].axvline(1.6, color=BAD, ls="--", lw=0.8)
    axes[1].set_xlabel("threshold (× expected)")
    axes[1].set_ylabel("loads flagged")
    axes[1].set_title("Any threshold 1.2×–2× flags the same rows")
    plateau = [t for t, n in zip(ths, n_flag) if n == out.sum()]
    reg.save(fig, "18_outliers", "Anomalous price labels",
             "Left: ratio of each actual rate to a robust (Huber) log-linear expectation. Right: number of "
             "loads flagged as the threshold changes.",
             f"Normal loads all fall within ±{(ratio[~out].max() - 1) * 100:.0f}% of expectation. Then there is an "
             f"empty gap, and {out.sum()} loads ({out.mean():.1%}) sit at 0.2–0.5× or 2–5×. Every threshold "
             f"from {min(plateau):.2f}× to {max(plateau):.2f}× flags exactly these rows. They are spread evenly "
             "across months, equipment types and distances, and their features look normal.",
             "Treat them as anomalous labels (likely data-entry or unit errors, but not proven) and exclude them "
             "from training; the threshold choice does not matter. Still report metrics with them included.")

    # 19 December lane history ---------------------------------------------
    lane = tr[(tr.pickup == "Lexington") & (tr.delivery == "Fort Wayne")]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for eq, col in EQUIP_COLORS.items():
        s = lane[lane.equipment == eq]
        ax.scatter(s.date, s.posted_rate, color=col, s=22, label=eq)
    ax.set_ylabel("posted rate ($)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title(f"Lexington → Fort Wayne history: {len(lane)} loads")
    reg.save(fig, "19_december_lane", "History of the December chart lane",
             "Every training load on Lexington → Fort Wayne (the fixed lane in the December chart).",
             f"{len(lane)} loads, ${lane.posted_rate.min():.0f}–${lane.posted_rate.max():.0f}, mixed equipment "
             f"and weights (dry van only: {(lane.equipment == 'Dry Van').sum()} loads, "
             f"${lane[lane.equipment == 'Dry Van'].posted_rate.min():.0f}–"
             f"${lane[lane.equipment == 'Dry Van'].posted_rate.max():.0f}).",
             "Sanity reference for the December chart: predictions around $820–$860 for a 32,000 lb dry van "
             "sit inside this historical range.")
    # 20 validation design --------------------------------------------------
    folds = [("Fold 1", "2025-05-01", "2025-07-01"), ("Fold 2", "2025-07-01", "2025-09-01"),
             ("Fold 3", "2025-09-01", "2025-11-01"), ("Final", "2025-11-01", "2026-01-01")]
    fig, ax = plt.subplots(figsize=(8.5, 2.5))
    start = pd.Timestamp("2025-01-01")
    for i, (name, a, b) in enumerate(folds):
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        yv = len(folds) - 1 - i
        ax.barh(yv, (a - start).days, left=start, color=ACCENT, height=0.55)
        ax.barh(yv, (b - a).days, left=a, color=BAD if name != "Final" else INK, height=0.55)
        lab = f"{a:%d %b} – {(b - pd.Timedelta(days=1)):%d %b}"
        ax.text(a + (b - a) / 2, yv, lab, ha="center", va="center", fontsize=7, color="white")
        ax.text(start + (a - start) / 2, yv, f"train: 01 Jan – {(a - pd.Timedelta(days=1)):%d %b}", ha="center",
                va="center", fontsize=7, color="white")
    ax.set_yticks(range(len(folds)), [f[0] for f in folds][::-1])
    ax.grid(False)
    ax.set_title("Rolling-origin validation: train on the past (teal), score the next 2 months (orange)")
    reg.save(fig, "20_cv_design", "Validation design",
             "Three expanding-window folds plus the final fit. Dates are inclusive.",
             "Each fold reproduces the real task: everything before a cut-off is training data, the next two "
             "calendar months are scored. The final model is trained on all of Jan 1 – Oct 31 and predicts "
             "Nov 1 – Dec 31, the same 2-month horizon.",
             "Same folds, rows and metrics for every model compared; outlier detection is re-fitted inside each "
             "fold on training months only.")
    print("EDA done")


if __name__ == "__main__":
    main()
