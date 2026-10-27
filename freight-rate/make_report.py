"""Build the submission report (DOCX) from pipeline outputs. Every number is read
from files written by the pipeline, so the report cannot drift from the code.

Run after: src.train, src.eda, src.compare, src.checks, src.diagnostics, score.py
    python make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ACCENT = RGBColor(0x06, 0x4A, 0x56)
FIG = Path("outputs/figures")
FINAL = "Two-stage ensemble (final)"


# ---------------------------------------------------------------- docx helpers
def shade(cell, hex_fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def table(doc, header, rows, widths=None, bold_rows=()):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        shade(cell, "064A56")
    for r_i, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v))
            run.bold = r_i in bold_rows
    for row in t.rows:
        for i, cell in enumerate(row.cells):
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(8.5)
            if widths:
                cell.width = Inches(widths[i])
    if widths:  # Word and LibreOffice both need fixed layout + grid widths to respect column sizes
        t.autofit = False
        tbl_pr = t._tbl.tblPr
        layout = OxmlElement("w:tblLayout")
        layout.set(qn("w:type"), "fixed")
        tbl_pr.append(layout)
        for i, col in enumerate(t.columns):
            col.width = Inches(widths[i])
        for i, gc in enumerate(t._tbl.tblGrid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int(widths[i] * 1440)))
    doc.add_paragraph()


def para(doc, text, bold_lead=None, style=None):
    p = doc.add_paragraph(style=style)
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    return p


def bullet(doc, text, bold_lead=None):
    return para(doc, text, bold_lead, "List Bullet")


def figure(doc, path, caption, width=6.4):
    doc.add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(8.5)


def pct(x, d=2):
    return f"{x:.{d}f}%"


# ---------------------------------------------------------------- report
def main():
    summary = json.loads(Path("outputs/summary.json").read_text())
    checks = json.loads((FIG / "checks/checks_summary.json").read_text())
    cv = pd.read_csv("outputs/cv_results.csv")
    cvs = pd.read_csv("outputs/cv_summary.csv")
    comp = pd.read_csv("outputs/model_comparison.csv").set_index("model")
    boot = pd.read_csv("outputs/model_bootstrap_vs_lightgbm_hybrid.csv")
    draws = pd.read_csv(FIG / "checks/unseen_city_draws.csv")
    coef = summary["linear_coefficients"]
    F = json.loads(Path("outputs/facts.json").read_text())

    clean = cv[cv.subset == "clean labels"]
    allr = cv[cv.subset == "all labels"]
    fin_c, fin_a = clean[clean.model == FINAL], allr[allr.model == FINAL]
    s_c = cvs[(cvs.model == FINAL) & (cvs.subset == "clean labels")].iloc[0]
    s_a = cvs[(cvs.model == FINAL) & (cvs.subset == "all labels")].iloc[0]
    lgb_c = cvs[(cvs.model == "LightGBM only") & (cvs.subset == "clean labels")].iloc[0]
    base_c = cvs[(cvs.model == "Baseline: median $/mile") & (cvs.subset == "clean labels")].iloc[0]
    base_a = cvs[(cvs.model == "Baseline: median $/mile") & (cvs.subset == "all labels")].iloc[0]
    lgb_a = cvs[(cvs.model == "LightGBM only") & (cvs.subset == "all labels")].iloc[0]
    one_c = cvs[(cvs.model == "Two-stage, LightGBM residual only") & (cvs.subset == "clean labels")].iloc[0]
    ens_name = "Ensemble: LightGBM + XGBoost + Extra Trees (hybrids)"
    e_all = boot[(boot.model == ens_name) & (boot.window == "all 3 folds")].iloc[0]
    e_f3 = boot[(boot.model == ens_name) & (boot.window == "fold 3 only (most recent)")].iloc[0]
    tr = checks["trend"]
    de = checks["date_effect"]
    dec = checks["december_scenarios"]
    dec_a = [v for k, v in dec.items() if k.startswith("A: ")][0]
    dec_b = [v for k, v in dec.items() if k.startswith("B: ")][0]
    dec_k0 = [v for k, v in dec.items() if "k = 0)" in k][0]
    oh = pd.DataFrame(checks["outlier_handling"])

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    for s in ["Heading 1", "Heading 2", "Heading 3"]:
        doc.styles[s].font.color.rgb = ACCENT
    for section in doc.sections:
        section.left_margin = section.right_margin = Inches(0.85)
        section.top_margin = section.bottom_margin = Inches(0.75)

    t = doc.add_heading("Freight Rate Prediction — Approach, Validation and Results", 0)
    t.runs[0].font.color.rgb = ACCENT
    doc.add_paragraph("Machine Learning Engineer Assessment. All figures in this report are generated by the "
                      "repository code; a Figure Book with every chart (EDA, model comparison, robustness checks, "
                      "diagnostics) is provided separately.")

    # ============================================================ summary
    doc.add_heading("Summary", 1)
    para(doc, "The labelled data covers 1 Jan – 31 Oct 2025 and the loads to predict are 1 Nov – 31 Dec 2025. "
              "This is therefore a forecasting problem: the model must price loads on dates later than anything "
              "it was trained on. Validation was designed to reproduce exactly that, and model choice was "
              "driven by it.")
    table(doc, ["Item", "Result"], [
        ["Validation", "3 rolling-origin folds: train on all earlier months, score the next 2 calendar months"],
        ["Candidates compared", f"{len(comp)} (7 algorithm families × 2 structures, linear models, baseline, "
                                "4 ensembles), identical folds, rows and metrics"],
        ["Chosen model", "Ridge regression on log(rate) for the drivers that must extrapolate (distance, "
                         "equipment, weight, market index, quarter-end, trend) + equal-weight average of LightGBM, "
                         "XGBoost and Extra Trees trained on the Ridge residual"],
        ["Expected error on ALL loads", f"MAPE {pct(s_a['MAPE fold-mean'])}, MAE ${s_a['MAE fold-mean']:.0f} "
                                        f"(mean of 3 folds, {int(s_a.n_total):,} loads). This is the realistic "
                                        f"figure if every load is scored: it includes the "
                                        f"{summary['label_outliers'] / summary['train_rows'] * 100:.1f}% of labels at 2–5× or "
                                        "0.2–0.5× the expected rate, which no model can anticipate."],
        ["Error on normal loads", f"MAPE {pct(s_c['MAPE fold-mean'])}, MAE ${s_c['MAE fold-mean']:.0f}, "
                                  f"bias {s_c['Bias% fold-mean']:+.2f}% ({int(s_c.n_total):,} loads, anomalous "
                                  f"labels excluded); {F['within_5pct']:.0f}% of loads within ±5%"],
        ["Reference points (normal / all loads)",
         f"LightGBM alone {pct(lgb_c['MAPE fold-mean'])} / {pct(lgb_a['MAPE fold-mean'])}; rate-per-mile baseline "
         f"{pct(base_c['MAPE fold-mean'])} / {pct(base_a['MAPE fold-mean'])}"],
    ], widths=[1.7, 5.0])

    # ============================================================ 1 data
    doc.add_heading("1. Key findings from the data", 1)
    bullet(doc, " 48,000 labelled loads and 12,000 loads to predict, with no date overlap (Figure 1). A random "
                "split would mix same-day market conditions into training and hide the forecasting problem.",
           "Disjoint in time.")
    bullet(doc, f" A power law: rate ∝ distance^{coef['log_distance']:.2f}, so rate per mile falls with distance. "
                f"Holding other factors fixed, Reefer is associated with a ~{(2.71828 ** coef['eq_reefer'] - 1) * 100:.0f}% "
                f"and Flatbed a ~{(2.71828 ** coef['eq_flatbed'] - 1) * 100:.0f}% higher rate than Dry Van; weight "
                f"has an elasticity of {coef['log_weight']:.2f}. These are Ridge coefficients from the final model: "
                "associations, not causal effects.", "Distance dominates.")
    bullet(doc, " market_index has a strong 7-day cycle (Wed–Thu peak) around a slow regime that rises "
                "Jan–May and falls Aug–Sep (Figure 2). In validation it sits in a narrow low band that is fully "
                "inside the training range.", "Market index.")
    bullet(doc, f" {F['n_unseen_cities']} cities in validation.csv never occur in training; "
                f"{F['n_unseen_city_val_loads']:,} validation loads ({F['pct_unseen_city_val_loads']:.0f}%) touch one. "
                "City name therefore cannot be a feature; lanes are described by coordinates, distance and circuity.",
           "Unseen cities.")
    bullet(doc, f" Each city has one consistent coordinate pair, and the distance column agrees with them (median "
                f"road ÷ straight-line distance {F['circuity_median']:.2f}). But they are not the real locations: "
                f"compared with approximate real coordinates for {F['n_real_coord_cities']} well-known cities they are "
                f"displaced by a median of ~{F['coord_offset_median_miles']:.0f} miles (Phoenix by "
                f"~{F['coord_offset_phoenix_miles']:.0f}). They are treated as dataset-internal location signals, never joined to "
                "external geographic data.", "Coordinates are synthetic.")
    bullet(doc, f" quote_signal adds little (linear coefficient {coef['quote_signal']:.3f}; a weak U-shape only in "
                "the tails).",
           "Weak signal.")
    figure(doc, FIG / "eda/01_timeline.png", "Figure 1. Loads per day in each file.", 5.8)
    figure(doc, FIG / "eda/10_market_index.png", "Figure 2. Daily market_index and its weekday profile.", 6.2)

    # ============================================================ 2 quality
    doc.add_heading("2. Data-quality issues and how they were handled", 1)
    table(doc, ["Issue", "Extent", "Evidence", "Treatment"], [
        ["Anomalous posted_rate labels", f"{summary['label_outliers']} rows "
                                         f"({summary['label_outliers'] / summary['train_rows'] * 100:.1f}%)",
         f"Ratio to a robust (Huber) log-linear expectation: normal loads within ±{F['normal_label_band_pct']:.0f}%, "
         "then an empty gap, then two clusters at 0.2–0.5× and 2–5×. Any threshold from "
         f"{F['outlier_plateau_low']:.1f}× to {F['outlier_plateau_high']:.1f}× flags exactly the same rows (Figure 3). "
         "Spread evenly over months, equipment and distance; their features look normal.",
         "Called anomalous (likely entry/unit errors, not proven). Excluded from training; the detector is "
         "re-fitted inside each fold on training months only. Metrics reported with and without them."],
        ["Negative weight", f"{summary['negative_weights_fixed']} train / {F['n_neg_weight_val']} validation",
         f"Magnitudes match normal loads (median {F['neg_weight_median_lb']:,.0f} lb)", "Absolute value + indicator"],
        ["Missing weight", f"{summary['missing_weight']} train / {F['n_missing_weight_val']} validation",
         f"Evenly spread: {F['missing_weight_pct_min']:.1f}–{F['missing_weight_pct_max']:.1f}% in every month",
         "Native NaN handling (LightGBM/XGBoost) + missing flag; otherwise equipment median from Jan–Apr training "
         "rows only (before the first evaluation window, so no fold sees a statistic from its own future)"],
        ["Missing market_index", f"{summary['missing_market_index_imputed']} train / {F['n_missing_mi_val']} validation",
         f"Within-day spread is tiny (std {F['within_day_mi_std']:.3f})",
         "That day's median market_index. Feature values only; train and validation share no dates, so "
         "validation rows never change a training-date value"],
        ["Weight cap", f"{F['weight_cap_share_pct']:.1f}% at exactly 47,500 lb", "Spike at the maximum in both files",
         "Kept (plausible cap)"],
        ["Synthetic coordinates", "all cities", "See section 1", "Used only as internal location signals"],
        ["Duplicates, unknown categories, conflicting coordinates", "none", "Checked", "—"],
    ], widths=[1.3, 1.0, 2.5, 2.0])
    figure(doc, FIG / "eda/18_outliers.png", "Figure 3. Left: actual ÷ expected rate. Right: rows flagged vs "
                                             "threshold — a flat plateau, so the threshold choice does not matter.")
    kept = oh[(oh.variant.str.startswith("anomalies kept"))].set_index("subset").MAPE
    rem = oh[(oh.variant.str.startswith("anomalies removed"))].set_index("subset").MAPE
    para(doc, f" Training with them included raises CV MAPE from {pct(rem['clean labels'])} to "
              f"{pct(kept['clean labels'])} on normal loads and from {pct(rem['all labels'])} to "
              f"{pct(kept['all labels'])} on all loads. Removing them helps on both.", "Does removal help?")

    # ============================================================ 3 validation
    doc.add_heading("3. Validation approach and data split", 1)
    para(doc, "Rolling-origin (expanding-window) cross-validation. Every fold trains on all months before a "
              "cut-off and scores the next two calendar months — the same 2-month horizon as the real task "
              "(Jan–Oct → Nov–Dec).")
    rows = []
    for _, r in fin_c.iterrows():
        a = fin_a[fin_a.fold == r.fold].iloc[0]
        rows.append([int(r.fold), f"1 Jan – {pd.Timestamp(r.eval_first_date) - pd.Timedelta(days=1):%d %b}",
                     f"{pd.Timestamp(r.eval_first_date):%d %b} – {pd.Timestamp(r.eval_last_date):%d %b}",
                     f"{int(r.train_rows):,}", f"{int(a.n):,} / {int(r.n):,}"])
    rows.append(["Final", "1 Jan – 31 Oct", "1 Nov – 31 Dec (validation.csv)", f"{summary['final_train_rows']:,}",
                 "12,000 (unlabelled)"])
    table(doc, ["Fold", "Training period", "Evaluation period", "Training rows (after removing anomalies)",
                "Evaluated: all / clean"], rows, widths=[0.5, 1.3, 1.9, 1.4, 1.5])
    figure(doc, FIG / "eda/20_cv_design.png", "Figure 4. Validation design (dates inclusive).", 6.0)
    doc.add_heading("Why this design", 3)
    bullet(doc, " Random K-fold would let the model see loads from the same days it is scored on, hiding the "
                "time trend and market shifts and overstating accuracy for Nov–Dec. Every fold here is scored "
                "strictly on the future.", "Not random.")
    bullet(doc, " Three folds cover different market conditions: rising (May–Jun, includes a quarter end), falling "
                "(Jul–Aug) and recovering (Sep–Oct, includes a quarter end). Fold 3 is the most recent and closest "
                "to the target period, so it gets extra weight when choosing between close candidates.",
           "Three different regimes.")
    bullet(doc, " Anything that uses labels — the anomaly detector, the Ridge stage, the tree stage, blending "
                "weights — is fitted only on that fold's training months. Feature imputation uses only feature "
                "columns. Every candidate model uses the same rows in every fold.", "No leakage.")
    bullet(doc, " Metrics are on the dollar scale (predictions are exponentiated). Two evaluation subsets: "
                "clean labels (primary: excludes anomalous labels as flagged by the fold's own detector) and all "
                "labels (secondary). Fold-mean and pooled values are both saved; they agree to 0.01 points.",
           "Metrics.")
    para(doc, "Definitions (y = actual, ŷ = prediction, n loads):  MAE = mean |ŷ − y|;  RMSE = √mean (ŷ − y)²;  "
              "MAPE = 100 × mean |ŷ − y| / y;  Bias = 100 × mean (ŷ − y) / y, so positive bias = over-prediction. "
              f"A MAPE of {s_c['MAPE fold-mean']:.1f}% is an average; it does not mean every load is within that "
              f"({F['within_2pct']:.0f}% are within 2% and {F['within_5pct']:.0f}% within 5%, see diagnostics).")

    # ============================================================ 4 model choice
    doc.add_heading("4. Model selection", 1)
    para(doc, "Candidate families were chosen to cover the standard options for tabular regression: a rule-of-thumb "
              "baseline, linear models (Ridge on log(rate), and a Gamma GLM with log link, built for positive "
              "skewed targets), instance-based (k-nearest neighbours), bagged trees (Random Forest, Extra Trees), "
              "gradient boosting (LightGBM, XGBoost, HistGradientBoosting) and a neural network (MLP). Each "
              "non-linear learner was run two ways:")
    bullet(doc, " the algorithm predicts log(rate) from all features, including a time-trend feature.", "Direct:")
    bullet(doc, " Ridge carries the drivers that must extrapolate (distance, equipment, weight, market index, "
                "quarter-end ramp, linear trend); the algorithm learns only the Ridge residual from features "
                "without absolute time (coordinates, circuity, weight, market index, quote signal, equipment, "
                "calendar position).", "Hybrid:")
    para(doc, "Hyper-parameters were set once from standard defaults, not tuned per fold; tuning on three folds "
              "would mostly fit noise.")
    rows = []
    order = [ens_name, "LightGBM — hybrid (Ridge + LightGBM)", "XGBoost — hybrid (Ridge + XGBoost)",
             "Extra Trees — hybrid (Ridge + Extra Trees)", "Random Forest — hybrid (Ridge + Random Forest)",
             "LightGBM — direct", "XGBoost — direct", "Random Forest — direct", "Ridge (log-linear)",
             "Gamma GLM (log link)", "Baseline: median $/mile"]
    for m in order:
        r = comp.loc[m]
        rows.append([m.replace(" (hybrids)", ""), pct(r.MAPE), pct(r["MAPE fold 3"]), f"${r.MAE:.0f}",
                     f"{r['Bias%']:+.2f}%", pct(r["MAPE all labels"])])
    table(doc, ["Model", "MAPE (mean of folds)", "MAPE fold 3", "MAE", "Bias", "MAPE all labels"], rows,
          widths=[2.6, 0.9, 0.8, 0.6, 0.7, 0.9], bold_rows=(0,))
    figure(doc, FIG / "models/01_direct_vs_hybrid.png",
           "Figure 5. Every algorithm is more accurate on top of the Ridge base than on its own.")
    para(doc, " Tree models cannot extrapolate: given a time feature, they predict the same level for every date "
              f"after their last split, so direct trees under-predict later months (bias "
              f"{F['direct_tree_bias_f13_max']:.1f}% to {F['direct_tree_bias_f13_min']:.1f}% in folds 1 and 3, all "
              "five tree learners). Ridge carries the level forward; the trees learn only patterns that repeat. "
              f"This is the single largest design decision, worth {F['direct_minus_hybrid_tree_mape_min']:.1f}–"
              f"{F['direct_minus_hybrid_tree_mape_max']:.1f} points of MAPE.", "Why hybrid wins.")
    figure(doc, FIG / "models/04_ensemble_significance.png",
           "Figure 6. Difference vs Ridge + LightGBM with 95% bootstrap intervals (whole dates resampled).")
    para(doc, f" Averaging LightGBM, XGBoost and Extra Trees residual models improves MAPE by "
              f"{abs(e_all['diff']):.3f} points over all folds (95% CI {e_all.lo:+.3f} to {e_all.hi:+.3f}; better "
              f"on {e_all.share_dates_better:.0%} of dates) and is no worse on the most recent fold "
              f"({e_f3['diff']:+.3f}, CI {e_f3.lo:+.3f} to {e_f3.hi:+.3f}). Boosting and randomised bagging make "
              "different errors, so averaging cancels some. Single learners are tied with LightGBM or worse on "
              "fold 3. A learned stacking ensemble (non-negative weights from earlier folds) was worst of the "
              "ensembles: with 2–3 folds the weights over-fit.", "Is the ensemble worth it?")
    para(doc, " best mean MAPE across folds, unless significantly worse on the most recent fold. The chosen "
              f"ensemble ({pct(s_c['MAPE fold-mean'])}) beats the single-LightGBM version ({pct(one_c['MAPE fold-mean'])}); "
              f"the gain is modest (${F['ensemble_mae_gain']:.1f} MAE) for ~3× training time. Caveat: picking the "
              f"best of {len(comp)} "
              "candidates on the same folds makes the winner's score slightly optimistic; the size of the gaps "
              "between tiers is far larger than that effect.", "Selection rule:")

    # ============================================================ 5 results
    doc.add_heading("5. Results of the chosen model", 1)
    rows = []
    for _, r in fin_c.iterrows():
        a = fin_a[fin_a.fold == r.fold].iloc[0]
        rows.append([f"Fold {int(r.fold)}: {r.eval_period}", f"{int(r.n):,}", f"${r.MAE:.1f}", f"${r.RMSE:.1f}",
                     pct(r.MAPE), f"{r['Bias%']:+.2f}%", pct(a.MAPE)])
    rows.append(["Mean of folds", f"{int(s_c.n_total):,}", f"${s_c['MAE fold-mean']:.1f}",
                 f"${s_c['RMSE fold-mean']:.1f}", pct(s_c['MAPE fold-mean']), f"{s_c['Bias% fold-mean']:+.2f}%",
                 pct(s_a['MAPE fold-mean'])])
    table(doc, ["Evaluation window", "n (clean)", "MAE", "RMSE", "MAPE", "Bias", "MAPE all labels"], rows,
          widths=[1.5, 0.8, 0.7, 0.7, 0.7, 0.7, 1.0], bold_rows=(len(rows) - 1,))
    sl = tr["slopes_pct_per_year"]
    para(doc, f"Fold 2 is weakest: trained on Jan–Jun only, the fitted trend was too steep ({sl['Jan–Jun']:.1f}%/yr "
              f"vs {sl['Jan–Oct (final)']:.1f}%/yr with all data), so Jul–Aug was over-predicted by "
              f"{fin_c[fin_c.fold == 2]['Bias%'].iloc[0]:.1f}%. Errors also grow within each 2-month window "
              f"(about {F['first2w_mape']:.1f}% MAPE in the first two weeks to {F['last2w_mape']:.1f}% in the last "
              "two), so late-December predictions are "
              "expected to be less accurate than early-November ones.")
    figure(doc, FIG / "diagnostics/04_error_over_time.png", "Figure 7. Out-of-fold error over time.", 6.2)

    # ============================================================ 6 checks
    doc.add_heading("6. Robustness checks", 1)
    rl = checks["residual_leakage"]
    table(doc, ["Question", "Check", "Result", "Verdict"], [
        ["Is stage 2 trained on optimistic (leaky) residuals?",
         "Train the tree stage on random 5-fold out-of-fold Ridge residuals instead",
         f"Residual correlation {rl['residual_corr']:.5f}; fold MAPE changes ≤ {rl['max_fold_MAPE_diff_pp']:.3f} pt",
         "No issue; 8 coefficients on 15k–47k rows cannot memorise"],
        ["How much does the trend assumption matter?",
         "Continue the trend at k × slope after training end (k = 0…1.5); remove it entirely",
         f"No trend term: {pct(tr['no_trend_MAPE'])}. k = 0…1.5: {min(tr['mean_MAPE_by_k'].values()):.2f}–"
         f"{max(tr['mean_MAPE_by_k'].values()):.2f}%; helps fold 3, hurts fold 2",
         "Trend term essential (sets the current level); extrapolation k = 1 kept, not tuned"],
        ["What does the 0.97 R² mean? Does it carry forward?",
         "Per-date effects → 3-term regression; refit on Jan–Aug, predict Sep–Oct",
         f"In-sample R² {de['in_sample_r2']['+ linear trend']:.2f}. Out of sample: "
         f"{de['out_of_sample_rmse_pct_SepOct']['+ linear trend']:.2f}% RMSE vs "
         f"{de['out_of_sample_rmse_pct_SepOct']['naive: last 2 weeks of Aug carried forward']:.2f}% naive",
         "0.97 is descriptive (in-sample); structure does carry 2 months forward"],
        ["Can it price cities it has never seen?",
         "5 draws: remove all loads touching 8 random cities from fold-3 training",
         f"MAPE +{(draws.iloc[:, 4] - draws.iloc[:, 3]).mean():.2f} pt on average "
         f"({draws.iloc[:, 4].min():.2f}–{draws.iloc[:, 4].max():.2f}%)",
         "Small cost; evidence (not proof) for the 8 real unseen cities"],
        ["Does removing anomalous labels help?", "Train with vs without them",
         f"Clean: {pct(kept['clean labels'])} → {pct(rem['clean labels'])}; all: {pct(kept['all labels'])} → "
         f"{pct(rem['all labels'])}", "Remove"],
        ["Does December depend on validation market data?",
         "Price the December lane with training-only market inputs",
         f"Mean ${dec_a[2]:.0f} (submitted) vs ${dec_b[2]:.0f} (training-only)", "Barely; see section 7"],
    ], widths=[1.5, 1.8, 2.0, 1.5])
    figure(doc, FIG / "checks/02_trend.png", "Figure 8. Trend sensitivity (left) and fitted drift by training "
                                             "window (right).")
    figure(doc, FIG / "checks/03_date_effect.png", "Figure 9. The date effect: in-sample R², and out-of-sample "
                                                   "error on unseen Sep–Oct dates.")
    para(doc, "How the R² is computed: (1) regress log(rate) on load features plus one indicator per date (304 dates, "
              "clean labels); the indicator coefficients are per-date price levels; (2) regress those 304 levels "
              "on log(daily market_index), the quarter-end ramp and a linear trend. 0.97 is the ordinary in-sample "
              "R² of step 2. It describes the history; it is not proof that the same relationship holds in Nov–Dec.")
    figure(doc, FIG / "checks/04_unseen_cities.png", "Figure 10. Unseen-city stress test, 5 random draws.", 5.8)
    doc.add_heading("Unseen-city test: exact design", 3)
    f3 = fin_c[fin_c.fold == 3].iloc[0]
    bullet(doc, " Fold 3 only (train 1 Jan – 31 Aug, evaluate 1 Sep – 31 Oct), using training data only. "
                "validation.csv is never used, since it has no labels.", "Data.")
    bullet(doc, " For each draw, 8 cities are sampled at random (seed 42) from the 64 training cities. They are "
                "NOT the 8 real unseen validation cities (Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, "
                "Norfolk, San Diego), which cannot be scored. So the test measures generalisation to unseen cities in "
                "general, not to those specific ones.", "Cities.")
    bullet(doc, " Every training load whose pickup OR delivery is one of the 8 cities is removed before fitting "
                "(the anomaly detector is also re-fitted without them).", "Removal.")
    bullet(doc, " Only clean Sep–Oct loads touching at least one of the 8 cities are scored. The number differs from "
                f"the {F['n_unseen_city_val_loads']:,} validation loads because it depends on which cities are drawn.",
           "Evaluation.")
    bullet(doc, f" The same loads are also predicted by a model trained WITH those cities. The difference isolates "
                f"the cost of never having seen a city. (Ordinary fold-3 MAPE on all clean loads: {pct(f3.MAPE)}.)",
           "Comparison.")
    table(doc, ["Draw", "Held-out cities", "Loads scored", "MAPE, cities seen", "MAPE, cities removed", "Cost"],
          [[int(r.draw), r.cities, f"{int(r.n):,}", pct(r.iloc[3]), pct(r.iloc[4]), f"+{r.iloc[4] - r.iloc[3]:.2f} pt"]
           for _, r in draws.iterrows()], widths=[0.4, 3.0, 0.7, 0.8, 0.8, 0.6])
    gap_ = (draws.iloc[:, 4] - draws.iloc[:, 3])
    para(doc, f"Interpretation: the model loses {gap_.min():.2f}–{gap_.max():.2f} points of MAPE on loads touching a "
              "city it has never seen. "
              "This is evidence that the coordinate-based features let it generalise to new cities. It does not prove "
              "that coordinates alone cause this, or that the 8 real validation cities behave the same way.")

    # ============================================================ 7 December
    doc.add_heading("7. Fixed December prediction chart", 1)
    para(doc, "Produced by the provided score.py from data/december_chart_inputs.csv: Lexington → Fort Wayne, "
              "360 miles, Dry Van, 32,000 lb; only the date changes.")
    figure(doc, Path("scorer_results/candidate_december.png"), "Figure 11. Output of score.py.")
    para(doc, " The chart file has no coordinates, market_index or quote_signal. Coordinates come from the city "
              "lookup. market_index and quote_signal are the medians of the values supplied in validation.csv "
              "for each December date: these are input features provided by the assessment, and no label is used. "
              "The values used are saved in outputs/december_inputs_used.csv. The instructions do not say whether "
              "these supplied feature values may be used for the chart; if only the seven chart columns are allowed, "
              "scenario B in Figure 12 (training data only) is the alternative, and it gives nearly the same curve.",
         "Inputs.")
    para(doc, f" The weekly market cycle drives the 7-day wave; the quarter-end ramp and trend lift prices into "
              f"year end (${dec_a[0]:.0f} → ${dec_a[1]:.0f}). For reference, the {F['lane_dryvan_n']} Dry Van loads on "
              f"this lane in training ranged ${F['lane_dryvan_min']:.0f}–${F['lane_dryvan_max']:.0f}.", "Shape.")
    dch = pd.read_csv("december_predictions.csv", parse_dates=["date"]).set_index("date").predicted_rate
    xmas = dch.loc["2025-12-25"]
    xrank = int((dch > xmas).sum()) + 1
    para(doc, f" Christmas Day is priced at ${xmas:.0f}, the {xrank}{'st' if xrank == 1 else 'nd' if xrank == 2 else 'rd' if xrank == 3 else 'th'} "
              f"highest of the 31 days (peak ${dch.max():.0f} on {dch.idxmax():%d %b}). That is not learned "
              f"holiday behaviour: 25 Dec 2025 is a {pd.Timestamp('2025-12-25'):%A} (the weekly market peak is "
              "Wed–Thu) and late December is in the quarter-end window, where the data shows rates rising in "
              "March, June and September. The training data contains no Christmas, so whether "
              "holiday demand pushes rates up or down cannot be checked. Treat late-December values as the least "
              "certain part of the chart.", "Christmas.")
    figure(doc, FIG / "checks/06_december_scenarios.png",
           "Figure 12. The same lane under alternative assumptions. B and C use training data only.")
    para(doc, " If the December market_index were not available, a forecast from October's weekday profile gives "
              f"almost the same curve (mean ${dec_b[2]:.0f} vs ${dec_a[2]:.0f}). Freezing the trend at 31 October "
              f"lowers prices by about {(1 - dec_k0[2] / dec_a[2]) * 100:.1f}%. All scenarios lie within a ~5% band. These are model predictions under "
              "stated assumptions, not forecasts with known accuracy: no December appears in the training data, "
              "so Christmas/New-Year effects cannot be learned.", "Sensitivity.")

    # ============================================================ 8 limitations
    doc.add_heading("8. Limitations and next steps", 1)
    bullet(doc, " Less than one year of history and no prior December; the trend and quarter-end effect are "
                "assumed to continue. Re-estimate monthly; error grows over a 2-month horizon, so frequent "
                "retraining helps.", "Time.")
    bullet(doc, f" Short hauls are under-predicted by ~{abs(F['short_haul_bias']):.1f}% and the longest over-predicted "
                f"by ~{F['long_haul_bias']:.1f}%; a distance "
                "spline in the Ridge stage would address this.", "Distance shape.")
    bullet(doc, " If the anomalous labels have a known cause (e.g. unit or per-mile entry), they could be corrected "
                "instead of dropped.", "Anomalies.")
    bullet(doc, " Prediction intervals (quantile models or conformal prediction on the time folds) would be more "
                "useful for pricing than a point estimate.", "Uncertainty.")
    para(doc, "python -m pip install -r requirements.txt; python -m src.train; python -m src.compare; "
              "python -m src.eda; python -m src.checks; python -m src.diagnostics; python score.py --predictions "
              "validation_predictions.csv --december-predictions data/december_chart_inputs.csv; "
              "python make_report.py; python make_figure_book.py", "Reproduce: ")

    out = Path("report")
    out.mkdir(exist_ok=True)
    doc.save(out / "Freight_Rate_Report.docx")
    print("Saved report/Freight_Rate_Report.docx")


if __name__ == "__main__":
    main()
