"""Publication figures, regenerated from aggregate tables in outputs/tables/.

Palette: validated 3-slot categorical set (blue/orange/aqua) + neutral gray for
'unknown'; diverging blue (consistent) vs red (inconsistent). Every figure is
saved as SVG, PDF and 300-dpi PNG with a caption + alt text in
outputs/figures/captions.md.

Usage: python -m src.make_figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from src.paths import FIGURES, INTERIM, TABLES
from src.stats_utils import km_curve

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, GRAY, LIGHTGRAY = "#d03b3b", "#898781", "#e1e0d9"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "legend.frameon": False, "svg.fonttype": "none",
    "figure.dpi": 100, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

CAPTIONS: list[tuple[str, str, str]] = []


def _save(fig, name: str, caption: str, alt: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    fig.savefig(FIGURES / f"{name}.png", dpi=300)
    plt.close(fig)
    CAPTIONS.append((name, caption, alt))


def t(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


# ---------------------------------------------------------------------------

def fig1_agreement():
    a = t("label_agreement")
    a = a[a.source == "gdc_gt"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"wspace": 0.38})
    cols = ["early death", "alive >365d", "unknown"]
    rows = ["released early death", "released survivor"]
    for ax, cohort in zip(axes, ["TCGA", "CPTAC"]):
        s = a[a.cohort == cohort].groupby(["released", "source_state"]).n.sum()
        n = int(s.sum())
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                v = int(s.get((r, c), 0))
                consistent = (r == "released early death" and c == "early death") or \
                             (r == "released survivor" and c == "alive >365d")
                col = GRAY if c == "unknown" else (BLUE if consistent else RED)
                alpha = 0.18 if v == 0 else 0.85
                ax.add_patch(plt.Rectangle((j + 0.04, 1 - i + 0.04), 0.92, 0.92, color=col, alpha=alpha, lw=0))
                ax.text(j + 0.5, 1 - i + 0.5, f"{v}", ha="center", va="center",
                        color="white" if v and alpha > 0.5 else INK, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5, 2.5], ["Early death\n(≤365 d)", "Alive\n>365 d", "Not\nascertainable"])
        ax.set_yticks([1.5, 0.5], ["Released:\nearly death", "Released:\nsurvivor"])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{cohort} (n={n})", loc="left", color=INK)
    fig.supxlabel("Source-derived 365-day status (GDC Data Release 46.0)", fontsize=9, color=INK2, y=-0.09)
    fig.text(0.5, -0.16, "Blue = consistent  ·  Red = inconsistent  ·  Gray = outcome not ascertainable at 365 days",
             color=INK2, fontsize=8, ha="center")
    _save(fig, "fig1_label_agreement",
          "Released MMIST-LUNG 12-month labels versus source-derived 365-day status (GDC Data Release 46.0), "
          "by cohort. Counts are patients. In TCGA, every ascertainable label agrees with the source; in CPTAC, "
          "none of the 62 released early-death labels and none of the 26 source early deaths agree. "
          "All 211 patients whose status cannot be ascertained were released as survivors.",
          "Two 2-by-3 count matrices. TCGA: 134 consistent early deaths, 760 consistent survivors, 132 unknown "
          "released as survivors. CPTAC: 0 consistent early deaths, 62 released early deaths that were alive "
          "beyond 365 days, 26 source early deaths released as survivors, 172 consistent survivors, 79 unknown.")


def fig2_mechanism():
    L = pd.read_csv(INTERIM / "linked.csv", low_memory=False)
    d = L[L["demographic.days_to_death"].notna() & (L["demographic.days_to_death"] >= 0)]
    bins = np.arange(0, 2200, 60)
    hist_rows = []
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True)
    for ax, cohort in zip(axes, ["TCGA", "CPTAC"]):
        s = d[d.cohort == cohort]
        for lab, col, name in [(1, ORANGE, "released: early death"), (0, BLUE, "released: survivor")]:
            v = s.loc[s.y_released_death == lab, "demographic.days_to_death"].clip(upper=2159)
            h, _ = np.histogram(v, bins=bins)
            ax.bar(bins[:-1] + 1, h, width=56, align="edge", color=col, label=f"{name} (n={len(v)})", lw=0)
            hist_rows += [{"cohort": cohort, "released": name, "bin_start": b, "n": int(x)} for b, x in zip(bins[:-1], h)]
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.axvline(365, color=INK, lw=1, ls="--")
        ax.text(375, ax.get_ylim()[1] * 0.92, "365 days", color=INK, fontsize=8)
        ax.set_title(f"{cohort}: recorded deaths (n={len(s)})", loc="left", color=INK)
        ax.set_ylabel("Patients")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].set_xlabel("Recorded days from diagnosis to death (MMIST demographics.csv; values >2160 binned at end)")
    pd.DataFrame(hist_rows).to_csv(TABLES / "fig2_death_time_histogram.csv", index=False)
    _save(fig, "fig2_polarity_mechanism",
          "Released 12-month label by recorded day of death. In TCGA, deaths on or before day 365 carry the "
          "early-death label; in CPTAC the assignment is reversed, exactly as produced by the rule in the authors' "
          "encoding notebook (survivor if days_to_death <= 365 or missing). One CPTAC death recorded at day -1 is "
          "omitted.",
          "Two stacked histograms of death day. TCGA: orange bars only before the 365-day line, blue after. "
          "CPTAC: blue bars before the line and orange after, showing inverted polarity.")


def fig3_km():
    L = pd.read_csv(INTERIM / "linked.csv", low_memory=False)
    km = t("km_365")
    grid = np.arange(0, 1826, 5)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    rows = []
    for cohort, col in [("TCGA", BLUE), ("CPTAC", ORANGE)]:
        s = L[L.cohort == cohort]
        S, risk = km_curve(s.os_time_gdc.to_numpy(), s.os_event_gdc.to_numpy(), grid)
        ax.step(grid, S, where="post", color=col, lw=2)
        k = km[km.group == cohort].iloc[0]
        ax.errorbar([365], [k.km_survival_365], yerr=[[k.km_survival_365 - k.km_ci_lo], [k.km_ci_hi - k.km_survival_365]],
                    color=col, capsize=3, lw=1.2)
        ax.plot([365], [k.released_survivor_fraction], marker="D", ms=7, mfc="white", mec=col, mew=2)
        ax.text(1830, S[-1], f"{cohort}\nKM 12-mo: {k.km_survival_365:.1%}\nreleased: {k.released_survivor_fraction:.1%}",
                color=INK2, fontsize=8, va="center")
        rows += [{"cohort": cohort, "day": int(g), "km_survival": float(x), "at_risk": int(r)} for g, x, r in zip(grid, S, risk)]
    ax.axvline(365, color=GRAY, lw=0.8, ls="--")
    ax.set_ylim(0.3, 1.01)
    ax.set_xlim(0, 1825)
    ax.set_xlabel("Days from diagnosis (GDC Data Release 46.0 overall survival)")
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan–Meier overall survival; diamond = released fraction labelled alive at 12 months", loc="left",
                 color=INK, fontsize=9)
    rt = pd.DataFrame(rows)
    rt.to_csv(TABLES / "km_curves.csv", index=False)
    ticks = [0, 365, 730, 1095, 1460, 1825]
    ax.set_xticks(ticks)
    trans = ax.get_xaxis_transform()
    ax.text(-0.05, -0.25, "At risk", ha="right", fontsize=7, color=INK2, fontweight="bold",
            transform=ax.transAxes)
    for i, cohort in enumerate(["TCGA", "CPTAC"]):
        r = rt[(rt.cohort == cohort) & rt.day.isin(ticks)].set_index("day").at_risk
        y = -0.32 - 0.07 * i
        ax.text(-0.05, y, cohort, ha="right", fontsize=7, color=INK2, transform=ax.transAxes)
        for d0 in ticks:
            ax.text(d0, y, str(r.get(d0, "")), ha="center", fontsize=7, color=INK2, transform=trans)
    _save(fig, "fig3_km_censoring",
          "Kaplan–Meier overall survival from GDC Data Release 46.0 with 95% CI at 365 days (Greenwood, log-log), "
          "against the fraction released as alive at 12 months (diamonds). The CPTAC released fraction (81.7%) "
          "lies below the KM estimate (90.5%) because later deaths were labelled as early deaths.",
          "Two step curves (TCGA blue, CPTAC orange) with interval bars at day 365 and hollow diamonds for the "
          "released fractions; numbers at risk below the axis.")


def fig4_primary():
    ad = t("exp_a_paired_deltas")
    am = t("exp_a_metrics")
    am = am[(am.cohort == "All") & (am.analysis == "primary")]
    sens = t("exp_sensitivity")
    models = [("logistic", "Logistic regression"), ("hgb", "Gradient boosting"), ("stage_only", "Stage only")]
    evals = [("released labels, all patients (benchmark setting)", "Released labels, all 1365", GRAY),
             ("released labels, ascertainable patients", "Released labels, ascertainable", ORANGE),
             ("source-derived labels, ascertainable patients", "Source-derived labels, ascertainable", BLUE)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.8), gridspec_kw={"width_ratios": [1, 1.1]})
    for i, (m, mname) in enumerate(models):
        for j, (ev, en, col) in enumerate(evals):
            v = am[(am.model == m) & (am.evaluation == ev)].auroc_foldavg.iloc[0]
            a1.plot(v, i + (j - 1) * 0.22, "o", color=col, ms=7, label=en if i == 0 else None)
    a1.axvline(0.5, color=LIGHTGRAY, lw=1)
    a1.set_xlim(0.49, 0.66)
    a1.set_yticks(range(len(models)), [m[1] for m in models])
    a1.invert_yaxis()
    a1.set_xlabel("AUROC (fold-averaged, benchmark folds)")
    a1.set_title("A. Same out-of-fold scores, three label sets", loc="left", color=INK)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.5, ncol=1)
    # panel B: deltas with CI
    rows = []
    for m, mname in models:
        r = ad[(ad.model == m) & (ad.metric == "AUROC") & (ad.analysis == "primary")].iloc[0]
        rows.append((mname, r.delta, r.ci_lo, r.ci_hi, BLUE if m == "logistic" else INK2))
    short = {"boundary": "Day-365 death = survivor", "source: TCGA": "TCGA-CDR + LinkedOmics labels",
             "source: consensus": "Consensus of source releases", "exclude": "Excluding flagged cases",
             "TCGA only": "TCGA only", "CPTAC only": "CPTAC only",
             "folds: released": "Folds in ascending-ID order", "folds: new": "New stratified folds"}
    for _, r in sens.iterrows():
        if r.sensitivity.startswith("primary"):
            continue
        name = next(v for k, v in short.items() if r.sensitivity.startswith(k))
        rows.append(("Logistic: " + name, r.delta, r.ci_lo, r.ci_hi, GRAY))
    for i, (lab, d, lo, hi, col) in enumerate(rows):
        a2.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
        a2.plot(d, i, "o", color=col, ms=6)
    a2.axvline(0, color=INK, lw=0.8)
    a2.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.5)
    a2.invert_yaxis()
    a2.set_xlabel("ΔAUROC: source-derived − released (95% CI)")
    a2.set_title("B. Paired difference on identical patients", loc="left", color=INK)
    fig.tight_layout()
    _save(fig, "fig4_fixed_score_contrast",
          "Experiment A. Identical out-of-fold predictions (models trained on released labels in the released "
          "benchmark folds) scored against released versus source-derived labels on the same 1154 ascertainable "
          "patients. Negative ΔAUROC means the released labels make the model look better than it is. Intervals "
          "are 2000-replicate bootstrap intervals stratified by fold and cohort, conditional on the fitted scores.",
          "Dot plot of AUROC for three models under three label sets, and a forest plot of paired AUROC "
          "differences with confidence intervals for models and sensitivity analyses.")


def fig5_training():
    b = t("exp_b_summary")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    order = sorted(b.variant.unique())
    short = {v: v.split(" ", 1)[1].replace("training patients", "train").replace("(benchmark practice)", "[benchmark]")
             for v in order}
    for ax, (metric, title) in zip(axes, [("auroc_source_known", "Evaluated on source-derived labels"),
                                          ("auroc_released_all", "Evaluated on released labels (benchmark lens)")]):
        for k, (model, col) in enumerate([("logistic", BLUE), ("hgb", ORANGE)]):
            sub = b[b.model == model].set_index("variant")
            for i, v in enumerate(order):
                y = i + (k - 0.5) * 0.25
                ax.plot([sub.at[v, f"{metric}_p2.5"], sub.at[v, f"{metric}_p97.5"]], [y, y], color=col, lw=2)
                ax.plot(sub.at[v, f"{metric}_mean"], y, "o", color=col, ms=6,
                        label=("Logistic regression" if model == "logistic" else "Gradient boosting") if i == 0 else None)
        ax.set_title(title, loc="left", color=INK, fontsize=9)
        ax.set_xlabel("AUROC")
    axes[0].set_yticks(range(len(order)), [f"{v[:2]}: {short[v]}" for v in order], fontsize=7.5)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, loc="upper center", ncol=2,
               bbox_to_anchor=(0.6, 1.08))
    fig.tight_layout()
    fig.text(0.6, -0.04, "Mean and 2.5–97.5 percentile over 50 repeats of 5-fold CV", ha="center", fontsize=8, color=INK2)
    _save(fig, "fig5_training_labels",
          "Experiment B. Models trained on released labels (B1, benchmark practice), on released labels restricted "
          "to ascertainable patients (B2), or on source-derived labels (B3). Left: performance on held-out "
          "ascertainable patients judged against source-derived labels. Right: the same models judged by the "
          "released labels on all patients, as a benchmark would. Intervals show split-to-split variability, not "
          "full sampling uncertainty.",
          "Two dot-and-interval panels comparing three training-label variants for two models under two "
          "evaluation label sets.")


def fig6_censoring():
    c = t("exp_c_censoring")
    c = c[c.scores == "logistic trained on source-derived labels"]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    labels = []
    for i, (_, r) in enumerate(c.iterrows()):
        if pd.notna(r.get("auroc_lo")):
            ax.plot([r.auroc_lo, r.auroc_hi], [i, i], color=GRAY, lw=6, alpha=0.5, solid_capstyle="butt")
            ax.text(r.auroc_hi + 0.01, i, f"{r.auroc_lo:.2f}–{r.auroc_hi:.2f}", va="center", fontsize=7.5, color=INK2)
        elif pd.notna(r.get("c_index")):
            ax.plot(r.c_index, i, "s", color=AQUA, ms=7)
            ax.text(r.c_index + 0.01, i, f"{r.c_index:.3f}", va="center", fontsize=7.5, color=INK2)
        else:
            ax.plot(r.auroc, i, "o", color=BLUE, ms=7)
            ax.text(r.auroc + 0.01, i, f"{r.auroc:.3f}", va="center", fontsize=7.5, color=INK2)
        labels.append(f"{r.method} (n={int(r.n_used)})")
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0.5, color=LIGHTGRAY, lw=1)
    ax.set_xlim(0.42, 0.82)
    ax.set_xlabel("AUROC at 365 days (C-index for the time-to-event row)")
    ax.set_title("Experiment C: handling of not-ascertainable outcomes (logistic, trained on source labels)",
                 loc="left", color=INK, fontsize=9)
    _save(fig, "fig6_censoring_handling",
          "Experiment C. The same out-of-fold scores evaluated with different treatments of the 211 patients whose "
          "365-day status is not ascertainable: complete-case, inverse-probability-of-censoring weighting, "
          "imputing unknowns as survivors (released practice) or as deaths, sharp bounds over every possible "
          "completion, and Harrell's C-index on full overall-survival follow-up.",
          "Dot plot of AUROC by evaluation method with a gray bar showing the partial-identification bounds.")


def fig7_scores():
    s = t("score_summary_by_label_category")
    order = ["concordant survivor", "concordant early death", "source survivor, released early death",
             "source early death, released survivor", "not ascertainable"]
    s = s.set_index("category").reindex([o for o in order if o in set(s.category)]).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    for i, r in s.iterrows():
        col = BLUE if r.category.startswith("concordant") else (GRAY if r.category == "not ascertainable" else RED)
        ax.plot([r.q25, r.q75], [i, i], color=col, lw=6, solid_capstyle="butt", alpha=0.85)
        ax.plot(r.median_score, i, "|", color=INK, ms=14, mew=2)
    ax.set_yticks(range(len(s)), [f"{c} (n={n})" for c, n in zip(s.category, s.n)], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Out-of-fold predicted probability of early death (logistic, trained on released labels): median and IQR")
    ax.set_title("Model scores by label category", loc="left", color=INK)
    _save(fig, "fig7_scores_by_category",
          "Distribution (median, interquartile range) of out-of-fold predicted risk from the benchmark-style "
          "logistic model, by agreement category. Both discordant groups are CPTAC patients with a recorded death "
          "and receive similar, relatively high predicted risk; which of them counts as a 'correct' prediction is "
          "decided entirely by the label set, not by the model. The figure does not by itself show that the model "
          "learned the inversion (see Experiment B for that).",
          "Horizontal interquartile bars per label category with median tick marks.")


def fig8_mechanism():
    m = t("exp_a_cohort_mechanism")
    first = m[m.scores == m.scores.iloc[0]].set_index("cohort")
    rows = [("Event rate, released labels", first.at["TCGA", "released_event_rate"], first.at["CPTAC", "released_event_rate"], ORANGE, "s"),
            ("Event rate, source-derived labels", first.at["TCGA", "source_event_rate"], first.at["CPTAC", "source_event_rate"], BLUE, "s")]
    names = {"logistic, trained on released": "Predicted: logistic, trained on released",
             "gradient boosting, trained on released": "Predicted: boosting, trained on released",
             "stage only, trained on released": "Predicted: stage only, trained on released",
             "logistic, trained on source": "Predicted: logistic, trained on source"}
    for k, lab in names.items():
        g = m[m.scores == k].set_index("cohort")
        col = BLUE if "trained on source" in k else (GRAY if "stage" in k else ORANGE)
        rows.append((lab, g.at["TCGA", "mean_predicted_risk"], g.at["CPTAC", "mean_predicted_risk"], col, "o"))
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, (lab, tc, cp, col, mk) in enumerate(rows):
        ax.plot([tc, cp], [i, i], color=col, lw=1.5, alpha=0.6)
        ax.plot(tc, i, marker=mk, ms=8, mfc="white", mec=col, mew=2, ls="none", label="TCGA" if i == 0 else None)
        ax.plot(cp, i, marker=mk, ms=8, color=col, ls="none", label="CPTAC" if i == 0 else None)
        ax.text(max(tc, cp) + 0.006, i, f"TCGA {tc:.1%} · CPTAC {cp:.1%}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.07, 0.31)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Early-death rate or mean predicted risk, ascertainable patients (n=1154)")
    ax.set_title("Hollow = TCGA, filled = CPTAC", loc="left", color=INK, fontsize=9)
    _save(fig, "fig8_cohort_mechanism",
          "Why flexible models are affected. Released labels make CPTAC look higher-risk than TCGA (23.8% vs 15.0%), "
          "whereas source-derived labels show the opposite (10.0% vs 15.0%). Models that can identify the cohort learn "
          "whichever ordering they are trained on; the stage-only model cannot see the cohort and assigns the same mean "
          "risk to both. Post-hoc analysis.",
          "Dot plot comparing TCGA and CPTAC. Released event rates and released-trained model predictions are higher "
          "in CPTAC; source event rates and the source-trained model are lower in CPTAC; stage-only is equal.")


def write_captions():
    lines = ["# Figure captions and alt text", "",
             "All figures regenerate with `python -m src.make_figures` from `outputs/tables/`.", ""]
    for name, cap, alt in CAPTIONS:
        lines += [f"## {name}", "", f"**Caption.** {cap}", "", f"**Alt text.** {alt}", ""]
    (FIGURES / "captions.md").write_text("\n".join(lines), encoding="utf-8")


def run():
    CAPTIONS.clear()
    for f in [fig1_agreement, fig2_mechanism, fig3_km, fig4_primary, fig5_training, fig6_censoring, fig7_scores, fig8_mechanism]:
        f()
    write_captions()


if __name__ == "__main__":
    run()
    print("figures ->", FIGURES)
