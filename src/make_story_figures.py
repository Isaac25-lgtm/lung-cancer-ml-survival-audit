"""README story charts. Each chart's title states its takeaway; every number is read
from outputs/tables/, so the charts cannot drift from the analysis.

Saved as PNG on a white background (readable in GitHub light and dark themes) in
outputs/figures/readme/. Usage: python -m src.make_story_figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src.paths import FIGURES, TABLES

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RED, GRAY, LIGHT = "#d03b3b", "#898781", "#e1e0d9"
INK, INK2 = "#0b0b0b", "#52514e"
OUT = FIGURES / "readme"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False,
})


def t(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def _header(fig, title: str, subtitle: str, top: float = 0.80) -> None:
    fig.text(0.012, 0.975, title, fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.975 - 0.42 * (1 - top), subtitle, fontsize=9.5, color=INK2, va="top")
    fig.subplots_adjust(top=top)


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


# ---------------------------------------------------------------------------

def label_composition():
    a = t("label_agreement")
    a = a[a.source == "gdc_gt"]
    seg = {"consistent": BLUE, "inconsistent": RED, "unknown": GRAY}
    names = {"consistent": "agrees with the source", "inconsistent": "contradicts the source",
             "unknown": "cannot be checked (status at 12 months unknown)"}
    fig, ax = plt.subplots(figsize=(8.6, 3.0))
    for y, cohort in enumerate(["TCGA", "CPTAC"]):
        s = a[a.cohort == cohort]
        ok = s[((s.released == "released early death") & (s.source_state == "early death")) |
               ((s.released == "released survivor") & (s.source_state == "alive >365d"))].n.sum()
        unk = s[s.source_state == "unknown"].n.sum()
        n = s.n.sum()
        bad = n - ok - unk
        left = 0.0
        for k, v in [("consistent", ok), ("inconsistent", bad), ("unknown", unk)]:
            w = v / n
            if v:
                ax.barh(y, w - 0.004, left=left, color=seg[k], height=0.62, lw=0)
                ax.text(left + w / 2, y, f"{v:,}\n{w:.0%}", ha="center", va="center", color="white",
                        fontsize=10, fontweight="bold", linespacing=1.05)
            left += w
    ax.set_yticks([0, 1], [f"TCGA\n1,026 patients", f"CPTAC\n339 patients"], fontsize=10.5)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=seg[k]) for k in seg]
    fig.legend(handles, [names[k] for k in seg], loc="lower left", ncol=3, fontsize=9,
               bbox_to_anchor=(0.012, -0.05), handlelength=1, columnspacing=1.4)
    _header(fig, "In CPTAC, only half of the 12-month labels can be confirmed",
            "Released label compared with each patient's 365-day status in the upstream GDC record", top=0.76)
    fig.subplots_adjust(bottom=0.12)
    _save(fig, "1_label_composition")


def opposite_rules():
    h = t("fig2_death_time_histogram")
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 4.7), sharex=True, gridspec_kw={"hspace": 0.32})
    notes = {"TCGA": "Died within a year: labelled early death", "CPTAC": "Died within a year: labelled survivor"}
    for ax, cohort in zip(axes, ["TCGA", "CPTAC"]):
        for name, col in [("released: early death", ORANGE), ("released: survivor", BLUE)]:
            s = h[(h.cohort == cohort) & (h.released == name)]
            ax.bar(s.bin_start + 2, s.n, width=56, align="edge", color=col, lw=0)
        top = h[h.cohort == cohort].groupby("bin_start").n.sum().max()
        ax.set_ylim(0, top * 1.32)
        ax.axvline(365, color=INK, lw=1, ls="--")
        ax.text(385, top * 1.30, notes[cohort], fontsize=10, fontweight="bold", color=INK, va="top")
        ax.set_title(cohort, loc="left", color=INK, fontsize=11, fontweight="bold")
        ax.set_ylabel("Patients")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))
    axes[1].set_xlabel("Recorded days from diagnosis to death (365 marked; later values grouped in the last bar)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    fig.legend(handles, ["Released label: early death", "Released label: survivor"], loc="lower left",
               ncol=2, fontsize=9, bbox_to_anchor=(0.012, -0.045), handlelength=1)
    _header(fig, "One label, two meanings: deaths are coded in opposite directions",
            "Every patient with a recorded death in the benchmark, by day of death and released label", top=0.83)
    _save(fig, "2_opposite_rules")


def hidden_in_average():
    k = t("km_365").set_index("group")
    order = ["All", "TCGA LUAD", "TCGA LUSC", "CPTAC LUAD", "CPTAC LUSC"]
    names = {"All": "All patients", "TCGA LUAD": "TCGA adenocarcinoma", "TCGA LUSC": "TCGA squamous",
             "CPTAC LUAD": "CPTAC adenocarcinoma", "CPTAC LUSC": "CPTAC squamous"}
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    for i, g in enumerate(order):
        r = k.loc[g]
        km, rel = r.km_survival_365, r.released_survivor_fraction
        ax.plot([r.km_ci_lo, r.km_ci_hi], [i, i], color=BLUE, lw=5, alpha=0.22, solid_capstyle="round")
        ax.plot([km, rel], [i, i], color=INK2, lw=1)
        ax.plot(km, i, "o", color=BLUE, ms=9, label="Kaplan–Meier estimate from source records (95% CI band)" if i == 0 else None)
        ax.plot(rel, i, "D", color=ORANGE, ms=8, label="Share labelled alive at 12 months in the benchmark" if i == 0 else None)
        gap = 100 * (rel - km)
        ax.text(1.035, i, f"{gap:+.1f} pts".replace("-", "\u2212"), va="center", ha="right", fontsize=10,
                color=INK if abs(gap) > 5 else INK2, fontweight="bold" if abs(gap) > 5 else "normal")
    ax.set_yticks(range(len(order)), [names[g] for g in order], fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0.72, 1.04)
    ax.set_xticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Patients alive at 12 months")
    ax.grid(axis="x", color=LIGHT, lw=0.7)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", fontsize=9, ncol=1,
               bbox_to_anchor=(0.012, -0.135))
    _header(fig, "The overall survival rate looks right only because the errors offset",
            "Close for all patients and TCGA; far apart in CPTAC, where recorded deaths are inverted", top=0.80)
    fig.subplots_adjust(left=0.24)
    _save(fig, "3_hidden_in_average")


def cohort_flip():
    m = t("exp_a_cohort_mechanism")
    first = m[m.scores == m.scores.iloc[0]].set_index("cohort")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 4.1), sharey=True, gridspec_kw={"wspace": 0.42})

    def line(ax, tc, cp, col, lab, ls="-", dy=0.0, dy0=0.0):
        ax.plot([0, 1], [tc, cp], color=col, lw=2.4, ls=ls, marker="o", ms=8, clip_on=False)
        ax.text(-0.05, tc + dy0, f"{tc:.1%}", ha="right", va="center", fontsize=9.5, color=INK2)
        ax.text(1.05, cp + dy, f"{cp:.1%}  {lab}", ha="left", va="center", fontsize=9.5, color=INK, fontweight="bold")

    line(a1, first.at["TCGA", "released_event_rate"], first.at["CPTAC", "released_event_rate"], ORANGE, "released")
    line(a1, first.at["TCGA", "source_event_rate"], first.at["CPTAC", "source_event_rate"], BLUE, "source")
    a1.set_title("Early-death rate in the data", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    g = {k: v.set_index("cohort").mean_predicted_risk for k, v in m.groupby("scores")}
    r = g["logistic, trained on released"]
    line(a2, r["TCGA"], r["CPTAC"], ORANGE, "trained on released", dy0=-0.0045)
    r = g["stage only, trained on released"]
    line(a2, r["TCGA"], r["CPTAC"], GRAY, "stage only", dy0=-0.0005)
    r = g["logistic, trained on source"]
    line(a2, r["TCGA"], r["CPTAC"], BLUE, "trained on source", dy0=0.0045)
    a2.set_title("Average risk the model predicts", loc="left", fontsize=10.5, color=INK, fontweight="bold")
    for ax in (a1, a2):
        ax.set_xlim(-0.32, 1.0)
        ax.set_xticks([0, 1], ["TCGA", "CPTAC"], fontsize=10.5)
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0.07, 0.26)
    _header(fig, "The labels flip which cohort looks riskier, and models learn the flip",
            "1,154 patients with known 12-month status. The clinical features identify the cohort perfectly. Post-hoc analysis.",
            top=0.76)
    _save(fig, "4_cohort_flip")


def wrong_winner():
    b = t("exp_b_benchmark_folds")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), sharey=True, gridspec_kw={"wspace": 0.95})
    for ax, (model, title) in zip(axes, [("logistic", "Logistic regression"), ("hgb", "Gradient boosting")]):
        for key, col, lab in [("released", ORANGE, "judged by\nreleased labels"), ("source", BLUE, "judged by\nsource labels")]:
            r = b[(b.model == model) & b.evaluation.str.startswith(key)].iloc[0]
            y0, y1 = r.auroc_trained_on_released, r.auroc_trained_on_source
            ax.plot([0, 1], [y0, y1], color=col, lw=2.4, marker="o", ms=8, clip_on=False)
            ax.text(-0.05, y0, f"{y0:.3f}", ha="right", va="center", fontsize=9.5, color=INK2)
            ax.text(1.05, y1, f"{y1:.3f}  {lab}\nΔ {r.delta:+.3f} ({r.ci_lo:+.3f} to {r.ci_hi:+.3f})", ha="left",
                    va="center", fontsize=8.8, color=INK, linespacing=1.15)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
        ax.set_xlim(-0.3, 1.0)
        ax.set_xticks([0, 1], ["Trained on\nreleased labels", "Trained on\nsource labels"], fontsize=9.5)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(0.54, 0.67)
    _header(fig, "Judged by the released labels, the model trained on source labels looks worse",
            "AUROC in the released folds, with 95% bootstrap CI for each difference. Post-hoc; models not refitted in the bootstrap.\n"
            "Source labels are scored on the 1,154 patients with known status; released labels on all 1,365, as a benchmark would.",
            top=0.74)
    _save(fig, "5_wrong_winner")


def primary_contrast():
    s = t("exp_sensitivity")
    short = [("primary", "Primary analysis (pre-specified)"), ("boundary", "Death on day 365 counted as survivor"),
             ("source: TCGA", "Labels from TCGA-CDR + LinkedOmics"), ("source: consensus", "Only where source releases agree"),
             ("folds: released", "Fold indices in the released code's order"), ("exclude", "Excluding flagged records"),
             ("folds: new", "New stratified fold split"), ("CPTAC only", "CPTAC only (26 early deaths)")]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    for i, (k, lab) in enumerate(short):
        r = s[s.sensitivity.str.startswith(k)].iloc[0]
        crosses = r.ci_lo <= 0 <= r.ci_hi
        col = GRAY if crosses else BLUE
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=2.4, solid_capstyle="round")
        ax.plot(r.delta, i, "o", color=col, ms=9 if i == 0 else 7)
        ax.text(0.40, i, f"{r.delta:+.3f}  ({r.ci_lo:+.3f} to {r.ci_hi:+.3f})".replace("-", "\u2212"), va="center", ha="right", fontsize=9,
                color=INK if i == 0 else INK2, fontweight="bold" if i == 0 else "normal")
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks(range(len(short)), [x[1] for x in short], fontsize=9.5)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(-0.24, 0.40)
    ax.set_xticks([-0.2, -0.1, 0, 0.1])
    ax.set_xlabel("Change in AUROC: source-derived minus released labels")
    fig.text(0.012, -0.045, "Blue: 95% interval excludes zero.   Gray: interval includes zero.", fontsize=9, color=INK2)
    _header(fig, "The pre-specified test: a small effect that does not hold up everywhere",
            "Logistic regression, predictions held fixed, identical patients", top=0.84)
    fig.subplots_adjust(left=0.30)
    _save(fig, "6_primary_contrast")


def three_datasets():
    """Label integrity across the three audited resources (MMIST-LUNG, MMIST-ccRCC, I3LUNG)."""
    rows = []
    a = t("label_agreement")
    a = a[a.source == "gdc_gt"]
    for cohort in ["TCGA", "CPTAC"]:
        x = a[a.cohort == cohort]
        ok = x[((x.released == "released early death") & (x.source_state == "early death")) |
               ((x.released == "released survivor") & (x.source_state == "alive >365d"))].n.sum()
        unk_s = x[(x.source_state == "unknown") & (x.released == "released survivor")].n.sum()
        unk_d = x[(x.source_state == "unknown") & (x.released == "released early death")].n.sum()
        rows.append((f"MMIST-LUNG  {cohort}", ok, x.n.sum() - ok - unk_s - unk_d, unk_s, unk_d, 0))
    p = TABLES / "v2_ccrcc_label_agreement.csv"
    if p.exists():
        c = pd.read_csv(p)
        for cohort, lab in [("TCGA", "TCGA-KIRC"), ("CPTAC", "CPTAC-CCRCC")]:
            x = c[c.cohort == cohort]
            ok = x[((x.released == "released early death") & (x.source_state == "early death")) |
                   ((x.released == "released survivor") & (x.source_state == "alive >365d"))].n.sum()
            unk_s = x[(x.source_state == "unknown") & (x.released == "released survivor")].n.sum()
            unk_d = x[(x.source_state == "unknown") & (x.released == "released early death")].n.sum()
            rows.append((f"MMIST-ccRCC  {lab.split('-')[0]}", ok, x.n.sum() - ok - unk_s - unk_d, unk_s, unk_d, 0))
    i3 = t("i3lung_label_audit")
    agree = i3[(i3.label == "OS_6") & i3.check.str.contains("1 = alive")].value.iloc[0]
    k, n = (int(v) for v in agree.split("/"))
    miss = int(i3[(i3.label == "OS_6") & (i3.check == "label missing")].value.iloc[0])
    rows.append(("I3LUNG  (6-month)", k, n - k, 0, 0, miss))
    cols = [("agrees with source", BLUE), ("contradicts source", RED), ("unknown, labelled survivor", GRAY),
            ("unknown, labelled death", ORANGE), ("unknown, left blank", LIGHT)]
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    for i, r in enumerate(rows):
        tot = sum(r[1:])
        left = 0.0
        for j, (name, col) in enumerate(cols):
            v = r[1 + j]
            w = v / tot
            if v:
                ax.barh(i, w - 0.003, left=left, color=col, height=0.66, lw=0)
                if w > 0.045:
                    ax.text(left + w / 2, i, f"{v:,}", ha="center", va="center", fontsize=9,
                            color=INK if col == LIGHT else "white", fontweight="bold")
            left += w
        ax.text(1.01, i, f"n = {tot:,}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in cols]
    fig.legend(handles, [n for n, _ in cols], loc="lower left", ncol=3, fontsize=8.8, bbox_to_anchor=(0.012, -0.07),
               handlelength=1, columnspacing=1.2)
    _header(fig, "Three resources, three different label stories",
            "Released label vs 12-month status in GDC (MMIST datasets) or vs the dataset's own time and event fields (I3LUNG)", top=0.80)
    fig.subplots_adjust(left=0.20, bottom=0.14)
    _save(fig, "0_three_datasets")


def run() -> None:
    for f in [three_datasets, label_composition, opposite_rules, hidden_in_average, cohort_flip, wrong_winner, primary_contrast]:
        f()


if __name__ == "__main__":
    run()
    print("story figures ->", OUT)
