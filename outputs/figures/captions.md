# Figure captions and alt text

All figures regenerate with `python -m src.make_figures` from `outputs/tables/`.

## fig1_label_agreement

**Caption.** Released MMIST-LUNG 12-month labels versus source-derived 365-day status (GDC Data Release 46.0), by cohort. Counts are patients. In TCGA, every ascertainable label agrees with the source; in CPTAC, none of the 62 released early-death labels and none of the 26 source early deaths agree. All 211 patients whose status cannot be ascertained were released as survivors.

**Alt text.** Two 2-by-3 count matrices. TCGA: 134 consistent early deaths, 760 consistent survivors, 132 unknown released as survivors. CPTAC: 0 consistent early deaths, 62 released early deaths that were alive beyond 365 days, 26 source early deaths released as survivors, 172 consistent survivors, 79 unknown.

## fig2_polarity_mechanism

**Caption.** Released 12-month label by recorded day of death. In TCGA, deaths on or before day 365 carry the early-death label; in CPTAC the assignment is reversed, exactly as produced by the rule in the authors' encoding notebook (survivor if days_to_death <= 365 or missing). One CPTAC death recorded at day -1 is omitted.

**Alt text.** Two stacked histograms of death day. TCGA: orange bars only before the 365-day line, blue after. CPTAC: blue bars before the line and orange after, showing inverted polarity.

## fig3_km_censoring

**Caption.** Kaplan–Meier overall survival from GDC Data Release 46.0 with 95% CI at 365 days (Greenwood, log-log), against the fraction released as alive at 12 months (diamonds). The CPTAC released fraction (81.7%) lies below the KM estimate (90.5%) because later deaths were labelled as early deaths.

**Alt text.** Two step curves (TCGA blue, CPTAC orange) with interval bars at day 365 and hollow diamonds for the released fractions; numbers at risk below the axis.

## fig4_fixed_score_contrast

**Caption.** Experiment A. Identical out-of-fold predictions (models trained on released labels in the released benchmark folds) scored against released versus source-derived labels on the same 1154 ascertainable patients. Negative ΔAUROC means the released labels make the model look better than it is. Intervals are 2000-replicate bootstrap intervals stratified by fold and cohort, conditional on the fitted scores.

**Alt text.** Dot plot of AUROC for three models under three label sets, and a forest plot of paired AUROC differences with confidence intervals for models and sensitivity analyses.

## fig5_training_labels

**Caption.** Experiment B. Models trained on released labels (B1, benchmark practice), on released labels restricted to ascertainable patients (B2), or on source-derived labels (B3). Left: performance on held-out ascertainable patients judged against source-derived labels. Right: the same models judged by the released labels on all patients, as a benchmark would. Intervals show split-to-split variability, not full sampling uncertainty.

**Alt text.** Two dot-and-interval panels comparing three training-label variants for two models under two evaluation label sets.

## fig6_censoring_handling

**Caption.** Experiment C. The same out-of-fold scores evaluated with different treatments of the 211 patients whose 365-day status is not ascertainable: complete-case, inverse-probability-of-censoring weighting, imputing unknowns as survivors (released practice) or as deaths, sharp bounds over every possible completion, and Harrell's C-index on full overall-survival follow-up.

**Alt text.** Dot plot of AUROC by evaluation method with a gray bar showing the partial-identification bounds.

## fig7_scores_by_category

**Caption.** Distribution (median, interquartile range) of out-of-fold predicted risk from the benchmark-style logistic model, by agreement category. Both discordant groups are CPTAC patients with a recorded death and receive similar, relatively high predicted risk; which of them counts as a 'correct' prediction is decided entirely by the label set, not by the model. The figure does not by itself show that the model learned the inversion (see Experiment B for that).

**Alt text.** Horizontal interquartile bars per label category with median tick marks.

## fig8_cohort_mechanism

**Caption.** Why flexible models are affected. Released labels make CPTAC look higher-risk than TCGA (23.8% vs 15.0%), whereas source-derived labels show the opposite (10.0% vs 15.0%). Models that can identify the cohort learn whichever ordering they are trained on; the stage-only model cannot see the cohort and assigns the same mean risk to both. Post-hoc analysis.

**Alt text.** Dot plot comparing TCGA and CPTAC. Released event rates and released-trained model predictions are higher in CPTAC; source event rates and the source-trained model are lower in CPTAC; stage-only is equal.
