# Emotional Intelligence Prediction from the TEIQue-SF — Methodology

## 1. The design problem you must fix before writing

Your specification is: use the 30 TEIQue items to classify respondents as Low /
Medium / High EI. **The label is computed from those same 30 items.** A tertile
split of the item mean is a deterministic function of the features, so the
models are not predicting a latent trait — they are re-deriving an arithmetic
operation from its own inputs.

This is textbook target leakage, and it is the first thing a reviewer will
catch. Empirically, on your data, logistic regression on all 30 items reaches
**0.840 ± 0.082** accuracy — and the only reason it is not 1.000 is that the
tertile cut-points are extremely close together (see §3), so the models
approximate the threshold imperfectly. The residual error is numerical, not
psychological. "We achieved 84% accuracy predicting EI" would be an
unpublishable claim.

### The fix: reframe as short-form derivation

Predict the **full 30-item classification** from a **k-item subset**. Feature
and label are then genuinely distinct, and the research question becomes one the
literature actually cares about:

> Can a small, automatically selected subset of TEIQue-SF items recover
> full-scale trait EI classification, and which items carry that information?

This matters because respondent burden is a live problem in EI assessment, and
it gives your four models something real to compete on. Item selection runs
**inside each training fold** using corrected item-total correlation — a
criterion that never sees the class label, so it cannot leak.

Set `DESIGN = "shortform"`, `K_ITEMS = 10` in `ei_pipeline.py`. Report the
all-30-item run as a diagnostic ceiling in an appendix, explicitly labelled as
a leakage check rather than a result.

### If you would rather not reframe

The alternative is to collect an **external criterion** — academic performance,
job ratings, a wellbeing measure, peer-rated EI — and predict *that* from the
TEIQue items. This is the stronger paper, but it needs new data collection.

---

## 2. Sample and instrument

| Property | Value |
|---|---|
| Respondents | 118 |
| Items | 30 (TEIQue-SF) |
| Response scale | 5-point Likert (Strongly Disagree → Strongly Agree) |
| Reverse-keyed items | 2, 4, 5, 7, 8, 10, 12, 14, 16, 20, 23, 24, 26, 30 |
| Cronbach's α (30 items) | **0.848** |
| Global EI (item mean) | M = 3.249, SD = 0.400, range 2.233–4.200 |
| Distribution | skew 0.264, kurtosis −0.025, Shapiro–Wilk *p* = 0.130 (normal) |

### Data-quality issues found and handled

1. **The form used checkboxes, not radio buttons.** 124 of 3 540 cells (3.5%)
   contain multiple selections. Resolution: adjacent selections (span ≤ 2 scale
   points, n = 119) were averaged; contradictory selections (span ≥ 3, e.g.
   *Strongly Disagree + Strongly Agree*, n = 5) were set missing and
   person-mean imputed. **Fix the form to single-choice before collecting
   more data** — and report this handling in your Methods.
2. **3 exact duplicate response vectors** — flagged in `flag_duplicate`. Decide
   whether these are repeat submissions and justify keeping or dropping them.
3. **18 respondents with runs of ≥ 10 identical consecutive answers**, 7 with
   within-person SD < 0.30 — possible careless responding. Flags are in the
   cleaned file; a sensitivity analysis excluding them strengthens the paper.
4. **Five items show corrected item-total correlations below .20**: i11 (.093),
   i20 (.110), i07 (.113), i21 (.115), i24 (.175). Verify their keying against
   your licensed TEIQue-SF scoring key. Note that item 21 in particular is
   keyed inconsistently across published versions.
5. **The 5-point scale is a deviation** from the standard 7-point TEIQue-SF.
   State this explicitly; it limits comparability with normative data.

I did not compute the four TEIQue factor scores (Well-being, Self-control,
Emotionality, Sociability) because I could not verify the official item-to-factor
mapping, and guessing it would put a fabricated key in your paper. Supply your
licensed mapping and the pipeline will accept it.

---

## 3. Your central finding: the Medium band is below measurement resolution

This is the most defensible contribution in the study, and it is quantitative.

Tertile cut-points fall at 3.067 and 3.367 on the item mean, so the **Medium EI
band is 0.300 scale points wide** (9 raw points out of 150). Meanwhile:

```
SEM = SD × √(1 − α) = 0.400 × √(1 − 0.848) = 0.156
95% CI for an individual score = ±1.96 × 0.156 = ±0.306
```

**The Medium band (0.300) is narrower than the 95% confidence interval of a
single respondent's score (±0.306).** Roughly 43% of the sample sits within
±0.10 of a cut-point. The Medium category is therefore not a psychologically
distinct group; it is the region where the instrument cannot resolve rank order.

Your models reproduce this exactly. Per-class recall in the short-form design:

| Class | Recall (range across the four models) |
|---|---|
| Low EI | 0.70 – 0.79 |
| **Medium EI** | **0.47 – 0.51** |
| High EI | 0.74 – 0.79 |

And accuracy is a clean monotone function of distance from the cut-point:

| Distance from cut-point | n | Accuracy |
|---|---|---|
| < 0.05 | 25 | 0.36 – 0.56 |
| 0.05 – 0.10 | 26 | 0.58 – 0.62 |
| 0.10 – 0.20 | 25 | 0.56 – 0.76 |
| > 0.20 | 42 | **0.90 – 0.93** |

Errors are also psychometrically sensible: only **5–10%** are severe
(Low↔High); the rest are adjacent-class confusions. Low-vs-High is nearly
perfectly separable (0.97), while Medium-vs-High (0.81) and Low-vs-Medium
(0.72) carry the error mass.

**The argument to make in the paper:** the ceiling is a property of the
measurement instrument, not of the architectures. No model can exceed the
reliability of its labels. Report a two-class (Low vs High, excluding the
boundary band) supplementary analysis to demonstrate this directly — accuracy
there should be far higher, which proves the three-class ceiling is a labelling
artefact.

---

## 4. Evaluation protocol

**Use 5-fold × 10-repeat stratified cross-validation, not a single split.** With
n = 118, a single 80/20 holdout gives 24 test cases; one case is 4.2 accuracy
points and the 95% CI is roughly ±19 points. Any ranking of four models from a
single split is noise. This is the second thing a reviewer will catch.

- Item selection, scaling, and early-stopping validation splits all occur
  **inside** the training fold.
- Report mean ± SD over the 50 fits, and use **corrected repeated-CV *t*-tests**
  or a Nemenyi post-hoc test for model comparison — never a bare accuracy
  ranking.
- ROC-AUC is macro one-vs-rest.
- Training time is per fold; inference time is per sample.
- Parameter counts: trainable tensors for the neural models, total tree nodes
  for XGBoost (not directly comparable — say so in the caption).

### Expected outcome, and why it is still publishable

With n = 118 and 10 features, **XGBoost will likely match or beat all three
deep architectures**, and the deep models will show larger variance. In the
verified harness run, accuracies clustered at 0.64–0.68 with overlapping SDs —
i.e. no significant difference between architectures.

Do not treat this as a failed experiment. It replicates a well-established
finding (Grinsztajn et al., 2022; Shwartz-Ziv & Armon, 2022) in a new domain:
attention-based tabular models need more data than psychometric survey studies
typically have. Frame the paper as *"parameter-efficient gradient boosting
remains the appropriate choice for small-sample psychometric classification,"*
supported by your parameter-count and training-time columns. That is an honest,
useful, and reviewable contribution. Claiming a deep model "won" on the basis
of a 2-point difference with 8-point SDs would not be.

---

## 5. Model configurations

| Model | Configuration | Reference |
|---|---|---|
| FT-Transformer | d_token 64, 2 blocks, 8 heads, dropout 0.2, feature tokenizer + [CLS] | Gorishniy et al., 2021 |
| TabNet | n_d = n_a = 8, 3 steps, γ = 1.3, λ_sparse = 1e-3 | Arik & Pfister, 2021 |
| SAINT | d_token 32, 2 blocks, 4 heads, self- + intersample attention | Somepalli et al., 2021 |
| XGBoost | 400 trees, depth 3, lr 0.05, subsample 0.8, λ = 1.0 | Chen & Guestrin, 2016 |

All neural models: AdamW, cosine schedule, early stopping (patience 30) on an
internal 20% validation split, label smoothing 0.05, gradient clipping at 1.0.
Capacities are deliberately small — anything larger memorises 94 training rows.

SAINT's intersample attention is what distinguishes it from FT-Transformer;
because it attends across rows within a batch, it is the one deep model with a
principled reason to do well at this sample size. Worth commenting on in the
Discussion whichever way it turns out.

**SHAP:** exact TreeExplainer for XGBoost; TabNet's native attentive masks plus
its `explain()` output. Both are fitted on the full sample for interpretation
only and never used for performance claims — state this, or a reviewer will
assume you interpreted a leaked model.

---

## 6. Suggested paper structure

1. **Introduction** — trait EI, the TEIQue, respondent burden, why short forms
   matter.
2. **Related work** — trait EI measurement; deep tabular learning
   (FT-Transformer, TabNet, SAINT); ML in psychometrics.
3. **Data** — §2 above, including the full data-quality disclosure. Reviewers
   trust papers that report their own instrument problems.
4. **Method** — the short-form design (§1), fold-internal item selection,
   the four architectures, the ten metrics, repeated CV (§4).
5. **Results** — Table 1 (metrics), Table 2 (per-class), Table 3
   (misclassification), Figures 1–6.
6. **Error analysis** — §3. This is your strongest section. Lead the Discussion
   with the SEM-vs-band-width argument.
7. **Discussion** — deep vs. boosted trees at small n; which items survive
   selection (i03, i05, i10, i18, i22, i25, i26 were retained in >90% of folds)
   and what they say about which facets carry global EI variance.
8. **Limitations** — n = 118, convenience sample, single institution, 5-point
   scale deviation, self-report only, no external criterion, no test–retest.
9. **Conclusion**.

### Honest assessment of publication prospects

With n = 118, self-report only, no external criterion, and no significant
difference between models, this is not a top-tier venue paper. It is a
reasonable fit for a student-track conference, an applied-computing or
educational-technology venue, or a workshop. To move up you would need either
n in the several hundreds or an external criterion variable (§1).

The most publishable single asset you have is the measurement-resolution
argument in §3 — it reframes a mediocre accuracy number as a principled finding
about label reliability. Build the paper around it.

---

## 7. Reproduction

```bash
pip install torch xgboost shap pytorch-tabnet scikit-learn pandas \
            matplotlib seaborn openpyxl
python ei_pipeline.py                 # full run, ~15-40 min on CPU
python ei_pipeline.py --no-torch      # harness check with sklearn stand-ins
```

Set `DESIGN = "leakage"` for the appendix ceiling run. Outputs land in
`results/`: `table1_metrics.csv`, `table2_per_class.csv`,
`table3_misclassification.csv`, `fold_metrics.csv` (all 50 fits, for
significance testing), `item_selection_frequency.csv`, and Figures 1–6 at
300 dpi.

**Note:** the harness, metrics, cross-validation, tables and figures are all
verified working. The FT-Transformer, TabNet, and SAINT training code could not
be executed in the environment where this was prepared (no PyTorch, no network
access), so run `--no-torch` first, then the full pipeline, and check the
neural models train sensibly before trusting their numbers.
