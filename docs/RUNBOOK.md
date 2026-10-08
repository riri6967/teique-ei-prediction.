# Runbook — order of operations

## Step 0. Environment (30 min, do this on Google Colab)
Colab is the right choice: free GPU, network access, PyTorch preinstalled.
Upload the whole folder, then:

    !pip install -q xgboost shap pytorch-tabnet
    import torch; print(torch.__version__, torch.cuda.is_available())

Local alternative: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## Step 1. Regenerate the cleaned data (1 min)
    python 01_clean_and_score.py
Confirm: n=118, alpha=0.848, tertiles 42/38/38. If these differ, stop.

## Step 2. Confirm the design is not circular (5 min)  <-- DO NOT SKIP
    python 02_diagnostics.py
    python 00_sanity_checks.py --model xgb
Gate: the permutation test must show shuffled-label accuracy near 0.333.
Verified numbers: real 0.683 vs shuffled 0.331. If shuffled scores high, you
have leakage and nothing downstream is trustworthy.

## Step 3. Verify the harness without deep models (2 min)
    python ei_pipeline.py --no-torch
Should print Tables 1-3 and write 4 figures to results/.

## Step 4. Full run (15-40 min CPU, ~10 min GPU)
    python ei_pipeline.py
Watch for: FT-Transformer / TabNet / SAINT train accuracy hitting 1.000 with
test ~0.65 (memorisation -> lower d_token, raise dropout), or a model
collapsing to one predicted class (check step 4 of sanity checks).

## Step 5. Statistics (1 min)
    python 03_significance_tests.py
This decides what you are allowed to write. It prints the exact sentence.

## Step 6. Appendix leakage ceiling
Set DESIGN = "leakage" in ei_pipeline.py, rerun, save to results_leakage/.
Report as a diagnostic, labelled as such.

## Step 7. Robustness checks reviewers will ask for
- Sensitivity: drop the 18 straight-lining and 3 duplicate respondents
  (flag_longest_run >= 10, flag_duplicate), rerun, confirm conclusions hold.
- Vary K_ITEMS over 5, 8, 10, 12, 15, 20 -> accuracy-vs-item-count figure.
  This is the short-form paper's key result.
- Two-class supplementary: Low vs High only, excluding the boundary band.
  High accuracy here proves the 3-class ceiling is a labelling artefact.
- Alternative labelling: +/-0.5 SD split instead of tertiles (gives 35/50/33).

## What "the model works" means here
1. Permutation test passes                     -> no leakage
2. Every class predicted, none collapsing      -> not degenerate
3. Train-test gap under ~0.25                  -> not memorising
4. Accuracy-when-confident > overall accuracy  -> probabilities usable
5. Errors adjacent, not Low<->High             -> psychometrically sensible
6. Accuracy rises with distance from cut-point -> failure is the label, not the model

All six are verified passing on your data with the sklearn stand-ins.
Re-confirm 2 and 3 for the deep models after Step 4.
