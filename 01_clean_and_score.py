"""
01_clean_and_score.py
Clean the TEIQue-SF Google Forms export and produce an analysis-ready dataset.

Handles:
  - multi-select cells (Google Form used checkboxes, not radio buttons)
  - Likert -> numeric mapping (5-point scale as administered)
  - reverse-keying (derived from item wording; see REVERSE_ITEMS)
  - duplicate submissions
  - careless-responding flags (straight-lining, low variance)
  - global trait EI score + tertile-based Low/Medium/High labels
"""
import re
import numpy as np
import pandas as pd

SRC = "./Trait_Emotional_Intelligence_Questionnaire___Responses_.xlsx"
OUT_DIR = "."

LIKERT = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Neutral": 3,
    "Agree": 4,
    "Strongly Agree": 5,
}
SCALE_MAX = 5

# Negatively-worded items, derived directly from item text (verify against your
# licensed TEIQue-SF scoring key before publication).
REVERSE_ITEMS = {2, 4, 5, 7, 8, 10, 12, 14, 16, 20, 23, 24, 26, 30}


def parse_cell(raw):
    """Return (value, status) for one Likert cell.

    status: 'single' | 'multi_adjacent' | 'multi_contradictory' | 'missing'
    Multi-select cells: if the chosen options are adjacent/near-adjacent
    (span <= 2 scale points) we take their mean; if they contradict
    (span >= 3, e.g. Strongly Disagree + Strongly Agree) we treat as missing.
    """
    if pd.isna(raw):
        return np.nan, "missing"
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    nums = [LIKERT[p] for p in parts if p in LIKERT]
    if not nums:
        return np.nan, "missing"
    if len(nums) == 1:
        return float(nums[0]), "single"
    if max(nums) - min(nums) >= 3:
        return np.nan, "multi_contradictory"
    return float(np.mean(nums)), "multi_adjacent"


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_excel(SRC)

    item_cols = [c for c in df.columns if re.match(r"^\s*\d+\s*\.", str(c))]
    assert len(item_cols) == 30, f"expected 30 items, found {len(item_cols)}"
    item_no = {c: int(re.match(r"^\s*(\d+)", str(c)).group(1)) for c in item_cols}
    item_cols = sorted(item_cols, key=lambda c: item_no[c])

    # ---- parse to numeric -------------------------------------------------
    raw = pd.DataFrame(index=df.index)
    status = pd.DataFrame(index=df.index)
    for c in item_cols:
        vals, sts = zip(*[parse_cell(v) for v in df[c]])
        raw[f"i{item_no[c]:02d}"] = vals
        status[f"i{item_no[c]:02d}"] = sts

    ncols = list(raw.columns)
    st_counts = pd.Series([v for v in status.values.ravel()]).value_counts()

    # ---- duplicate submissions -------------------------------------------
    dup_mask = raw.duplicated(keep="first")
    n_dup = int(dup_mask.sum())

    # ---- careless responding ---------------------------------------------
    # longest run of identical consecutive answers, and within-person SD
    def longest_run(row):
        best = run = 1
        for a, b in zip(row[:-1], row[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        return best

    filled = raw.apply(lambda r: r.fillna(r.median() if r.notna().any() else 3), axis=1)
    long_run = filled.apply(lambda r: longest_run(list(r.values)), axis=1)
    within_sd = raw.std(axis=1)

    # ---- reverse-key and score -------------------------------------------
    scored = raw.copy()
    for c in ncols:
        n = int(c[1:])
        if n in REVERSE_ITEMS:
            scored[c] = (SCALE_MAX + 1) - scored[c]

    # person-mean imputation for the few remaining missing cells
    n_missing_before = int(scored.isna().sum().sum())
    scored = scored.apply(lambda r: r.fillna(r.mean()), axis=1)

    global_mean = scored[ncols].mean(axis=1)          # 1..5, comparable across studies
    global_sum = scored[ncols].sum(axis=1)            # 30..150

    # ---- Cronbach's alpha -------------------------------------------------
    k = len(ncols)
    item_var = scored[ncols].var(ddof=1)
    alpha = (k / (k - 1)) * (1 - item_var.sum() / global_sum.var(ddof=1))

    # corrected item-total correlations
    cit = {}
    for c in ncols:
        rest = scored[ncols].drop(columns=[c]).sum(axis=1)
        cit[c] = float(np.corrcoef(scored[c], rest)[0, 1])

    # ---- labels: tertile split -------------------------------------------
    q1, q2 = global_mean.quantile([1 / 3, 2 / 3])
    label = pd.cut(global_mean, bins=[-np.inf, q1, q2, np.inf],
                   labels=["Low EI", "Medium EI", "High EI"])

    out = pd.DataFrame({"respondent_id": np.arange(1, len(df) + 1)})
    out = pd.concat([out, scored[ncols].round(4)], axis=1)
    out["ei_total"] = global_sum.round(4)
    out["ei_mean"] = global_mean.round(4)
    out["ei_class"] = label.astype(str)
    out["flag_duplicate"] = dup_mask.values
    out["flag_longest_run"] = long_run.values
    out["flag_within_sd"] = within_sd.round(4).values

    out.to_csv(f"{OUT_DIR}/teique_clean.csv", index=False)
    raw.round(4).to_csv(f"{OUT_DIR}/teique_raw_numeric.csv", index=False)

    # ---- report -----------------------------------------------------------
    print("=" * 68)
    print("TEIQue-SF CLEANING REPORT")
    print("=" * 68)
    print(f"Respondents            : {len(df)}")
    print(f"Items                  : {len(ncols)} (5-point Likert as administered)")
    print(f"\nCell parse status ({status.size} cells):")
    for k_, v in st_counts.items():
        print(f"   {k_:22s} {v:5d}  ({100*v/status.size:.2f}%)")
    print(f"\nCells missing after parse : {n_missing_before} "
          f"({100*n_missing_before/status.size:.2f}%) -> person-mean imputed")
    print(f"Exact duplicate response vectors : {n_dup}")
    print(f"Straight-lining (run >= 10)      : {int((long_run>=10).sum())}")
    print(f"Near-zero variance (SD < 0.30)   : {int((within_sd<0.30).sum())}")

    print(f"\nGlobal trait EI (item mean): M={global_mean.mean():.3f} "
          f"SD={global_mean.std(ddof=1):.3f} "
          f"min={global_mean.min():.3f} max={global_mean.max():.3f}")
    print(f"Global trait EI (sum)      : M={global_sum.mean():.2f} "
          f"SD={global_sum.std(ddof=1):.2f}")
    from scipy import stats
    print(f"Skew={stats.skew(global_mean):.3f}  Kurtosis={stats.kurtosis(global_mean):.3f}"
          f"  Shapiro p={stats.shapiro(global_mean).pvalue:.4f}")
    print(f"\nCronbach's alpha (30 items): {alpha:.4f}")

    cits = pd.Series(cit).sort_values()
    print("\nWeakest corrected item-total correlations:")
    for c, v in cits.head(6).items():
        flag = "  <-- below .20, review keying" if v < 0.20 else ""
        print(f"   {c}  r={v:+.3f}{flag}")
    print("Strongest:")
    for c, v in cits.tail(5).items():
        print(f"   {c}  r={v:+.3f}")

    print(f"\nTertile cut-points (item mean): {q1:.4f} / {q2:.4f}")
    print("\nClass distribution:")
    print(out["ei_class"].value_counts().reindex(
        ["Low EI", "Medium EI", "High EI"]).to_string())
    print("\nSaved: teique_clean.csv, teique_raw_numeric.csv")


if __name__ == "__main__":
    main()
