"""
03_significance_tests.py
Turn ei_pipeline.py's 50 fold results into defensible statistical claims.

Run AFTER ei_pipeline.py. Reads results/fold_metrics.csv.

Why this is mandatory: repeated CV folds overlap, so a standard paired t-test on
them is anti-conservative and will hand you "significant" differences that are
not real. Two correct options are implemented:

  A. Corrected resampled t-test (Nadeau & Bengio, 2003) for pairwise comparison,
     which inflates the variance to account for the train-set overlap.
  B. Friedman omnibus test + Nemenyi post-hoc, the standard for comparing
     multiple classifiers (Demsar, 2006).

Also reports bootstrap CIs and per-fold win/loss counts.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

METRIC = "accuracy"      # change to f1 / mcc / kappa as needed
N_SPLITS, N_REPEATS = 5, 10
TEST_FRAC = 1 / N_SPLITS

res = pd.read_csv("results/fold_metrics.csv")
models = [m for m in ["FT-Transformer", "TabNet", "SAINT", "XGBoost"]
          if m in res.model.unique()]
piv = res.pivot_table(index="fold", columns="model", values=METRIC)[models]
n_folds = len(piv)

print("=" * 76)
print(f"MODEL COMPARISON on {METRIC}   ({n_folds} folds, "
      f"{N_SPLITS}x{N_REPEATS} repeated stratified CV)")
print("=" * 76)


def bootstrap_ci(x, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, (n_boot, len(x)), replace=True).mean(1)
    return np.percentile(bs, [2.5, 97.5])


print(f"\n{'Model':<16}{'Mean':>8}{'SD':>8}{'95% CI (bootstrap)':>26}")
for m in models:
    x = piv[m].dropna().values
    lo, hi = bootstrap_ci(x)
    print(f"{m:<16}{x.mean():>8.4f}{x.std(ddof=1):>8.4f}"
          f"{f'[{lo:.4f}, {hi:.4f}]':>26}")

# ---------------------------------------------------------------------------
# A. Corrected resampled t-test (Nadeau & Bengio)
# ---------------------------------------------------------------------------
print("\n" + "=" * 76)
print("A. CORRECTED RESAMPLED PAIRED t-TEST  (Nadeau & Bengio, 2003)")
print("=" * 76)
print("   Variance inflated by (1/n + test_frac/(1-test_frac)) to account for")
print("   overlapping training sets across repeated folds.\n")


def corrected_ttest(a, b, n, test_frac):
    d = a - b
    mean_d = d.mean()
    var_d = d.var(ddof=1)
    if var_d == 0:
        return mean_d, np.nan, 1.0
    denom = np.sqrt(var_d * (1 / n + test_frac / (1 - test_frac)))
    t = mean_d / denom
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return mean_d, t, p


rows = []
pairs = list(itertools.combinations(models, 2))
for a, b in pairs:
    xa, xb = piv[a].values, piv[b].values
    mask = ~(np.isnan(xa) | np.isnan(xb))
    diff, t, p = corrected_ttest(xa[mask], xb[mask], mask.sum(), TEST_FRAC)
    # naive (WRONG) test, shown for contrast
    _, p_naive = stats.ttest_rel(xa[mask], xb[mask])
    # Holm-Bonferroni applied below
    rows.append(dict(A=a, B=b, diff=diff, t=t, p_corrected=p, p_naive=p_naive,
                     wins_A=int((xa[mask] > xb[mask]).sum()),
                     wins_B=int((xb[mask] > xa[mask]).sum())))

cmp = pd.DataFrame(rows).sort_values("p_corrected")
# Holm-Bonferroni across the family of pairwise tests
m_tests = len(cmp)
cmp["p_holm"] = np.maximum.accumulate(
    np.minimum(1.0, cmp.p_corrected.values * (m_tests - np.arange(m_tests))))
cmp["sig"] = np.where(cmp.p_holm < 0.05, "*", "ns")

print(f"{'Comparison':<34}{'Δ':>9}{'t':>8}{'p_corr':>9}"
      f"{'p_holm':>9}{'sig':>5}{'W/L':>10}{'p_naive':>10}")
for _, r in cmp.iterrows():
    print(f"{r.A + ' vs ' + r.B:<34}{r['diff']:>+9.4f}{r.t:>8.2f}"
          f"{r.p_corrected:>9.4f}{r.p_holm:>9.4f}{r.sig:>5}"
          f"{f'{r.wins_A}/{r.wins_B}':>10}{r.p_naive:>10.4f}")

print("\n   Δ = mean(A) - mean(B).  * = significant at .05 after Holm correction.")
print("   Compare p_corr against p_naive: the naive paired t-test is")
print("   anti-conservative here. Never report p_naive.")

# ---------------------------------------------------------------------------
# B. Friedman + Nemenyi
# ---------------------------------------------------------------------------
print("\n" + "=" * 76)
print("B. FRIEDMAN OMNIBUS TEST + NEMENYI POST-HOC  (Demsar, 2006)")
print("=" * 76)
clean = piv.dropna()
if len(models) >= 3 and len(clean) > 0:
    stat, p = stats.friedmanchisquare(*[clean[m].values for m in models])
    print(f"\n   Friedman chi2 = {stat:.4f}, p = {p:.5f}  "
          f"({'differences exist' if p < 0.05 else 'NO significant difference between models'})")

    ranks = clean.rank(axis=1, ascending=False)
    avg_rank = ranks.mean().sort_values()
    print("\n   Average ranks (1 = best):")
    for m, r in avg_rank.items():
        print(f"      {m:<16}{r:.3f}")

    k, N = len(models), len(clean)
    # critical values of the Nemenyi two-tailed studentised range / sqrt(2)
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850}
    cd = q05.get(k, 2.569) * np.sqrt(k * (k + 1) / (6 * N))
    print(f"\n   Critical difference (alpha=.05, k={k}, N={N}) = {cd:.3f}")
    if p < 0.05:
        print("   Pairs whose average ranks differ by more than the CD:")
        any_sig = False
        for a, b in pairs:
            d = abs(avg_rank[a] - avg_rank[b])
            if d > cd:
                print(f"      {a} vs {b}:  Δrank = {d:.3f} > CD  *")
                any_sig = True
        if not any_sig:
            print("      none")
    else:
        print("   Omnibus test not significant -> do not run post-hoc "
              "comparisons or claim any model is better.")

# ---------------------------------------------------------------------------
# C. The sentence to write
# ---------------------------------------------------------------------------
print("\n" + "=" * 76)
print("C. WHAT YOU CAN ACTUALLY CLAIM")
print("=" * 76)
best = piv.mean().idxmax()
n_sig = int((cmp.p_holm < 0.05).sum())
print(f"\n   Highest mean {METRIC}: {best} ({piv[best].mean():.4f})")
print(f"   Significant pairwise differences after correction: {n_sig} of {m_tests}")
if n_sig == 0:
    print(f"""
   Template for the paper:

     "All four architectures performed comparably. A Friedman test across
      {n_folds} cross-validation folds revealed no significant differences in
      {METRIC} (chi2 = {stat:.2f}, p = {p:.3f}), and no pairwise comparison
      survived Holm correction. {best} achieved the highest mean {METRIC}
      ({piv[best].mean():.3f} +/- {piv[best].std(ddof=1):.3f}), but this
      advantage is within cross-validation variance. Given that XGBoost
      requires orders of magnitude fewer parameters and less training time
      (Table 1), it remains the preferable choice at this sample size."

   Do NOT write "X outperformed Y" for any pair marked ns above.""")
else:
    print("\n   You may claim superiority only for pairs marked * above,")
    print("   and you should report the effect size (Δ) alongside p.")

cmp.to_csv("results/table4_significance.csv", index=False)
print("\n   Saved: results/table4_significance.csv")
