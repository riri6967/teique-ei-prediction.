"""
05_psychometric_analysis.py
Produces the measurement-resolution argument that Section 5.2 of the paper rests
on. Every number quoted in the Discussion should come from here so it is
reproducible by a reviewer.

Outputs:
  results/table5_psychometrics.csv   reliability, SEM, band widths
  results/table6_boundary_density.csv  cases near each cut-point
  results/fig8_score_distribution.png  distribution with cut-points and SEM band

Run after 01_clean_and_score.py.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = "results"
import os
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv("teique_clean.csv")
items = [c for c in df.columns if c.startswith("i") and c[1:].isdigit()]
k = len(items)
X = df[items]
m = df["ei_mean"].values
total = X.sum(axis=1)

# ---------------------------------------------------------------- reliability
item_var = X.var(ddof=1)
alpha = (k / (k - 1)) * (1 - item_var.sum() / total.var(ddof=1))

# split-half (odd vs even items), Spearman-Brown corrected
odd = X[[c for i, c in enumerate(items) if i % 2 == 0]].sum(axis=1)
even = X[[c for i, c in enumerate(items) if i % 2 == 1]].sum(axis=1)
r_hh = np.corrcoef(odd, even)[0, 1]
sb = 2 * r_hh / (1 + r_hh)

sd = m.std(ddof=1)
sem = sd * np.sqrt(1 - alpha)
ci95 = 1.96 * sem

print("=" * 70)
print("RELIABILITY AND MEASUREMENT PRECISION")
print("=" * 70)
print(f"  Items                          k = {k}")
print(f"  Respondents                    n = {len(df)}")
print(f"  Cronbach's alpha                 = {alpha:.4f}")
print(f"  Split-half (Spearman-Brown)      = {sb:.4f}")
print(f"  Global EI (item mean)            M = {m.mean():.4f}, SD = {sd:.4f}")
print(f"  Standard error of measurement    SEM = SD x sqrt(1-alpha) = {sem:.4f}")
print(f"  95% CI for an individual score   +/-{ci95:.4f}")

# ---------------------------------------------------------------- band widths
schemes = {}
q1, q2 = np.quantile(m, [1/3, 2/3])
schemes["Tertile split"] = (q1, q2)
schemes["+/-0.5 SD"] = (m.mean() - 0.5*sd, m.mean() + 0.5*sd)
schemes["+/-1.0 SD"] = (m.mean() - sd, m.mean() + sd)

print("\n" + "=" * 70)
print("MIDDLE-CATEGORY WIDTH vs MEASUREMENT ERROR")
print("=" * 70)
rows = []
for name, (lo, hi) in schemes.items():
    width = hi - lo
    ratio = width / ci95
    verdict = "UNRESOLVABLE" if ratio < 1 else ("marginal" if ratio < 1.5 else "resolvable")
    counts = pd.cut(m, [-np.inf, lo, hi, np.inf],
                    labels=["Low", "Medium", "High"]).value_counts()
    print(f"\n  {name}")
    print(f"     cut-points        {lo:.4f} / {hi:.4f}")
    print(f"     Medium width      {width:.4f} scale points ({width*k:.1f} raw pts of {k*5})")
    print(f"     width / 95% CI    {ratio:.2f}   -> {verdict}")
    print(f"     class sizes       Low={counts.get('Low',0)} "
          f"Medium={counts.get('Medium',0)} High={counts.get('High',0)}")
    rows.append(dict(scheme=name, cut_low=round(lo,4), cut_high=round(hi,4),
                     medium_width=round(width,4),
                     width_over_ci95=round(ratio,3), verdict=verdict,
                     n_low=int(counts.get("Low",0)),
                     n_medium=int(counts.get("Medium",0)),
                     n_high=int(counts.get("High",0))))

pd.DataFrame([dict(k_items=k, n=len(df), alpha=round(alpha,4),
                   split_half_sb=round(sb,4), mean=round(m.mean(),4),
                   sd=round(sd,4), sem=round(sem,4), ci95=round(ci95,4))]) \
  .to_csv(f"{OUT}/table5_psychometrics.csv", index=False)
pd.DataFrame(rows).to_csv(f"{OUT}/table6_boundary_density.csv", index=False)

# ---------------------------------------------------------------- boundary density
print("\n" + "=" * 70)
print("PROPORTION OF SAMPLE NEAR A TERTILE CUT-POINT")
print("=" * 70)
d = np.minimum(np.abs(m - q1), np.abs(m - q2))
for thr in [0.05, 0.10, 0.156, 0.20]:
    n = int((d < thr).sum())
    tag = "  <-- within 1 SEM" if abs(thr - sem) < 0.01 else ""
    print(f"   within +/-{thr:.3f}: {n:3d} / {len(m)}  ({100*n/len(m):.1f}%){tag}")

# ---------------------------------------------------------------- normality
print("\n" + "=" * 70)
print("DISTRIBUTION OF THE GLOBAL SCORE")
print("=" * 70)
print(f"   skew {stats.skew(m):+.4f}   kurtosis {stats.kurtosis(m):+.4f}")
sh = stats.shapiro(m)
print(f"   Shapiro-Wilk W={sh.statistic:.4f}, p={sh.pvalue:.4f} "
      f"({'consistent with normality' if sh.pvalue>.05 else 'departs from normality'})")
print(f"   range {m.min():.3f} - {m.max():.3f}")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(m, bins=22, color="#4C72B0", edgecolor="white", alpha=.85)
for c, lab in [(q1, "tertile cut-points"), (q2, None)]:
    ax.axvline(c, color="crimson", lw=1.6, ls="--", label=lab)
ax.axvspan(q1, q2, color="crimson", alpha=.07,
           label=f"Medium band (width {q2-q1:.3f})")
mid = (q1 + q2) / 2
ax.errorbar([mid], [ax.get_ylim()[1]*.88], xerr=[ci95], fmt="o",
            color="black", capsize=5, lw=2,
            label=f"95% CI of one score (±{ci95:.3f})")
ax.set_xlabel("global trait EI (item mean, 1–5)")
ax.set_ylabel("respondents")
ax.set_title("Figure 8. The Medium band is narrower than the measurement error")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig8_score_distribution.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- AUC vs accuracy
print("\n" + "=" * 70)
print("DISCRIMINATION vs CLASSIFICATION (needs results/fold_metrics.csv)")
print("=" * 70)
try:
    r = pd.read_csv(f"{OUT}/fold_metrics.csv")
    g = r.groupby("model")[["roc_auc", "accuracy"]].mean()
    g["gap"] = g.roc_auc - g.accuracy
    print(g.round(4).to_string())
    print(f"\n   mean AUC {g.roc_auc.mean():.4f} vs mean accuracy "
          f"{g.accuracy.mean():.4f}  (gap {g.gap.mean():+.4f})")
    print("   A consistent positive gap means the models rank respondents well")
    print("   and fail at discretisation -- i.e. the cut-points, not the models.")
    g.round(4).to_csv(f"{OUT}/table7_auc_vs_accuracy.csv")
except FileNotFoundError:
    print("   fold_metrics.csv not found -- run ei_pipeline.py first")

print(f"\nSaved to {OUT}/: table5_psychometrics.csv, table6_boundary_density.csv,")
print(f"                 table7_auc_vs_accuracy.csv, fig8_score_distribution.png")
