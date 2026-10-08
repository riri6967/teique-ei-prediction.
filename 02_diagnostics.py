"""
02_diagnostics.py
Two questions that decide whether this study is publishable:

  (A) Is "predict EI class from the 30 items" circular?  The label is a
      deterministic function (tertile of the mean) of the 30 features, so any
      model should hit ~100%. Demonstrate it.

  (B) Under a non-circular design -- predict the FULL-SCALE class from a
      REDUCED item subset -- what accuracy is actually achievable? This is the
      realistic operating range the paper will report.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

df = pd.read_csv("./teique_clean.csv")
items = [c for c in df.columns if c.startswith("i") and c[1:].isdigit()]
y = df["ei_class"].map({"Low EI": 0, "Medium EI": 1, "High EI": 2}).values
X_all = df[items].values

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

models = {
    "Dummy (majority)": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000, C=1.0)),
    "Random Forest": RandomForestClassifier(n_estimators=400, random_state=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=0),
}

print("=" * 72)
print("(A) LEAKAGE TEST -- all 30 items as features, tertile label")
print("=" * 72)
for name, m in models.items():
    s = cross_val_score(m, X_all, y, cv=cv, scoring="accuracy", n_jobs=1)
    print(f"   {name:24s} acc = {s.mean():.4f} +/- {s.std():.4f}")

print("\n   Interpretation: the label IS the row mean, discretised. Nothing is")
print("   being predicted -- the models are re-deriving arithmetic.\n")

# ---------------------------------------------------------------------------
# (B) Non-circular design: reduced item subset -> full-scale class
# ---------------------------------------------------------------------------
print("=" * 72)
print("(B) SHORT-FORM DESIGN -- k items as features, 30-item class as label")
print("=" * 72)

# rank items by corrected item-total correlation (computed on TRAIN only in the
# real pipeline; here just to show the achievable range)
tot = df[items].sum(axis=1)
cit = {c: np.corrcoef(df[c], tot - df[c])[0, 1] for c in items}
ranked = sorted(items, key=lambda c: -cit[c])

print(f"\n{'k items':>8}  " + "  ".join(f"{n:>20s}" for n in list(models)[1:]))
for k in [5, 8, 10, 12, 15, 20]:
    sub = ranked[:k]
    Xs = df[sub].values
    row = f"{k:>8}  "
    for name, m in list(models.items())[1:]:
        s = cross_val_score(m, Xs, y, cv=cv, scoring="accuracy", n_jobs=1)
        row += f"  {s.mean():.3f} +/- {s.std():.3f}     "
    print(row)

print("\nChance level (3 balanced classes) = 0.333;  majority class = "
      f"{max(np.bincount(y))/len(y):.3f}")

# ---------------------------------------------------------------------------
# (C) How hard is the Medium-vs-High boundary specifically?
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("(C) PAIRWISE SEPARABILITY (10 items) -- where do errors concentrate?")
print("=" * 72)
sub10 = ranked[:10]
names = {0: "Low", 1: "Medium", 2: "High"}
for a, b in [(0, 1), (1, 2), (0, 2)]:
    mask = np.isin(y, [a, b])
    ya = (y[mask] == b).astype(int)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    s = cross_val_score(m, df[sub10].values[mask], ya,
                        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                                   random_state=1),
                        scoring="accuracy")
    print(f"   {names[a]:>6s} vs {names[b]:<6s}  acc = {s.mean():.3f} +/- {s.std():.3f}")

# ---------------------------------------------------------------------------
# (D) Sample-size reality check
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("(D) SAMPLE SIZE")
print("=" * 72)
print(f"   n = {len(df)}; 5-fold CV -> ~{len(df)//5} test cases per fold")
print(f"   One test case = {100/(len(df)//5):.1f} accuracy points.")
print("   A single 80/20 holdout (n_test=24) has a 95% CI of roughly +/-19 pts.")
print("   -> report repeated stratified CV with CIs, not a single split.")
