"""
00_sanity_checks.py
Run this BEFORE trusting any number from ei_pipeline.py.

These are the tests that catch the failures that actually kill papers: leakage,
a model that is silently predicting one class, a training loop that is not
learning, and overfitting hidden by a lucky split.

  1. Permutation (label-shuffle) test -- the single most important test.
  2. Learning curve -- is n the binding constraint?
  3. Train-vs-test gap -- how much are you overfitting?
  4. Degenerate-prediction check -- is the model just predicting the majority?
  5. Calibration -- are the probabilities meaningful (they feed ROC-AUC)?
  6. Leakage probe -- does selection inside vs outside the fold change results?

Usage:  python 00_sanity_checks.py            # sklearn stand-ins, fast
        python 00_sanity_checks.py --model xgb  # add XGBoost if installed
"""
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, learning_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["Low EI", "Medium EI", "High EI"]
K = 10
SEED = 42

df = pd.read_csv("teique_clean.csv")
items = [c for c in df.columns if c.startswith("i") and c[1:].isdigit()]
X = df[items].values.astype(float)
y = df["ei_class"].map({c: i for i, c in enumerate(CLASSES)}).values


def select_items(Xtr, k=K):
    """Item-total correlation ranking; label-blind."""
    tot = Xtr.sum(1)
    s = [np.corrcoef(Xtr[:, j], tot - Xtr[:, j])[0, 1] for j in range(Xtr.shape[1])]
    s = np.nan_to_num(s, nan=-1e9)
    return np.sort(np.argsort(s)[::-1][:k])


def cv_score(Xd, yd, model, n_repeats=10, seed=SEED, inside=True):
    """CV with item selection + scaling inside each fold (inside=True) or
    fitted once on all the data (inside=False, i.e. deliberately leaky)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats,
                                 random_state=seed)
    if not inside:
        keep = select_items(Xd)
        Xd = StandardScaler().fit_transform(Xd[:, keep])
    accs = []
    for tr, te in cv.split(Xd, yd):
        if inside:
            keep = select_items(Xd[tr])
            sc = StandardScaler().fit(Xd[tr][:, keep])
            a, b = sc.transform(Xd[tr][:, keep]), sc.transform(Xd[te][:, keep])
        else:
            a, b = Xd[tr], Xd[te]
        m = model()
        m.fit(a, yd[tr])
        accs.append(accuracy_score(yd[te], m.predict(b)))
    return np.array(accs)


MODELS = {
    "LogReg": lambda: LogisticRegression(max_iter=2000),
    "HistGB": lambda: HistGradientBoostingClassifier(random_state=0),
}
if "--model" in sys.argv and "xgb" in sys.argv:
    try:
        import xgboost as xgb
        MODELS["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="multi:softprob", num_class=3,
            random_state=0, n_jobs=1)
    except ImportError:
        print("xgboost not installed; continuing without it\n")

print(f"n={len(y)}  items={len(items)}  k={K}  classes={np.bincount(y).tolist()}")
maj = np.bincount(y).max() / len(y)
print(f"chance=0.333   majority-class baseline={maj:.3f}\n")

# ===========================================================================
print("=" * 74)
print("1. PERMUTATION TEST  (shuffle y, refit; accuracy MUST fall to chance)")
print("=" * 74)
print("   If a shuffled-label model still scores well above chance, you have")
print("   leakage or a bug. This is the test that would have caught the")
print("   original circular design.\n")
rng = np.random.default_rng(SEED)
for name, ctor in MODELS.items():
    real = cv_score(X, y, ctor, n_repeats=10).mean()
    perm = [cv_score(X, rng.permutation(y), ctor, n_repeats=2).mean()
            for _ in range(10)]
    perm = np.array(perm)
    p = (np.sum(perm >= real) + 1) / (len(perm) + 1)
    verdict = "PASS" if perm.mean() < 0.45 else "FAIL <-- investigate"
    print(f"   {name:9s} real={real:.3f}   shuffled={perm.mean():.3f} "
          f"± {perm.std():.3f}   p={p:.3f}   {verdict}")

# ===========================================================================
print("\n" + "=" * 74)
print("2. LEARNING CURVE  (is sample size the binding constraint?)")
print("=" * 74)
keep = select_items(X)
Xk = StandardScaler().fit_transform(X[:, keep])
for name, ctor in MODELS.items():
    sizes, tr_s, te_s = learning_curve(
        make_pipeline(StandardScaler(), ctor()), Xk, y,
        train_sizes=np.linspace(0.3, 1.0, 6),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
        scoring="accuracy", n_jobs=1)
    print(f"\n   {name}")
    print("     n_train : " + "  ".join(f"{int(s):5d}" for s in sizes))
    print("     train   : " + "  ".join(f"{v:.3f}" for v in tr_s.mean(1)))
    print("     test    : " + "  ".join(f"{v:.3f}" for v in te_s.mean(1)))
    slope = te_s.mean(1)[-1] - te_s.mean(1)[-3]
    print(f"     test accuracy still rising over last third: "
          f"{'YES (+%.3f) -> more data would help' % slope if slope > 0.01 else 'NO -> plateaued'}")

# ===========================================================================
print("\n" + "=" * 74)
print("3. TRAIN-VS-TEST GAP  (overfitting magnitude)")
print("=" * 74)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
for name, ctor in MODELS.items():
    tr_a, te_a = [], []
    for tr, te in cv.split(Xk, y):
        m = ctor(); m.fit(Xk[tr], y[tr])
        tr_a.append(accuracy_score(y[tr], m.predict(Xk[tr])))
        te_a.append(accuracy_score(y[te], m.predict(Xk[te])))
    gap = np.mean(tr_a) - np.mean(te_a)
    flag = "  <-- severe, reduce capacity" if gap > 0.25 else ""
    print(f"   {name:9s} train={np.mean(tr_a):.3f}  test={np.mean(te_a):.3f}"
          f"  gap={gap:.3f}{flag}")
print("\n   Deep models will show a larger gap. If train=1.000 and test~0.65,")
print("   the model memorised 94 rows -- shrink d_token / add dropout.")

# ===========================================================================
print("\n" + "=" * 74)
print("4. DEGENERATE-PREDICTION CHECK  (is every class actually predicted?)")
print("=" * 74)
for name, ctor in MODELS.items():
    preds = np.zeros(len(y), dtype=int)
    for tr, te in cv.split(Xk, y):
        m = ctor(); m.fit(Xk[tr], y[tr]); preds[te] = m.predict(Xk[te])
    counts = np.bincount(preds, minlength=3)
    ok = (counts > 0).all() and counts.max() / len(y) < 0.60
    print(f"   {name:9s} predicted counts {dict(zip(CLASSES, counts.tolist()))}"
          f"   {'PASS' if ok else 'FAIL <-- collapsing to one class'}")

# ===========================================================================
print("\n" + "=" * 74)
print("5. CALIBRATION  (ROC-AUC is meaningless if probabilities are not)")
print("=" * 74)
for name, ctor in MODELS.items():
    prob = np.zeros((len(y), 3))
    for tr, te in cv.split(Xk, y):
        m = ctor(); m.fit(Xk[tr], y[tr]); prob[te] = m.predict_proba(Xk[te])
    bs = np.mean([brier_score_loss((y == c).astype(int), prob[:, c])
                  for c in range(3)])
    conf = prob.max(1)
    acc_hi = accuracy_score(y[conf > 0.6], prob[conf > 0.6].argmax(1)) \
        if (conf > 0.6).sum() > 5 else np.nan
    print(f"   {name:9s} Brier={bs:.4f}   mean confidence={conf.mean():.3f}"
          f"   accuracy when confident(>0.6)={acc_hi:.3f} (n={(conf>0.6).sum()})")
print("\n   Accuracy-when-confident should clearly exceed overall accuracy.")
print("   That is what justifies reporting ROC-AUC and lets you defend a")
print("   'refer to human review when uncertain' screening use case.")

# ===========================================================================
print("\n" + "=" * 74)
print("6. LEAKAGE PROBE  (selection inside vs outside the fold)")
print("=" * 74)
for name, ctor in MODELS.items():
    ins = cv_score(X, y, ctor, n_repeats=10, inside=True).mean()
    out = cv_score(X, y, ctor, n_repeats=10, inside=False).mean()
    print(f"   {name:9s} inside-fold={ins:.3f}   outside(leaky)={out:.3f}"
          f"   inflation={out-ins:+.3f}")
print("\n   Report the inside-fold number. A large positive inflation is exactly")
print("   the optimism a reviewer will accuse you of if you preprocess first.")

print("\n" + "=" * 74)
print("If tests 1 and 4 pass and test 3's gap is not extreme, your pipeline is")
print("sound and you can trust ei_pipeline.py's numbers.")
print("=" * 74)
