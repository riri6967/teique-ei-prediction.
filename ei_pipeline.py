"""
================================================================================
Emotional Intelligence Prediction from the TEIQue-SF
Deep tabular architectures vs. gradient boosting, with SHAP interpretation
================================================================================

Models   : FT-Transformer | TabNet (+SHAP) | SAINT | XGBoost (+SHAP)
Metrics  : Accuracy, Precision, Recall, F1, ROC-AUC, Cohen's kappa, MCC,
           Training time, Inference time, Parameter count
Analysis : confusion matrices, per-class precision/recall, misclassification
           patterns, boundary-distance error analysis

DESIGN (important -- read METHODOLOGY.md first)
--------------------------------------------------------------------------------
Predicting the EI tertile from all 30 TEIQue items is circular: the label is a
discretised function of the features. This script therefore supports three
designs, selected by DESIGN below:

  "shortform"  (default, publishable) predict the 30-item class from a k-item
               subset selected INSIDE each training fold. Research question:
               how few items suffice to recover full-scale EI classification?
  "leakage"    all 30 items -> class. Reported only as a diagnostic ceiling.
  "holdout"    predict a held-out item block from the remaining items.

Run:  python ei_pipeline.py
Deps: pip install torch xgboost shap pytorch-tabnet scikit-learn pandas \
                  matplotlib seaborn openpyxl
================================================================================
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================== CONFIGURATION ==============================
DATA        = "teique_clean.csv"
OUTDIR      = "results"
DESIGN      = "shortform"      # "shortform" | "leakage" | "holdout"
K_ITEMS     = 10               # items retained in the short form
N_SPLITS    = 5
N_REPEATS   = 10               # 5x10 = 50 fits per model
SEED        = 42
CLASSES     = ["Low EI", "Medium EI", "High EI"]
DEVICE      = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"

os.makedirs(OUTDIR, exist_ok=True)
rng = np.random.default_rng(SEED)


# ============================== DEEP MODELS =================================
def _torch():
    import torch
    return torch


class FeatureTokenizer:
    """Placeholder so the module imports without torch; real class below."""


def build_torch_models():
    """Defined lazily so the metrics harness can be tested without torch."""
    import torch
    import torch.nn as nn

    # ---------------------------------------------------------------- FT-T
    class Tokenizer(nn.Module):
        """One learned embedding per numerical feature + [CLS] token.
        (Gorishniy et al., 2021, 'Revisiting Deep Learning Models for
        Tabular Data')."""

        def __init__(self, n_features: int, d_token: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(n_features, d_token))
            self.bias = nn.Parameter(torch.empty(n_features, d_token))
            self.cls = nn.Parameter(torch.empty(1, 1, d_token))
            for p in (self.weight, self.bias, self.cls):
                nn.init.uniform_(p, -n_features ** -0.5, n_features ** -0.5)

        def forward(self, x):                      # x: (B, F)
            t = self.weight[None] * x[..., None] + self.bias[None]
            return torch.cat([self.cls.expand(x.size(0), -1, -1), t], dim=1)

    class Block(nn.Module):
        """Pre-norm transformer block with GEGLU feed-forward."""

        def __init__(self, d, heads, drop):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, dropout=drop,
                                              batch_first=True)
            self.ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(),
                                    nn.Dropout(drop), nn.Linear(d * 4, d))
            self.drop = nn.Dropout(drop)

        def forward(self, x, inter=None):
            h = self.n1(x)
            a, _ = self.attn(h, h, h, need_weights=False)
            x = x + self.drop(a)
            return x + self.drop(self.ff(self.n2(x)))

    class FTTransformer(nn.Module):
        def __init__(self, n_features, n_classes, d_token=64, n_blocks=2,
                     heads=8, drop=0.2):
            super().__init__()
            self.tok = Tokenizer(n_features, d_token)
            self.blocks = nn.ModuleList(
                [Block(d_token, heads, drop) for _ in range(n_blocks)])
            self.norm = nn.LayerNorm(d_token)
            self.head = nn.Linear(d_token, n_classes)

        def forward(self, x):
            h = self.tok(x)
            for b in self.blocks:
                h = b(h)
            return self.head(self.norm(h[:, 0]))    # [CLS] representation

    # --------------------------------------------------------------- SAINT
    class IntersampleAttention(nn.Module):
        """SAINT's defining component: attention ACROSS rows in the batch
        (Somepalli et al., 2021). Each row attends to every other row, so the
        model can exploit neighbourhood structure -- valuable when n is small.
        Implemented by flattening tokens per row and treating the batch as the
        sequence dimension."""

        def __init__(self, d_token, n_tokens, heads, drop):
            super().__init__()
            dim = d_token * n_tokens
            self.norm = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, heads, dropout=drop,
                                              batch_first=True)
            self.n_tokens, self.d_token = n_tokens, d_token

        def forward(self, x):                       # (B, T, D)
            b, t, d = x.shape
            flat = self.norm(x.reshape(1, b, t * d))   # batch as sequence
            out, _ = self.attn(flat, flat, flat, need_weights=False)
            return x + out.reshape(b, t, d)

    class SAINT(nn.Module):
        def __init__(self, n_features, n_classes, d_token=32, n_blocks=2,
                     heads=4, drop=0.2):
            super().__init__()
            self.tok = Tokenizer(n_features, d_token)
            n_tok = n_features + 1
            self.self_blocks = nn.ModuleList(
                [Block(d_token, heads, drop) for _ in range(n_blocks)])
            self.inter_blocks = nn.ModuleList(
                [IntersampleAttention(d_token, n_tok, heads, drop)
                 for _ in range(n_blocks)])
            self.norm = nn.LayerNorm(d_token)
            self.head = nn.Linear(d_token, n_classes)

        def forward(self, x):
            h = self.tok(x)
            for sb, ib in zip(self.self_blocks, self.inter_blocks):
                h = ib(sb(h))                      # self-attn then inter-sample
            return self.head(self.norm(h[:, 0]))

    return FTTransformer, SAINT


def train_torch(model_cls, Xtr, ytr, Xte, n_classes, epochs=200, lr=1e-3,
                wd=1e-4, batch=32, patience=30, seed=0, **kw):
    """Train a torch model with early stopping on an internal validation split.
    Returns (probabilities, n_parameters, train_seconds, infer_seconds)."""
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split

    torch.manual_seed(seed)
    dev = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    # internal validation split for early stopping (never touches the test fold)
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2,
                                      stratify=ytr, random_state=seed)
    to = lambda a, t=torch.float32: torch.tensor(a, dtype=t, device=dev)
    Xa_, ya_ = to(Xa), to(ya, torch.long)
    Xv_, yv_ = to(Xv), to(yv, torch.long)
    Xte_ = to(Xte)

    model = model_cls(Xtr.shape[1], n_classes, **kw).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    best, best_state, bad = np.inf, None, 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xa_), device=dev)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 4:                       # inter-sample attn needs rows
                continue
            opt.zero_grad()
            loss = lossf(model(Xa_[idx]), ya_[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv_), yv_).item()
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    train_s = time.perf_counter() - t0

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        prob = torch.softmax(model(Xte_), dim=1).cpu().numpy()
    infer_s = time.perf_counter() - t0
    return prob, n_params, train_s, infer_s


# ============================== METRIC HARNESS ==============================
def compute_metrics(y_true, y_pred, y_prob):
    """All requested classification metrics for one fold."""
    m = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "kappa":     cohen_kappa_score(y_true, y_pred),
        "mcc":       matthews_corrcoef(y_true, y_pred),
    }
    # macro OVR AUC; needs every class present in the fold
    try:
        if len(np.unique(y_true)) == y_prob.shape[1]:
            m["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                         average="macro")
        else:
            m["roc_auc"] = np.nan
    except ValueError:
        m["roc_auc"] = np.nan
    # per-class precision / recall
    for i, c in enumerate(CLASSES):
        m[f"prec_{c}"] = precision_score(y_true, y_pred, labels=[i],
                                         average="macro", zero_division=0)
        m[f"rec_{c}"] = recall_score(y_true, y_pred, labels=[i],
                                     average="macro", zero_division=0)
    return m


def select_items(X, y, item_names, k):
    """Item selection performed INSIDE the training fold only.
    Ranks by corrected item-total correlation -- a standard psychometric
    criterion, and one that never looks at the class label, so it cannot leak
    label information."""
    tot = X.sum(axis=1)
    scores = []
    for j in range(X.shape[1]):
        rest = tot - X[:, j]
        s = np.corrcoef(X[:, j], rest)[0, 1]
        scores.append(-1e9 if np.isnan(s) else s)
    order = np.argsort(scores)[::-1][:k]
    return np.sort(order)


# ============================== MAIN EXPERIMENT =============================
def run(use_torch=True):
    df = pd.read_csv(DATA)
    items = [c for c in df.columns if c.startswith("i") and c[1:].isdigit()]
    X = df[items].values.astype(np.float32)
    y = df["ei_class"].map({c: i for i, c in enumerate(CLASSES)}).values
    ei_mean = df["ei_mean"].values
    cuts = np.quantile(ei_mean, [1 / 3, 2 / 3])
    boundary_dist = np.minimum(np.abs(ei_mean - cuts[0]),
                               np.abs(ei_mean - cuts[1]))

    print(f"n={len(y)}  items={len(items)}  design={DESIGN}  k={K_ITEMS}")
    print("class counts:", dict(zip(CLASSES, np.bincount(y))), "\n")

    # ------------------------------------------------ model registry
    models = {}

    def add_xgb():
        import xgboost as xgb

        def fit(Xtr, ytr, Xte, seed):
            clf = xgb.XGBClassifier(
                n_estimators=400, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, min_child_weight=2,
                objective="multi:softprob", num_class=3,
                tree_method="hist", random_state=seed, n_jobs=1,
                eval_metric="mlogloss")
            t0 = time.perf_counter(); clf.fit(Xtr, ytr)
            tr = time.perf_counter() - t0
            t0 = time.perf_counter(); p = clf.predict_proba(Xte)
            inf = time.perf_counter() - t0
            n_par = int(sum(t.count("\n") for t in
                            clf.get_booster().get_dump()))   # node count proxy
            return p, n_par, tr, inf, clf
        models["XGBoost"] = fit

    def add_tabnet():
        from pytorch_tabnet.tab_model import TabNetClassifier

        def fit(Xtr, ytr, Xte, seed):
            from sklearn.model_selection import train_test_split
            Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2,
                                              stratify=ytr, random_state=seed)
            clf = TabNetClassifier(n_d=8, n_a=8, n_steps=3, gamma=1.3,
                                   lambda_sparse=1e-3, seed=seed,
                                   optimizer_params=dict(lr=2e-2),
                                   verbose=0, device_name=DEVICE)
            t0 = time.perf_counter()
            clf.fit(Xa, ya, eval_set=[(Xv, yv)], eval_metric=["logloss"],
                    max_epochs=200, patience=30, batch_size=32,
                    virtual_batch_size=16, drop_last=False)
            tr = time.perf_counter() - t0
            t0 = time.perf_counter(); p = clf.predict_proba(Xte)
            inf = time.perf_counter() - t0
            n_par = sum(q.numel() for q in clf.network.parameters()
                        if q.requires_grad)
            return p, n_par, tr, inf, clf
        models["TabNet"] = fit

    def add_deep():
        FTT, SAINT = build_torch_models()
        for name, cls, kw in [("FT-Transformer", FTT,
                               dict(d_token=64, n_blocks=2, heads=8, drop=0.2)),
                              ("SAINT", SAINT,
                               dict(d_token=32, n_blocks=2, heads=4, drop=0.2))]:
            def fit(Xtr, ytr, Xte, seed, cls=cls, kw=kw):
                p, n_par, tr, inf = train_torch(cls, Xtr, ytr, Xte, 3,
                                                seed=seed, **kw)
                return p, n_par, tr, inf, None
            models[name] = fit

    if use_torch:
        for adder in (add_xgb, add_tabnet, add_deep):
            try:
                adder()
            except ImportError as e:
                print(f"  [skip] {e}")
    else:  # sklearn stand-ins, for verifying the harness without torch
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      RandomForestClassifier)
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        for name, ctor in [("XGBoost", lambda s: HistGradientBoostingClassifier(random_state=s)),
                           ("TabNet", lambda s: MLPClassifier((32, 16), max_iter=800, random_state=s)),
                           ("FT-Transformer", lambda s: MLPClassifier((64, 32), max_iter=800, random_state=s)),
                           ("SAINT", lambda s: LogisticRegression(max_iter=2000))]:
            def fit(Xtr, ytr, Xte, seed, ctor=ctor):
                clf = ctor(seed)
                t0 = time.perf_counter(); clf.fit(Xtr, ytr)
                tr = time.perf_counter() - t0
                t0 = time.perf_counter(); p = clf.predict_proba(Xte)
                inf = time.perf_counter() - t0
                n_par = sum(c.size for c in getattr(clf, "coefs_", []) ) or -1
                return p, n_par, tr, inf, clf
            models[name] = fit

    # ------------------------------------------------ cross-validation
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=SEED)
    fold_rows, oof = [], {m: np.full((len(y), 3), np.nan) for m in models}
    oof_count = {m: np.zeros(len(y)) for m in models}
    cms = {m: np.zeros((3, 3), dtype=int) for m in models}
    sel_counts = {c: 0 for c in items}

    for fold, (tr_i, te_i) in enumerate(cv.split(X, y)):
        Xtr_raw, Xte_raw = X[tr_i], X[te_i]
        ytr, yte = y[tr_i], y[te_i]

        if DESIGN == "shortform":
            keep = select_items(Xtr_raw, ytr, items, K_ITEMS)
        elif DESIGN == "holdout":
            keep = np.arange(len(items))[: len(items) - 10]
        else:
            keep = np.arange(len(items))
        for j in keep:
            sel_counts[items[j]] += 1

        sc = StandardScaler().fit(Xtr_raw[:, keep])
        Xtr = sc.transform(Xtr_raw[:, keep]).astype(np.float32)
        Xte = sc.transform(Xte_raw[:, keep]).astype(np.float32)

        for name, fit in models.items():
            prob, n_par, tr_s, inf_s, _ = fit(Xtr, ytr, Xte, SEED + fold)
            pred = prob.argmax(1)
            row = compute_metrics(yte, pred, prob)
            row.update(model=name, fold=fold, n_params=n_par,
                       train_time=tr_s,
                       infer_time=inf_s / max(len(te_i), 1))
            fold_rows.append(row)
            cms[name] += confusion_matrix(yte, pred, labels=[0, 1, 2])
            m0 = np.nan_to_num(oof[name][te_i], nan=0.0)
            oof[name][te_i] = m0 + prob
            oof_count[name][te_i] += 1

        if fold % 10 == 0:
            print(f"  fold {fold+1}/{N_SPLITS*N_REPEATS}")

    res = pd.DataFrame(fold_rows)
    res.to_csv(f"{OUTDIR}/fold_metrics.csv", index=False)

    # ------------------------------------------------ Table 1: main metrics
    core = ["accuracy", "precision", "recall", "f1", "roc_auc", "kappa", "mcc"]
    agg = res.groupby("model")[core + ["train_time", "infer_time", "n_params"]] \
             .agg(["mean", "std"])
    t1 = pd.DataFrame(index=agg.index)
    for c in core:
        t1[c.upper()] = (agg[(c, "mean")].map("{:.3f}".format) + " ± "
                         + agg[(c, "std")].map("{:.3f}".format))
    t1["Train (s)"] = agg[("train_time", "mean")].map("{:.2f}".format)
    t1["Infer (ms/sample)"] = (agg[("infer_time", "mean")] * 1000).map("{:.3f}".format)
    t1["Params"] = agg[("n_params", "mean")].map("{:,.0f}".format)
    order = [m for m in ["FT-Transformer", "TabNet", "SAINT", "XGBoost"]
             if m in t1.index]
    t1 = t1.loc[order]
    t1.to_csv(f"{OUTDIR}/table1_metrics.csv")

    print("\n" + "=" * 78)
    print("TABLE 1  Model comparison "
          f"({N_SPLITS}x{N_REPEATS} repeated stratified CV, mean ± SD)")
    print("=" * 78)
    print(t1.to_string())

    # ------------------------------------------------ Table 2: per class
    rows = []
    for m in order:
        sub = res[res.model == m]
        for c in CLASSES:
            rows.append(dict(Model=m, Class=c,
                             Precision=f"{sub[f'prec_{c}'].mean():.3f} ± {sub[f'prec_{c}'].std():.3f}",
                             Recall=f"{sub[f'rec_{c}'].mean():.3f} ± {sub[f'rec_{c}'].std():.3f}"))
    t2 = pd.DataFrame(rows)
    t2.to_csv(f"{OUTDIR}/table2_per_class.csv", index=False)
    print("\n" + "=" * 78)
    print("TABLE 2  Per-class precision and recall")
    print("=" * 78)
    print(t2.to_string(index=False))

    # ------------------------------------------------ Table 3: error patterns
    rows = []
    for m in order:
        cm = cms[m]
        tot_err = cm.sum() - np.trace(cm)
        for i in range(3):
            for j in range(3):
                if i != j and cm[i, j] > 0:
                    rows.append(dict(Model=m, True_=CLASSES[i], Pred=CLASSES[j],
                                     n=int(cm[i, j]),
                                     pct_of_errors=round(100 * cm[i, j] / max(tot_err, 1), 1)))
    t3 = (pd.DataFrame(rows).sort_values(["Model", "n"], ascending=[True, False]))
    t3.to_csv(f"{OUTDIR}/table3_misclassification.csv", index=False)
    print("\n" + "=" * 78)
    print("TABLE 3  Misclassification patterns (pooled over all CV folds)")
    print("=" * 78)
    print(t3.to_string(index=False))

    # adjacent vs non-adjacent errors -- are mistakes psychometrically sensible?
    print("\nAdjacent vs. distant errors (Low<->High is a 'severe' error):")
    for m in order:
        cm = cms[m]
        adj = cm[0, 1] + cm[1, 0] + cm[1, 2] + cm[2, 1]
        sev = cm[0, 2] + cm[2, 0]
        print(f"   {m:16s} adjacent={adj:5d}  severe={sev:4d} "
              f"({100*sev/max(adj+sev,1):.1f}% of errors)")

    # ------------------------------------------------ boundary analysis
    print("\nAccuracy by distance from the tertile cut-point (OOF predictions):")
    edges = [0, .05, .10, .20, np.inf]
    for m in order:
        p = oof[m] / np.maximum(oof_count[m], 1)[:, None]
        pred = np.nanargmax(p, axis=1)
        line = f"   {m:16s}"
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (boundary_dist >= lo) & (boundary_dist < hi)
            line += (f"  d∈[{lo:.2f},{hi if hi!=np.inf else 9:.2f}) "
                     f"n={sel.sum():3d} acc={accuracy_score(y[sel], pred[sel]):.3f}")
        print(line)

    # ------------------------------------------------ item selection stability
    ss = (pd.Series(sel_counts) / (N_SPLITS * N_REPEATS)).sort_values(ascending=False)
    ss.to_csv(f"{OUTDIR}/item_selection_frequency.csv", header=["selection_rate"])
    print(f"\nItems selected in >90% of folds: "
          f"{list(ss[ss > .9].index)}")

    # ------------------------------------------------ figures
    make_figures(res, cms, t1, order, core)

    # ------------------------------------------------ SHAP
    if use_torch:
        run_shap(X, y, items, cuts)

    with open(f"{OUTDIR}/summary.json", "w") as f:
        json.dump({"design": DESIGN, "k_items": K_ITEMS, "n": int(len(y)),
                   "cv": f"{N_SPLITS}x{N_REPEATS}",
                   "confusion_matrices": {k: v.tolist() for k, v in cms.items()}},
                  f, indent=2)
    print(f"\nAll outputs -> {OUTDIR}/")
    return res, cms


# ============================== FIGURES =====================================
def make_figures(res, cms, t1, order, core):
    sns.set_theme(style="whitegrid", font_scale=0.95)

    # Fig 1 -- grouped metric bars with SD error bars
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w, xs = 0.2, np.arange(len(core))
    for i, m in enumerate(order):
        sub = res[res.model == m]
        mu = [sub[c].mean() for c in core]
        sd = [sub[c].std() for c in core]
        ax.bar(xs + i * w - 1.5 * w, mu, w, yerr=sd, capsize=2.5, label=m)
    ax.axhline(1 / 3, ls="--", c="grey", lw=1, label="chance")
    ax.set_xticks(xs); ax.set_xticklabels([c.upper() for c in core])
    ax.set_ylabel("score"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Figure 1. Model performance ({N_SPLITS}×{N_REPEATS} repeated stratified CV)")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.12))
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig1_metrics.png", dpi=300)
    plt.close(fig)

    # Fig 2 -- confusion matrices, row-normalised
    fig, axes = plt.subplots(1, len(order), figsize=(4.1 * len(order), 3.7))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, order):
        cm = cms[m]
        sns.heatmap(cm / cm.sum(1, keepdims=True), annot=cm, fmt="d",
                    cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=[c.replace(" EI", "") for c in CLASSES],
                    yticklabels=[c.replace(" EI", "") for c in CLASSES], ax=ax)
        ax.set_title(m); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.suptitle("Figure 2. Confusion matrices (counts pooled over folds, shading = row proportion)")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig2_confusion.png", dpi=300)
    plt.close(fig)

    # Fig 3 -- per-class recall
    fig, ax = plt.subplots(figsize=(7.5, 4))
    long = []
    for m in order:
        sub = res[res.model == m]
        for c in CLASSES:
            for v in sub[f"rec_{c}"]:
                long.append(dict(Model=m, Class=c.replace(" EI", ""), Recall=v))
    sns.barplot(data=pd.DataFrame(long), x="Class", y="Recall", hue="Model",
                errorbar="sd", ax=ax)
    ax.axhline(1 / 3, ls="--", c="grey", lw=1)
    ax.set_title("Figure 3. Per-class recall — the Medium band is the failure mode")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig3_per_class_recall.png", dpi=300)
    plt.close(fig)

    # Fig 4 -- efficiency: accuracy vs training time, sized by parameters
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m in order:
        sub = res[res.model == m]
        npar = max(sub.n_params.mean(), 1)
        ax.scatter(sub.train_time.mean(), sub.accuracy.mean(),
                   s=60 + 340 * np.log10(npar + 10) / 6, alpha=.75, label=m)
        ax.annotate(f"{m}\n{npar:,.0f} params",
                    (sub.train_time.mean(), sub.accuracy.mean()),
                    textcoords="offset points", xytext=(9, -4), fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("training time per fold (s, log)")
    ax.set_ylabel("accuracy")
    ax.set_title("Figure 4. Accuracy vs. computational cost")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig4_efficiency.png", dpi=300)
    plt.close(fig)

    # Fig 4b -- the three cost metrics as explicit bars (log scale: the deep
    # models differ from XGBoost by orders of magnitude on all three)
    cost = [("train_time", "training time per fold (s)"),
            ("infer_time", "inference time (ms / sample)"),
            ("n_params", "trainable parameters")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, lab) in zip(axes, cost):
        mu = [res[res.model == m][col].mean() for m in order]
        sd = [res[res.model == m][col].std() for m in order]
        if col == "infer_time":
            mu = [v * 1000 for v in mu]; sd = [v * 1000 for v in sd]
        bars = ax.bar(range(len(order)), mu, yerr=sd, capsize=3,
                      color=sns.color_palette()[:len(order)])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(lab)
        if max(mu) / max(min([v for v in mu if v > 0], default=1), 1e-9) > 50:
            ax.set_yscale("log")
        for b, v in zip(bars, mu):
            ax.annotate(f"{v:,.3g}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7)
    fig.suptitle("Figure 4b. Computational cost: training time, inference time, "
                 "and model size")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig4b_cost_metrics.png", dpi=300)
    plt.close(fig)
    print(f"\nFigures written: fig1_metrics, fig2_confusion, "
          f"fig3_per_class_recall, fig4_efficiency, fig4b_cost_metrics (300 dpi)")


# ============================== SHAP ========================================
def run_shap(X, y, items, cuts):
    """SHAP for XGBoost (TreeExplainer, exact) and TabNet (its built-in masks +
    KernelExplainer). Fitted once on the full sample for interpretation only --
    never used for performance claims."""
    try:
        import shap
        import xgboost as xgb
    except ImportError as e:
        print(f"  [skip SHAP] {e}")
        return
    from sklearn.preprocessing import StandardScaler

    keep = select_items(X, y, items, K_ITEMS)
    names = [items[j] for j in keep]
    Xk = StandardScaler().fit_transform(X[:, keep])

    clf = xgb.XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            objective="multi:softprob", num_class=3,
                            random_state=SEED, n_jobs=1).fit(Xk, y)
    sv = shap.TreeExplainer(clf).shap_values(Xk)
    sv = np.stack(sv, -1) if isinstance(sv, list) else sv   # (n, f, c)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for c, ax in enumerate(axes):
        plt.sca(ax)
        shap.summary_plot(sv[:, :, c], Xk, feature_names=names,
                          show=False, plot_size=None, max_display=K_ITEMS)
        ax.set_title(f"{CLASSES[c]}")
    fig.suptitle("Figure 5. SHAP values per class — XGBoost")
    fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig5_shap_xgboost.png", dpi=300)
    plt.close(fig)

    imp = pd.DataFrame({"item": names,
                        "mean_abs_shap": np.abs(sv).mean(axis=(0, 2))}) \
            .sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(f"{OUTDIR}/shap_importance_xgboost.csv", index=False)
    print("\nTop SHAP items (XGBoost):")
    print(imp.head(8).to_string(index=False))

    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        tn = TabNetClassifier(n_d=8, n_a=8, n_steps=3, seed=SEED, verbose=0,
                              device_name=DEVICE)
        tn.fit(Xk, y, max_epochs=200, patience=30, batch_size=32,
               virtual_batch_size=16, drop_last=False)
        fi = pd.DataFrame({"item": names, "tabnet_importance": tn.feature_importances_}) \
               .sort_values("tabnet_importance", ascending=False)
        fi.to_csv(f"{OUTDIR}/tabnet_feature_importance.csv", index=False)
        expl, masks = tn.explain(Xk)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=fi, x="tabnet_importance", y="item", ax=ax)
        ax.set_title("Figure 6. TabNet attentive feature importance")
        fig.tight_layout(); fig.savefig(f"{OUTDIR}/fig6_tabnet_importance.png", dpi=300)
        plt.close(fig)
        print("\nTop TabNet items:")
        print(fi.head(8).to_string(index=False))
    except ImportError:
        print("  [skip TabNet SHAP] pytorch-tabnet not installed")


if __name__ == "__main__":
    import sys
    run(use_torch="--no-torch" not in sys.argv)
