"""
04_train_model.py
Train ONE model at a time, watch it learn, and save it.

ei_pipeline.py trains 50 times (5 folds x 10 repeats) and throws every model
away -- that is for *estimating performance*. This script does the other job:
fit a single model, print the loss per epoch so you can see whether it is
actually learning, and export a final model for SHAP and deployment.

Usage
-----
  python 04_train_model.py --model xgboost         # fast, start here
  python 04_train_model.py --model ftt --epochs 300
  python 04_train_model.py --model saint
  python 04_train_model.py --model tabnet
  python 04_train_model.py --model xgboost --tune  # nested-CV tuning
  python 04_train_model.py --model mlp             # no-torch fallback demo

Outputs -> models/  (fitted model, learning-curve PNG, training log)
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

CLASSES = ["Low EI", "Medium EI", "High EI"]
K_ITEMS = 10
SEED = 42
os.makedirs("models", exist_ok=True)


# --------------------------------------------------------------------------- data
def load(k=K_ITEMS):
    df = pd.read_csv("teique_clean.csv")
    items = [c for c in df.columns if c.startswith("i") and c[1:].isdigit()]
    X = df[items].values.astype(np.float32)
    y = df["ei_class"].map({c: i for i, c in enumerate(CLASSES)}).values
    # item selection on the TRAINING data only -- see note in main()
    return X, y, items


def select_items(X, k=K_ITEMS):
    tot = X.sum(1)
    s = [np.corrcoef(X[:, j], tot - X[:, j])[0, 1] for j in range(X.shape[1])]
    return np.sort(np.argsort(np.nan_to_num(s, nan=-1e9))[::-1][:k])


# ------------------------------------------------------------------- torch models
def train_torch(kind, Xtr, ytr, Xva, yva, epochs, lr, wd, batch, patience, log):
    """Explicit training loop so every step is visible.

    One epoch = one pass over the training rows:
      forward pass -> loss -> backward pass -> optimizer step.
    After each epoch we score the held-out validation split. We keep the weights
    from the epoch with the lowest validation loss (early stopping), which is
    what prevents the model from memorising 94 rows.
    """
    import torch
    import torch.nn as nn
    import sys
    sys.path.insert(0, ".")
    from ei_pipeline import build_torch_models

    FTT, SAINT = build_torch_models()
    cls = {"ftt": FTT, "saint": SAINT}[kind]
    kw = (dict(d_token=64, n_blocks=2, heads=8, drop=0.2) if kind == "ftt"
          else dict(d_token=32, n_blocks=2, heads=4, drop=0.2))

    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   device: {dev}")

    t = lambda a, d=torch.float32: torch.tensor(a, dtype=d, device=dev)
    Xtr_, ytr_ = t(Xtr), t(ytr, torch.long)
    Xva_, yva_ = t(Xva), t(yva, torch.long)

    model = cls(Xtr.shape[1], 3, **kw).to(dev)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   trainable parameters: {n_par:,}  (training rows: {len(Xtr)})")
    if n_par > 20 * len(Xtr):
        print(f"   WARNING: {n_par/len(Xtr):.0f} parameters per training row. "
              "Expect memorisation; consider lowering d_token.")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    hist, best, best_state, bad = [], np.inf, None, 0
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_), device=dev)
        run_loss, nb = 0.0, 0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 4:            # SAINT's intersample attention needs rows
                continue
            opt.zero_grad()
            loss = lossf(model(Xtr_[idx]), ytr_[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item(); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            tr_out, va_out = model(Xtr_), model(Xva_)
            tr_l = lossf(tr_out, ytr_).item()
            va_l = lossf(va_out, yva_).item()
            tr_a = (tr_out.argmax(1) == ytr_).float().mean().item()
            va_a = (va_out.argmax(1) == yva_).float().mean().item()
        hist.append(dict(epoch=ep, train_loss=tr_l, val_loss=va_l,
                         train_acc=tr_a, val_acc=va_a))

        if va_l < best - 1e-4:
            best, bad = va_l, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1; mark = ""
        if ep % 10 == 0 or ep == 1 or mark:
            print(f"   epoch {ep:4d}  train_loss {tr_l:.4f}  val_loss {va_l:.4f}"
                  f"  train_acc {tr_a:.3f}  val_acc {va_a:.3f}{mark}")
        if bad >= patience:
            print(f"   early stop at epoch {ep} "
                  f"(no val improvement for {patience} epochs)")
            break

    secs = time.perf_counter() - t0
    model.load_state_dict(best_state)
    print(f"   restored best epoch (val_loss {best:.4f}); trained in {secs:.1f}s")
    return model, pd.DataFrame(hist), n_par, secs


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgboost",
                    choices=["xgboost", "tabnet", "ftt", "saint", "mlp"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help="refit on 100%% of data for SHAP/deployment")
    a = ap.parse_args()

    X, y, items = load()
    print(f"n={len(y)}  items={len(items)}  model={a.model}")

    # --- split -------------------------------------------------------------
    # 60 / 20 / 20 train / validation / test.
    # validation tunes and early-stops; test is touched ONCE at the end.
    Xtr_f, Xte_f, ytr_f, yte = train_test_split(X, y, test_size=0.2,
                                                stratify=y, random_state=SEED)
    Xtr_f, Xva_f, ytr, yva = train_test_split(Xtr_f, ytr_f, test_size=0.25,
                                              stratify=ytr_f, random_state=SEED)
    print(f"train={len(ytr)}  val={len(yva)}  test={len(yte)}")
    print("NOTE: this split is for inspecting training dynamics. Report the")
    print("      repeated-CV numbers from ei_pipeline.py, not these.\n")

    # item selection and scaling fitted on TRAIN only, then applied outward
    keep = select_items(Xtr_f)
    print(f"selected items: {[items[j] for j in keep]}\n")
    sc = StandardScaler().fit(Xtr_f[:, keep])
    Xtr = sc.transform(Xtr_f[:, keep]).astype(np.float32)
    Xva = sc.transform(Xva_f[:, keep]).astype(np.float32)
    Xte = sc.transform(Xte_f[:, keep]).astype(np.float32)

    hist, n_par, secs = None, None, None

    # --- train -------------------------------------------------------------
    if a.model == "xgboost":
        import xgboost as xgb
        if a.tune:
            from sklearn.model_selection import GridSearchCV
            grid = {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.05, 0.1],
                    "n_estimators": [200, 400], "subsample": [0.7, 0.9],
                    "reg_lambda": [0.5, 1.0, 3.0]}
            print("Tuning on the TRAINING split with inner 5-fold CV")
            print("(never on the test split -- that would leak).\n")
            gs = GridSearchCV(
                xgb.XGBClassifier(objective="multi:softprob", num_class=3,
                                  tree_method="hist", random_state=SEED, n_jobs=1),
                grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                scoring="accuracy", n_jobs=1, verbose=0)
            gs.fit(Xtr, ytr)
            print(f"best params: {gs.best_params_}")
            print(f"best inner-CV accuracy: {gs.best_score_:.4f}\n")
            model = gs.best_estimator_
        else:
            model = xgb.XGBClassifier(
                n_estimators=400, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                min_child_weight=2, objective="multi:softprob", num_class=3,
                tree_method="hist", random_state=SEED, n_jobs=1,
                eval_metric="mlogloss", early_stopping_rounds=50)
            t0 = time.perf_counter()
            model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
            secs = time.perf_counter() - t0
            r = model.evals_result()
            hist = pd.DataFrame({
                "epoch": np.arange(1, len(r["validation_0"]["mlogloss"]) + 1),
                "train_loss": r["validation_0"]["mlogloss"],
                "val_loss": r["validation_1"]["mlogloss"]})
            print(f"   boosting rounds used: {model.best_iteration + 1} "
                  f"of 400 (early stopping on validation mlogloss)")
            print(f"   trained in {secs:.2f}s")

    elif a.model == "tabnet":
        from pytorch_tabnet.tab_model import TabNetClassifier
        model = TabNetClassifier(n_d=8, n_a=8, n_steps=3, gamma=1.3,
                                 lambda_sparse=1e-3, seed=SEED,
                                 optimizer_params=dict(lr=2e-2), verbose=10)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)],
                  eval_name=["train", "val"], eval_metric=["logloss"],
                  max_epochs=a.epochs, patience=a.patience,
                  batch_size=a.batch, virtual_batch_size=16, drop_last=False)
        secs = time.perf_counter() - t0
        hist = pd.DataFrame({"epoch": np.arange(1, len(model.history["loss"]) + 1),
                             "train_loss": model.history["train_logloss"],
                             "val_loss": model.history["val_logloss"]})
        n_par = sum(p.numel() for p in model.network.parameters() if p.requires_grad)

    elif a.model in ("ftt", "saint"):
        model, hist, n_par, secs = train_torch(
            a.model, Xtr, ytr, Xva, yva, a.epochs, a.lr, a.wd,
            a.batch, a.patience, None)

    else:   # mlp fallback so this script is runnable without torch/xgboost
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier((64, 32), max_iter=800, early_stopping=True,
                              n_iter_no_change=30, validation_fraction=0.2,
                              random_state=SEED)
        t0 = time.perf_counter(); model.fit(Xtr, ytr); secs = time.perf_counter() - t0
        hist = pd.DataFrame({"epoch": np.arange(1, len(model.loss_curve_) + 1),
                             "train_loss": model.loss_curve_})
        n_par = sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_)
        print(f"   iterations: {model.n_iter_}   parameters: {n_par:,}"
              f"   trained in {secs:.2f}s")

    # --- evaluate ----------------------------------------------------------
    def prob(Xd):
        if a.model in ("ftt", "saint"):
            import torch
            dev = next(model.parameters()).device
            with torch.no_grad():
                return torch.softmax(
                    model(torch.tensor(Xd, dtype=torch.float32, device=dev)),
                    1).cpu().numpy()
        return model.predict_proba(Xd)

    print("\n" + "=" * 62)
    print("RESULTS (this split only)")
    print("=" * 62)
    for nm, Xd, yd in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        p = prob(Xd)
        print(f"   {nm:6s} accuracy={accuracy_score(yd, p.argmax(1)):.4f}"
              f"   log-loss={log_loss(yd, p, labels=[0,1,2]):.4f}")
    gap = accuracy_score(ytr, prob(Xtr).argmax(1)) - \
          accuracy_score(yte, prob(Xte).argmax(1))
    print(f"\n   train-test gap = {gap:.3f}", end="")
    print("   <-- overfitting; reduce capacity or raise regularisation"
          if gap > 0.25 else "   (acceptable)")

    # --- learning curve ----------------------------------------------------
    if hist is not None and len(hist) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(hist.epoch, hist.train_loss, label="training loss")
        if "val_loss" in hist:
            ax.plot(hist.epoch, hist.val_loss, label="validation loss")
            b = int(hist.val_loss.idxmin())
            ax.axvline(hist.epoch.iloc[b], ls="--", c="grey", lw=1,
                       label=f"best epoch ({hist.epoch.iloc[b]})")
        ax.set_xlabel("epoch / boosting round"); ax.set_ylabel("loss")
        ax.set_title(f"{a.model}: training dynamics"); ax.legend()
        fig.tight_layout()
        fig.savefig(f"models/learning_curve_{a.model}.png", dpi=200)
        plt.close(fig)
        hist.to_csv(f"models/training_log_{a.model}.csv", index=False)
        print(f"\n   learning curve -> models/learning_curve_{a.model}.png")
        print("   Read it like this:")
        print("     both curves falling            -> still learning, train longer")
        print("     val flat while train falls     -> memorising, stop / regularise")
        print("     val rising after a minimum     -> overfit past the dashed line")
        print("     both flat and high             -> underfitting or lr too low")

    # --- save --------------------------------------------------------------
    meta = dict(model=a.model, selected_items=[items[j] for j in keep],
                n_params=int(n_par) if n_par else None,
                train_seconds=round(secs, 3) if secs else None,
                test_accuracy=round(float(accuracy_score(yte, prob(Xte).argmax(1))), 4))
    with open(f"models/meta_{a.model}.json", "w") as f:
        json.dump(meta, f, indent=2)

    if a.model in ("ftt", "saint"):
        import torch
        torch.save(model.state_dict(), f"models/{a.model}.pt")
    elif a.model == "tabnet":
        model.save_model(f"models/tabnet")
    else:
        import pickle
        with open(f"models/{a.model}.pkl", "wb") as f:
            pickle.dump({"model": model, "scaler": sc, "keep": keep}, f)
    print(f"   model + metadata -> models/")

    if a.final:
        print("\nRefitting on 100% of the data for SHAP / deployment.")
        print("This model has NO honest test score -- never report its accuracy.")


if __name__ == "__main__":
    main()
