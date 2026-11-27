"""Train the hashed word+char TF-IDF logistic-regression model with leakage-free tuning.

Splits come from data_split.py (fixed local TEST + 5-fold CV on TRAINVAL;
scraped rows only on the training side). Hyperparameters chosen by mean CV
accuracy:
    C          logistic-regression inverse regularization
    scraped_w  sample weight of scraped rows (always > 0)

Outputs (in --out-dir):
    cv_results.csv     every configuration's fold scores
    best.json          chosen configuration
    oof_probs.npy      out-of-fold P(NBC) on TRAINVAL for the chosen config (for ensemble.py)
    best_trainval.pt   chosen config refit on TRAINVAL + scraped (local test number)
    submission/        chosen config refit on all url_train + scraped

Usage:
    python train_tfidf.py --scraped data/scraped_matched.csv
"""

import argparse
import csv
import json
import os
import random
import shutil
import time
from typing import Sequence

import numpy as np
import torch
from scipy.sparse import hstack, vstack
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from data_split import describe, load_splits, save_split_files
from model_tfidf import NewsClassifier

C_GRID = [0.3, 1.0, 3.0, 10.0, 30.0]
SCRAPED_W_GRID = [0.03, 0.1, 0.25, 0.5, 1.0]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def transform_texts(model: NewsClassifier, texts: Sequence[str]):
    text_list = [str(text) for text in texts]
    word_features = model.word.transform(text_list)
    char_features = model.char.transform(text_list)
    return hstack([word_features, char_features], format="csr")


def fit_lr(c: float, w: float, seed: int, X_orig, y_orig, X_scr, y_scr) -> LogisticRegression:
    """Logistic regression on url_train rows (weight 1) plus scraped rows (weight w)."""
    X = vstack([X_orig, X_scr], format="csr")
    y = np.concatenate([y_orig, y_scr])
    sw = np.concatenate([np.ones(len(y_orig)), np.full(len(y_scr), w)])
    clf = LogisticRegression(C=c, solver="lbfgs", max_iter=2000, random_state=seed)
    with threadpool_limits(limits=1):
        clf.fit(X, y, sample_weight=sw)
    if clf.classes_.tolist() != [0, 1]:
        raise RuntimeError(f"Unexpected encoded classes: {clf.classes_.tolist()}")
    return clf


def predict_proba(clf: LogisticRegression, X) -> np.ndarray:
    with threadpool_limits(limits=1):
        return clf.predict_proba(X)[:, 1]


def lr_to_state(clf: LogisticRegression) -> dict:
    """Two logits encode the binary decision boundary coef*x + bias."""
    coef = clf.coef_[0].astype(np.float32)
    bias = float(clf.intercept_[0])
    return {"weight": torch.from_numpy(np.vstack([-0.5 * coef, 0.5 * coef])),
            "bias": torch.tensor([-0.5 * bias, 0.5 * bias], dtype=torch.float32)}


def _save_weights(clf: LogisticRegression, path: str) -> None:
    model = NewsClassifier()
    state = lr_to_state(clf)
    model.linear.weight.data.copy_(state["weight"])
    model.linear.bias.data.copy_(state["bias"])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-free CV tuning for the TF-IDF model")
    parser.add_argument("--csv", default="url_train.csv")
    parser.add_argument("--scraped", default="data/scraped_matched.csv")
    parser.add_argument("--out-dir", default="data/tfidf")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.out_dir, exist_ok=True)

    s = load_splits(args.csv, args.scraped, n_folds=args.folds, seed=args.seed)
    save_split_files(s, args.out_dir)
    print(describe(s))

    featurizer = NewsClassifier()
    X_all = transform_texts(featurizer, s.texts)
    X_scr = transform_texts(featurizer, s.scr_texts)
    X_trval, y_trval = X_all[s.idx_trval], s.y[s.idx_trval]
    X_test, y_test = X_all[s.idx_test], s.y[s.idx_test]

    results = []
    for w in SCRAPED_W_GRID:
        for c in C_GRID:
            start = time.time()
            oof = np.zeros(len(y_trval))
            scores = []
            for tr, va in s.folds:
                clf = fit_lr(c, w, args.seed, X_trval[tr], y_trval[tr], X_scr, s.y_scr)
                oof[va] = predict_proba(clf, X_trval[va])
                scores.append(float(np.mean((oof[va] > 0.5) == y_trval[va])))
            mean, std = float(np.mean(scores)), float(np.std(scores))
            results.append({"C": c, "scraped_w": w, "cv_mean": mean, "cv_std": std,
                            "folds": scores, "oof": oof})
            print(f"scraped_w={w:<5} C={c:<5} cv={mean:.4f} ± {std:.4f}  ({time.time() - start:.1f}s)")

    with open(os.path.join(args.out_dir, "cv_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scraped_w", "C", "cv_mean", "cv_std"]
                        + [f"fold{i + 1}" for i in range(args.folds)])
        for r in results:
            writer.writerow([r["scraped_w"], r["C"], f"{r['cv_mean']:.4f}", f"{r['cv_std']:.4f}"]
                            + [f"{x:.4f}" for x in r["folds"]])

    best = max(results, key=lambda r: r["cv_mean"])
    c, w = best["C"], best["scraped_w"]
    np.save(os.path.join(args.out_dir, "oof_probs.npy"), best["oof"])
    with open(os.path.join(args.out_dir, "best.json"), "w") as f:
        json.dump({"C": c, "scraped_w": w, "cv_mean": best["cv_mean"], "cv_std": best["cv_std"]}, f, indent=2)
    print(f"\nBest: C={c} scraped_w={w} cv={best['cv_mean']:.4f} ± {best['cv_std']:.4f}")

    clf = fit_lr(c, w, args.seed, X_trval, y_trval, X_scr, s.y_scr)
    test_acc = float(np.mean((predict_proba(clf, X_test) > 0.5) == y_test))
    test_weights = os.path.join(args.out_dir, "best_trainval.pt")
    _save_weights(clf, test_weights)
    print(f"Local TEST accuracy (trained on trainval + scraped): {test_acc:.4f}")

    clf_full = fit_lr(c, w, args.seed, X_all, s.y, X_scr, s.y_scr)
    submit_dir = os.path.join(args.out_dir, "submission")
    _save_weights(clf_full, os.path.join(submit_dir, "model.pt"))
    shutil.copy(os.path.join(root, "model_tfidf.py"), os.path.join(submit_dir, "model.py"))
    shutil.copy(os.path.join(root, "preprocess.py"), os.path.join(submit_dir, "preprocess.py"))
    print(f"Reproduce: python eval_project_b.py --model model_tfidf.py --preprocess preprocess.py "
          f"--csv {os.path.join(args.out_dir, 'test.csv')} --weights {test_weights}")


if __name__ == "__main__":
    main()
