"""Train TextCNN / BagMLP from model_nn.py with leakage-free 5-fold CV tuning.

Splits come from data_split.py (fixed local TEST + 5-fold CV on TRAINVAL;
scraped rows only on the training side, weighted by scraped_w in the loss).

The number of epochs is a hyperparameter: each configuration is trained for
--max-epochs in every fold, validation accuracy is recorded after every epoch,
and the epoch with the best mean accuracy across folds is selected.

Outputs (in --out-dir/<arch>):
    cv_results.csv   every configuration's mean CV accuracy at its best epoch
    best.json        chosen configuration (incl. epochs)
    oof_probs.npy    out-of-fold P(NBC) on TRAINVAL for the chosen config (for ensemble.py)

The local test evaluation and the final refit happen in ensemble.py.

Usage:
    python train_nn.py --arch cnn
    python train_nn.py --arch bag
"""

import argparse
import csv
import itertools
import json
import os
import random
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from data_split import describe, load_splits
from model_nn import BagMLP, TextCNN, collate, encode

GRIDS = {
    "cnn": {"lr": [2e-3], "dropout": [0.3, 0.5], "scraped_w": [0.5, 1.0]},
    "bag": {"lr": [3e-3, 1e-2], "dropout": [0.2, 0.4], "scraped_w": [0.5, 1.0]},
}
BATCH_SIZE = 128


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build(arch: str, dropout: float) -> torch.nn.Module:
    return TextCNN(dropout) if arch == "cnn" else BagMLP(dropout)


def predict_proba(model: torch.nn.Module, enc: Sequence, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(enc), batch_size):
            logits = model(*collate(enc[i:i + batch_size]))
            out.append(F.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_model(arch: str, cfg: Dict, enc: List, y: np.ndarray, weights: np.ndarray, epochs: int,
                seed: int, val_enc: List = None, val_y: np.ndarray = None):
    """Train for `epochs`; if validation data is given, return per-epoch val probabilities too."""
    set_seed(seed)
    model = build(arch, cfg["dropout"])
    # Sparse-gradient embeddings use SparseAdam; everything else uses Adam.
    sparse = [p for m in model.modules() if getattr(m, "sparse", False) for p in m.parameters()]
    sparse_ids = {id(p) for p in sparse}
    dense = [p for p in model.parameters() if id(p) not in sparse_ids]
    opts = [torch.optim.Adam(dense, lr=cfg["lr"]), torch.optim.SparseAdam(sparse, lr=cfg["lr"])]
    y_t = torch.tensor(y, dtype=torch.long)
    w_t = torch.tensor(weights, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    val_probs = []
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(enc))
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            logits = model(*collate([enc[j] for j in idx]))
            loss = F.cross_entropy(logits, y_t[idx], reduction="none")
            loss = (loss * w_t[idx]).sum() / w_t[idx].sum()
            for opt in opts:
                opt.zero_grad()
            loss.backward()
            for opt in opts:
                opt.step()
        if val_enc is not None:
            val_probs.append(predict_proba(model, val_enc))
    return model, val_probs


def training_set(enc_orig, y_orig, enc_scr, y_scr, w):
    enc = list(enc_orig) + list(enc_scr)
    y = np.concatenate([y_orig, y_scr])
    weights = np.concatenate([np.ones(len(y_orig)), np.full(len(y_scr), w)])
    return enc, y, weights


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold CV tuning for the neural models")
    parser.add_argument("--arch", choices=["cnn", "bag"], required=True)
    parser.add_argument("--csv", default="url_train.csv")
    parser.add_argument("--scraped", default="data/scraped_matched.csv")
    parser.add_argument("--out-dir", default="data/nn")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = os.path.join(args.out_dir, args.arch)
    os.makedirs(out_dir, exist_ok=True)

    s = load_splits(args.csv, args.scraped, n_folds=args.folds, seed=args.seed)
    print(describe(s))
    enc_all = [encode(t) for t in s.texts]
    enc_scr = [encode(t) for t in s.scr_texts]
    enc_trval = [enc_all[i] for i in s.idx_trval]
    y_trval = s.y[s.idx_trval]

    grid = GRIDS[args.arch]
    keys = list(grid)
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, values))
        start = time.time()
        # acc[f, e] = accuracy of fold f after epoch e+1; probs[e] = OOF probabilities after epoch e+1
        acc = np.zeros((len(s.folds), args.max_epochs))
        probs = np.zeros((args.max_epochs, len(y_trval)))
        for f, (tr, va) in enumerate(s.folds):
            enc, y, w = training_set([enc_trval[i] for i in tr], y_trval[tr], enc_scr, s.y_scr,
                                     cfg["scraped_w"])
            _, val_probs = train_model(args.arch, cfg, enc, y, w, args.max_epochs, args.seed + f,
                                       [enc_trval[i] for i in va], y_trval[va])
            for e, p in enumerate(val_probs):
                acc[f, e] = np.mean((p > 0.5) == y_trval[va])
                probs[e, va] = p
        mean_by_epoch = acc.mean(axis=0)
        best_e = int(np.argmax(mean_by_epoch))
        r = dict(cfg, epochs=best_e + 1, cv_mean=float(mean_by_epoch[best_e]),
                 cv_std=float(acc[:, best_e].std()), by_epoch=mean_by_epoch.tolist(), oof=probs[best_e])
        results.append(r)
        curve = " ".join(f"{a:.3f}" for a in mean_by_epoch)
        print(f"{cfg} -> best epoch {best_e + 1}: cv={r['cv_mean']:.4f} ± {r['cv_std']:.4f} "
              f"| by epoch: {curve} ({time.time() - start:.0f}s)", flush=True)

    with open(os.path.join(out_dir, "cv_results.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(keys + ["epochs", "cv_mean", "cv_std"]
                        + [f"epoch{e + 1}" for e in range(args.max_epochs)])
        for r in results:
            writer.writerow([r[k] for k in keys] + [r["epochs"], f"{r['cv_mean']:.4f}", f"{r['cv_std']:.4f}"]
                            + [f"{a:.4f}" for a in r["by_epoch"]])

    best = max(results, key=lambda r: r["cv_mean"])
    np.save(os.path.join(out_dir, "oof_probs.npy"), best["oof"])
    chosen = {k: best[k] for k in keys + ["epochs", "cv_mean", "cv_std"]}
    with open(os.path.join(out_dir, "best.json"), "w") as fh:
        json.dump(chosen, fh, indent=2)
    print(f"\nBest {args.arch}: {chosen}")


if __name__ == "__main__":
    main()
