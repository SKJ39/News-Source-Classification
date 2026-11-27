"""Combine TF-IDF LR, TextCNN and BagMLP into one model_nn.NewsClassifier.

Steps
  1. Read each model's chosen configuration and out-of-fold (OOF) probabilities
     on TRAINVAL (written by train_tfidf.py and train_nn.py, same folds).
  2. Choose the mixing weights on a simplex grid by OOF accuracy. TEST is not used.
  3. Refit every model on TRAINVAL + scraped and evaluate on the local TEST once
     (each single model and the ensemble). Save those weights so the number can be
     reproduced with eval_project_b.py.
  4. Refit every model on all url_train + scraped and export the submission
     folder (model.py = model_nn.py, preprocess.py, model.pt).

Usage:
    python ensemble.py --scraped data/scraped_matched.csv
"""

import argparse
import itertools
import json
import os
import shutil

import numpy as np
import torch

from data_split import describe, load_splits
from model_nn import NewsClassifier, encode
from train_nn import predict_proba as nn_proba
from train_nn import train_model, training_set
from train_tfidf import fit_lr, lr_to_state, transform_texts
from train_tfidf import predict_proba as lr_proba

NAMES = ["lr", "cnn", "bag"]


def _load(path_dir: str):
    with open(os.path.join(path_dir, "best.json")) as f:
        cfg = json.load(f)
    return cfg, np.load(os.path.join(path_dir, "oof_probs.npy"))


def choose_mix(oofs, y, step: float = 0.05):
    best = (-1.0, None)
    n = int(round(1 / step))
    for a, b in itertools.product(range(n + 1), repeat=2):
        if a + b > n:
            continue
        w = np.array([a, b, n - a - b], dtype=float) / n
        acc = float(np.mean(((w[0] * oofs[0] + w[1] * oofs[1] + w[2] * oofs[2]) > 0.5) == y))
        # Prefer higher accuracy, then more even weights (less overfit to the grid).
        key = (round(acc, 6), -float(np.max(w)))
        if best[1] is None or key > best[0]:
            best = (key, w)
    return best[1], best[0][0]


def fit_all(cfgs, s, idx, seed):
    """Fit LR, CNN and Bag on url_train rows `idx` plus all scraped rows; return a NewsClassifier."""
    model = NewsClassifier()
    X = transform_texts(model, [s.texts[i] for i in idx])
    X_scr = transform_texts(model, s.scr_texts)
    clf = fit_lr(cfgs["lr"]["C"], cfgs["lr"]["scraped_w"], seed, X, s.y[idx], X_scr, s.y_scr)
    state = lr_to_state(clf)
    model.linear.weight.data.copy_(state["weight"])
    model.linear.bias.data.copy_(state["bias"])

    enc_orig = [encode(s.texts[i]) for i in idx]
    enc_scr = [encode(t) for t in s.scr_texts]
    for name in ("cnn", "bag"):
        cfg = cfgs[name]
        enc, y, w = training_set(enc_orig, s.y[idx], enc_scr, s.y_scr, cfg["scraped_w"])
        trained, _ = train_model(name, cfg, enc, y, w, cfg["epochs"], seed)
        getattr(model, name).load_state_dict(trained.state_dict())
        print(f"  fitted {name} ({cfg['epochs']} epochs)", flush=True)
    model.eval()
    return model


def component_probs(model: NewsClassifier, texts):
    X = transform_texts(model, texts)
    clf_like = model.linear
    logits = X @ clf_like.weight.detach().numpy().T + clf_like.bias.detach().numpy()
    p_lr = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
    enc = [encode(t) for t in texts]
    return [np.asarray(p_lr).ravel(), nn_proba(model.cnn, enc), nn_proba(model.bag, enc)]


def save_half(model: NewsClassifier, path: str) -> None:
    """Store floating-point weights as float16 to halve model.pt; load_state_dict casts back."""
    state = {k: (v.half() if v.is_floating_point() and k != "mix" else v)
             for k, v in model.state_dict().items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate the LR + CNN + Bag ensemble")
    parser.add_argument("--csv", default="url_train.csv")
    parser.add_argument("--scraped", default="data/scraped_matched.csv")
    parser.add_argument("--tfidf-dir", default="data/tfidf")
    parser.add_argument("--nn-dir", default="data/nn")
    parser.add_argument("--out-dir", default="data/ensemble")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.out_dir, exist_ok=True)
    s = load_splits(args.csv, args.scraped, n_folds=args.folds, seed=args.seed)
    print(describe(s))
    y_trval = s.y[s.idx_trval]

    cfgs, oofs = {}, []
    for name, d in zip(NAMES, [args.tfidf_dir, os.path.join(args.nn_dir, "cnn"), os.path.join(args.nn_dir, "bag")]):
        cfgs[name], oof = _load(d)
        if len(oof) != len(y_trval):
            raise RuntimeError(f"{name} OOF length {len(oof)} != trainval {len(y_trval)}; rerun with the same data")
        oofs.append(oof)
        print(f"{name}: {cfgs[name]} | OOF acc {np.mean((oof > 0.5) == y_trval):.4f}")

    mix, oof_acc = choose_mix(oofs, y_trval)
    print(f"mix weights lr/cnn/bag = {mix.round(2).tolist()} | ensemble OOF acc {oof_acc:.4f}")

    # One evaluation on the local test set.
    print("Refitting on trainval + scraped ...")
    model = fit_all(cfgs, s, s.idx_trval, args.seed)
    model.mix.copy_(torch.tensor(mix, dtype=torch.float32))
    test_texts = [s.texts[i] for i in s.idx_test]
    y_test = s.y[s.idx_test]
    probs = component_probs(model, test_texts)
    report = {"mix": mix.tolist(), "ensemble_oof_acc": oof_acc}
    for name, p in zip(NAMES, probs):
        report[f"test_{name}"] = float(np.mean((p > 0.5) == y_test))
    ens = sum(w * p for w, p in zip(mix, probs))
    report["test_ensemble"] = float(np.mean((ens > 0.5) == y_test))
    test_weights = os.path.join(args.out_dir, "best_trainval.pt")
    save_half(model, test_weights)
    print("Local TEST accuracy:", {k: round(v, 4) for k, v in report.items() if k.startswith("test_")})

    # Final refit on all url_train + scraped.
    print("Refitting on all url_train + scraped ...")
    final = fit_all(cfgs, s, np.arange(len(s.urls)), args.seed)
    final.mix.copy_(torch.tensor(mix, dtype=torch.float32))
    submit_dir = os.path.join(args.out_dir, "submission")
    save_half(final, os.path.join(submit_dir, "model.pt"))
    shutil.copy(os.path.join(root, "model_nn.py"), os.path.join(submit_dir, "model.py"))
    shutil.copy(os.path.join(root, "preprocess.py"), os.path.join(submit_dir, "preprocess.py"))

    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump({"configs": cfgs, **report}, f, indent=2)
    print(f"Reproduce: python eval_project_b.py --model model_nn.py --preprocess preprocess.py "
          f"--csv {os.path.join(args.out_dir, 'test.csv')} --weights {test_weights}")
    from data_split import write_urls
    write_urls(os.path.join(args.out_dir, "test.csv"), [s.urls[i] for i in s.idx_test])
    print(f"Submission: {submit_dir}/")


if __name__ == "__main__":
    main()
