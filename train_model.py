import argparse
import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.sparse import hstack, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from model_tfidf import NewsClassifier
from build_data import build_training_data


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode_labels(labels: Sequence[str]) -> Tuple[torch.Tensor, dict]:
    """
    Map labels to ints. Expected labels: 'FoxNews' and 'NBC'.
    """
    unique = sorted(list({str(l) for l in labels}))
    mapping = {lab: i for i, lab in enumerate(unique)}
    y = torch.tensor([mapping[str(l)] for l in labels], dtype=torch.long)
    return y, mapping


def transform_texts(model: NewsClassifier, texts: Sequence[str]):
    text_list = [str(text) for text in texts]
    word_features = model.word.transform(text_list)
    char_features = model.char.transform(text_list)
    return hstack([word_features, char_features], format="csr")


def train(
    model: NewsClassifier,
    texts_train: Sequence[str],
    y_train: torch.Tensor,
    texts_val: Sequence[str],
    y_val: torch.Tensor,
    c_value: float = 3.0,
) -> None:
    """Fit the sparse TF-IDF classifier and refit its selected model on all data."""
    if c_value <= 0:
        raise ValueError("C must be greater than zero")

    if len(set(y_train.tolist())) != 2:
        raise RuntimeError("Training data must contain both FoxNews and NBC labels")

    train_features = transform_texts(model, texts_train)
    val_features = transform_texts(model, texts_val)

    classifier = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=500,
    )
    classifier.fit(train_features, y_train.numpy())

    val_acc = float(np.mean(classifier.predict(val_features) == y_val.numpy()))
    print(f"TF-IDF logistic regression | holdout acc {val_acc:.4f} | C {c_value:g}")

    all_features = vstack([train_features, val_features], format="csr")
    all_targets = torch.cat([y_train, y_val]).numpy()
    classifier.fit(all_features, all_targets)

    classes = classifier.classes_.tolist()
    if classes == [0, 1]:
        class1_index = 1
    else:
        raise RuntimeError(f"Unexpected encoded classes: {classes}")

    coef = classifier.coef_[class1_index - 1].astype(np.float32)
    bias = np.float32(classifier.intercept_[class1_index - 1])

    # Two logits encode the binary decision boundary classifier.coef*x+bias.
    model.linear.weight.data.copy_(torch.tensor(
        np.vstack([-0.5 * coef, 0.5 * coef]), dtype=torch.float32
    ))
    model.linear.bias.data.copy_(torch.tensor(
        [-0.5 * bias, 0.5 * bias], dtype=torch.float32
    ))


def main():
    parser = argparse.ArgumentParser(description="Train fixed-dimensional word+character TF-IDF classifier")
    parser.add_argument("--csv", type=str, default="url_train.csv", help="Path to the training URL CSV")
    parser.add_argument("--output", type=str, default="./model.pt", help="Output path to save trained model")
    parser.add_argument("--c", type=float, default=3.0, help="Logistic-regression inverse regularization strength")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-online", action="store_true", help="Train only from the supplied CSV")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load and prepare data
    texts, labels = build_training_data(args.csv,n_drop_nbc=0)
    if len(texts) == 0:
        raise RuntimeError("No data produced by preprocess. Ensure URLs map to known outlets.")

    y_all, mapping = encode_labels(labels)
    # Prefer class order FoxNews=0, NBC=1 if present
    # (mapping is only used for training; saved model predicts indices)

    X_train, X_val, y_train, y_val = train_test_split(texts, y_all, test_size=0.2, random_state=args.seed, stratify=y_all)

    # Build model
    model = NewsClassifier()

    # Train
    train(
        model=model,
        texts_train=X_train,
        y_train=y_train,
        texts_val=X_val,
        y_val=y_val,
        c_value=args.c,
    )

    # Save only the parameters required by model_tfidf.NewsClassifier.
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    main()
