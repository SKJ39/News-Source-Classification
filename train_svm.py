"""Train a linear SVM over hashed word and character n-gram features."""

import argparse
import os
import random
from typing import Sequence

import numpy as np
import torch
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from build_data import build_training_data
from model_svm import NewsClassifier


def _encode_labels(labels: Sequence[str]) -> torch.Tensor:
    """
    Encode string labels as integers.

    Parameters:
    labels (Sequence[str]): List of string labels, expected to be "FoxNews" or "NBC".
    """
    classes = sorted(set(labels))
    if classes != ["FoxNews", "NBC"]:
        raise RuntimeError(
            f"Training data must contain FoxNews and NBC labels; found {classes}"
        )
    label_to_index = {label: index for index, label in enumerate(classes)}
    return torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)


def _transform_texts(model: NewsClassifier, texts: Sequence[str]):
    """
    Transform a sequence of texts into a sparse feature matrix using the model's vectorizers.
    
    Parameters:
    model (NewsClassifier): The NewsClassifier instance with TF-IDF vectorizers.
    texts (Sequence[str]): Sequence of text data to transform.
    """
    text_list = [str(text) for text in texts]
    word_features = model.word.transform(text_list)
    char_features = model.char.transform(text_list)
    return hstack([word_features, char_features], format="csr")


def _fit(model: NewsClassifier, texts: Sequence[str], labels: torch.Tensor, c_value: float, seed: int) -> LinearSVC:
    """
    Fit a linear SVM classifier to the given texts and labels.

    Parameters:
    model (NewsClassifier): The NewsClassifier instance with TF-IDF vectorizers.
    texts (Sequence[str]): Sequence of text data to transform.
    labels (torch.Tensor): Tensor of integer labels.
    c_value (float): Regularization parameter for the SVM.
    seed (int): Random seed for reproducibility.
    """
    features = _transform_texts(model, texts)
    classifier = LinearSVC(C=c_value, dual="auto", max_iter=5000, random_state=seed)
    with threadpool_limits(limits=1):
        classifier.fit(features, labels.numpy())
    return classifier


def _copy_classifier_weights(model: NewsClassifier, classifier: LinearSVC) -> None:
    """
    Copy the weights from a fitted LinearSVC classifier to the model's linear layer.

    Parameters:
    model (NewsClassifier): The NewsClassifier instance to update.
    classifier (LinearSVC): The fitted LinearSVC classifier from which to copy weights.
    """
    if classifier.classes_.tolist() != [0, 1]:
        raise RuntimeError(f"Unexpected encoded classes: {classifier.classes_.tolist()}")

    coef = classifier.coef_[0].astype(np.float32)
    bias = np.float32(classifier.intercept_[0])
    model.linear.weight.data.copy_(torch.from_numpy(np.vstack([-0.5 * coef, 0.5 * coef])))
    model.linear.bias.data.copy_(torch.tensor([-0.5 * bias, 0.5 * bias]))


def train(model: NewsClassifier, texts_train: Sequence[str], labels_train: torch.Tensor, texts_val: Sequence[str], 
          labels_val: torch.Tensor, c_value: float = 1.0, seed: int = 42,) -> None:
    """
    Fit the SVM, report held-out accuracy, and store full-data weights.
    
    Parameters:
    model (NewsClassifier): The NewsClassifier instance with TF-IDF vectorizers.
    texts_train (Sequence[str]): Training text data.
    labels_train (torch.Tensor): Training labels.
    texts_val (Sequence[str]): Validation text data.
    labels_val (torch.Tensor): Validation labels.
    c_value (float): Regularization parameter for the SVM.
    seed (int): Random seed for reproducibility.
    """
    if c_value <= 0:
        raise ValueError("C must be greater than zero")
    if len(set(labels_train.tolist())) != 2:
        raise RuntimeError("Training data must contain both FoxNews and NBC labels")

    validation_classifier = _fit(model, texts_train, labels_train, c_value, seed)
    validation_features = _transform_texts(model, texts_val)
    with threadpool_limits(limits=1):
        predictions = validation_classifier.predict(validation_features)
    accuracy = float(np.mean(predictions == labels_val.numpy()))
    print(f"Linear SVM | validation accuracy: {accuracy:.4f}")

    all_texts = list(texts_train) + list(texts_val)
    all_labels = torch.cat([labels_train, labels_val])
    final_classifier = _fit(model, all_texts, all_labels, c_value, seed)
    _copy_classifier_weights(model, final_classifier)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a hashed-feature linear SVM")
    parser.add_argument("--csv", default="url_train.csv", help="Training URL CSV")
    parser.add_argument("--output", default="./model_svm.pt", help="Path for the trained PyTorch state_dict")
    parser.add_argument("--c", type=float, default=1.0, help="LinearSVC regularization parameter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-online", action="store_true", help="Train only from the supplied CSV without scraping additional URLs")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    texts, labels = build_training_data(args.csv, n_drop_nbc=0, fetch_online=not args.no_online)
    if not texts:
        raise RuntimeError("No training rows were produced from the supplied CSV.")

    examples = sorted(zip(texts, labels))
    texts = [text for text, _ in examples]
    labels = [label for _, label in examples]
    all_labels = _encode_labels(labels)
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, all_labels, test_size=0.2, 
                                                                        random_state=args.seed, stratify=all_labels)

    model = NewsClassifier()
    train(model, train_texts, train_labels, val_texts, val_labels, c_value=args.c, seed=args.seed)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    main()