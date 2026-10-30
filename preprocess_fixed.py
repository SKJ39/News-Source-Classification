"""Inference-only preprocessing for CIS 5190 Project B.

The evaluator supplies headline slugs with source information removed.
Preserve every input row and its order. Collect training URLs separately.
"""
import csv
import os
from typing import List, Tuple


def prepare_data(path: str) -> Tuple[List[str], List[str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    inputs: List[str] = []
    labels: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if index == 0 and row and row[0].strip().lower() == "url":
                continue
            # One model input for each row, even for blank/malformed values.
            text = row[0].strip() if row else ""
            inputs.append(text)
            labels.append("")  # The evaluator holds ground-truth labels separately.
    return inputs, labels
