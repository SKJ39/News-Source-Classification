import os
import csv
import re
import tempfile
import torch
from urllib.parse import unquote, urlparse
from typing import Any, List, Tuple

def _slug_to_path(path: str) -> str:
    segments = [s for s in path.split("/") if s]
    last = unquote(segments[-1]) if segments else ""
    last = re.sub(r"\.(html?|shtml|php)$", "", last, flags=re.IGNORECASE)
    words = [w for w in re.split(r"[-_]+", last) if w and not re.search(r"\d", w)]
    link = "/" + "-".join(words)
    return link

def _label_from_host(host: str) -> str:
    if "foxnews" in host:
        return "FoxNews"
    elif "nbcnews" in host or "nbc.news" in host or "nbc" in host:
        return "NBC"
    return ""

def _process(path: str, urls: List[str], labels: List[str]) -> None:
    with open(path, "r") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].strip().lower() == "url":
        rows = rows[1:]
    for row in rows:
        if not row or not row[0].strip():
            continue
        parsed = urlparse(row[0].strip())
        label = _label_from_host(parsed.netloc.lower())
        if parsed.netlock and not label:
            labels.append(label)
        else:
            continue
        link = _slug_to_path(parsed.path)
        urls.append(link)

def prepare_data(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Template preprocessing for leaderboard.

    Requirements:
    - Must read the provided data path at `path`.
    - Must return a tuple (X, y):
        X: a list of model-ready inputs (these must match what your model expects in predict(...))
        y: a list of ground-truth labels aligned with X (same length)

    Notes:
    - The evaluation backend will call this function with the shared validation data
    - Ensure the output format (types, shapes) of X matches your model's predict(...) inputs.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    urls: List[str] = []
    labels: List[str] = []
    
    _process(path, urls, labels)
    return urls, labels
