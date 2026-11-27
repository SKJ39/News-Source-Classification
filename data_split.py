"""Shared, leakage-free data splits used by train_tfidf.py, train_nn.py and ensemble.py.

  - url_train.csv is deduplicated by headline slug.
  - Local TEST: fixed stratified 20% of url_train (seed 42). Used once at the end.
  - TRAINVAL: the remaining 80%, split into stratified K folds for validation.
  - Scraped rows are deduplicated against every url_train slug (including TEST)
    and against each other. They are only ever added to the training side.

Every script calls load_splits() with the same arguments, so their folds and
out-of-fold predictions line up row by row.
"""

import csv
import os
from dataclasses import dataclass
from typing import List, Tuple
from urllib.parse import urlparse

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from build_data import _label_from_host, _slug_to_path

LABELS = {"FoxNews": 0, "NBC": 1}


def read_urls(path: str) -> List[str]:
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].strip().lower() == "url":
        rows = rows[1:]
    return [row[0].strip() for row in rows if row and row[0].strip()]


def write_urls(path: str, urls: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url"])
        writer.writerows([[u] for u in urls])


def to_examples(urls: List[str], seen: set) -> Tuple[List[str], List[str], np.ndarray]:
    """Keep labelled article URLs whose slug is not in `seen`; updates `seen` in place."""
    kept, slugs, labels = [], [], []
    for url in urls:
        parsed = urlparse(url)
        label = _label_from_host(parsed.netloc.lower())
        slug = _slug_to_path(parsed.path)
        if not label or slug == "/" or slug in seen:
            continue
        seen.add(slug)
        kept.append(url)
        slugs.append(slug)
        labels.append(LABELS[label])
    return kept, slugs, np.array(labels, dtype=np.int64)


@dataclass
class Splits:
    urls: List[str]            # deduplicated url_train URLs
    texts: List[str]           # their model inputs (headline slugs)
    y: np.ndarray
    idx_trval: np.ndarray      # indices into urls/texts/y
    idx_test: np.ndarray
    folds: List[Tuple[np.ndarray, np.ndarray]]  # (train, val) indices into idx_trval
    scr_urls: List[str]
    scr_texts: List[str]
    y_scr: np.ndarray


def load_splits(csv_path: str, scraped_path: str, test_size: float = 0.2,
                n_folds: int = 5, seed: int = 42) -> Splits:
    if not os.path.isfile(scraped_path):
        raise FileNotFoundError(f"{scraped_path} not found; run scrape_matched.py first.")
    seen: set = set()
    urls, texts, y = to_examples(read_urls(csv_path), seen)
    idx_trval, idx_test = train_test_split(
        np.arange(len(urls)), test_size=test_size, random_state=seed, stratify=y
    )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(idx_trval)), y[idx_trval]))
    # `seen` already holds every url_train slug, including the test rows.
    scr_urls, scr_texts, y_scr = to_examples(read_urls(scraped_path), seen)
    if len(scr_urls) == 0:
        raise RuntimeError("No scraped rows left after deduplication; scraped data is required.")
    return Splits(urls, texts, y, idx_trval, idx_test, folds, scr_urls, scr_texts, y_scr)


def save_split_files(s: Splits, out_dir: str) -> None:
    write_urls(os.path.join(out_dir, "trainval.csv"), [s.urls[i] for i in s.idx_trval])
    write_urls(os.path.join(out_dir, "test.csv"), [s.urls[i] for i in s.idx_test])
    write_urls(os.path.join(out_dir, "scraped_used.csv"), s.scr_urls)


def describe(s: Splits) -> str:
    return (f"url_train after slug dedup: {len(s.urls)} | trainval {len(s.idx_trval)} | "
            f"test {len(s.idx_test)} | scraped after dedup: {len(s.scr_urls)} "
            f"(FoxNews={int((s.y_scr == 0).sum())}, NBC={int((s.y_scr == 1).sum())})")
