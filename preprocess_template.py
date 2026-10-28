import os
import csv
import re
import torch
from urllib.parse import unquote, urlparse
from typing import Any, List, Tuple
import random
import requests
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET

def _slug_to_path(path: str) -> str:
    """
    Convert a URL path to a slug-like format.
    """
    segments = [s for s in path.split("/") if s]
    last = unquote(segments[-1]) if segments else ""
    last = re.sub(r"\.(html?|shtml|php)$", "", last, flags=re.IGNORECASE)
    words = [w for w in re.split(r"[-_]+", last) if w and not re.search(r"\d", w)]
    link = "/" + "-".join(words)
    return link

def _label_from_host(host: str) -> str:
    """
    Determine the label based on the host name.
    """
    if "foxnews" in host:
        return "FoxNews"
    elif "nbcnews" in host or "nbc.news" in host or "nbc" in host:
        return "NBC"
    return ""

def _clean(rows: List[str]) -> List[str]:
    """
    Clean the list of URL rows by filtering out duplicates, unwanted domains, and irrelevant paths.
    """
    cleaned_rows = []
    seen = set()
    NON_ARTICLE_PATHS = ["/video/", "/author/", "authors", "/tag/", "/category/", "/gallery/", "/live/", "/podcast/",
                         "/photo/", "/slideshow/", "/watch/", "/newsletters/"]

    for row in rows:
        url = row[0].strip()
        parsed = urlparse(url)

        if url in seen:
            continue

        if not any(domain in parsed.netloc for domain in ["foxnews.com", "nbcnews.com"]):
            continue

        if parsed.path in ["", "/"] or any(word in parsed.path for word in NON_ARTICLE_PATHS):
            continue

        seen.add(url)
        cleaned_rows.append([url])

    return cleaned_rows

def _get_urls(sitemap_url: str, domain: str) -> List[str]:
    """
    Fetch URLs from a sitemap XML.
    """
    urls = []
    response = requests.get(sitemap_url, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "xml")
    
    for loc in soup.find_all("loc"):
        url = loc.text.strip()
        if urlparse(url).netloc.endswith(domain):
            urls.append(url)
    return urls

def _fetch_url_rows() -> List[str]:
    """
    Fetch additional URLs from Fox News and NBC News feeds using sitemaps. 
    """
    rows = []

    sites = {
        "foxnews.com": "https://www.foxnews.com/robots.txt",
        "nbcnews.com": "https://www.nbcnews.com/robots.txt",
    }

    for domain, robots in sites.items():
        response = requests.get(robots, timeout=20)

        for line in response.text.splitlines():
            # Only start looking for the URL on the line that starts with "Sitemap:"
            if line.lower().startswith("sitemap:"):
                sitemap = line.split(":", 1)[1].strip()
                try:
                    rows.extend([[url] for url in _get_urls(sitemap, domain)])
                except (requests.RequestException, ET.ParseError) as e:
                    print(f"Skipping sitemap: {e}")
    return rows


def _process(path: str, urls: List[str], labels: List[str]) -> None:
    with open(path, "r") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].strip().lower() == "url":
        rows = rows[1:]
    # Fetch additional rows from the web feeds
    rows.extend(_fetch_url_rows())
    # Clean the list of URLS
    rows = _clean(rows)

    for row in rows:
        # Skip links from other websites. Links WITHOUT a website name are kept, 
        # with an empty label, so every row gets a prediction.
        if not row or not row[0].strip():
            continue
        parsed = urlparse(row[0].strip())
        label = _label_from_host(parsed.netloc.lower())
        if parsed.netloc and not label:
            continue
        link = _slug_to_path(parsed.path)
        if not link:
            link = parsed.netloc.lower()
        urls.append(link)
        labels.append(label)

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

if __name__ == "__main__":
    input_csv = "url_train.csv"
    X, y = prepare_data(input_csv)
    print(len(X))
    print(len(y))
