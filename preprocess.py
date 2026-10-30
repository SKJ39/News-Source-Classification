import os
import csv
import re
import torch
from urllib.parse import unquote, urlparse
from typing import Any, List, Tuple
import random
import requests
from xml.etree import ElementTree as ET
from collections import Counter

VALIDATION = True
NON_ARTICLE_PATHS = ["/video/", "/videos/", "/author/", "/authors", "/tag/", "/category/",
                         "/gallery/", "/live/", "/podcast/", "/photo/", "/slideshow/", "/watch/",
                         "/newsletters/", "/person/", "/shows/", "/search", "/sitemap", "/about",
                         "/contact", "/privacy", "/terms", "/rss", "/feed", "/wp-content/",
                         "/api/", "/static/", "/assets/"]
NON_ARTICLE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".json", ".xml", 
                    ".pdf", ".ico", ".mp4", ".txt")

def _slug_to_path(path: str) -> str:
    """
    Convert a URL path to a slug-like format.
    The website name, section, and any word containing a digit (such as article IDs) are removed, 
    leaving only the headline portion of each URL.
    The leading (like https) and trailing extensions (like .html, .shtml, .php) are also removed. 
    The resulting slug is returned in lowercase.
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

    for row in rows:
        url = row[0].strip()
        parsed = urlparse(url)

        if url in seen:
            continue

        if not any(domain in parsed.netloc for domain in ["foxnews.com", "nbcnews.com"]):
            continue

        path = parsed.path.lower()
        if (path in ["", "/"] or 
            path.endswith(NON_ARTICLE_EXTS) or 
            any(word in path for word in NON_ARTICLE_PATHS)):
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
    root = ET.fromstring(response.content)

    # Iterate through the XML elements and extract URLs that match the specified domain
    for loc in root.iter():
        if loc.tag.rsplit("}", 1)[-1] != "loc":
            continue
        url = (loc.text or "").strip()
        if urlparse(url).netloc.endswith(domain):
            urls.append(url)
    return urls

def _fetch_url_rows() -> List[str]:
    """
    Fetch additional URLs from Fox News and NBC News feeds using sitemaps. 
    """
    rows = []
    sites = {"foxnews.com": "https://www.foxnews.com/robots.txt",
             "nbcnews.com": "https://www.nbcnews.com/robots.txt"}

    # Fetch URLs of both Foxnews and NBC from robots.txt files and their sitemaps
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

    # Fetch additional URLs from the Wayback Machine for Foxnews articles
    max_rows = 10_000
    r = requests.get("http://web.archive.org/cdx/search/cdx",
                    params={"url": "foxnews.com/*", "output": "txt", "fl": "original", 
                             "collapse": "urlkey", "filter": "statuscode:200", "limit": max_rows},
                    timeout=30)
    rows.extend([[u] for u in r.text.splitlines()])
    return rows

def drop_nbc(urls, labels, n=1000):
    """
    Drop a random sample of NBC rows to reduce class imbalance.
    """
    nbc = [idx for idx, domain in enumerate(labels) if domain == "NBC"]
    nbc_indices = set(random.sample(nbc, min(n, len(nbc))))
    urls = [url for idx, url in enumerate(urls) if idx not in nbc_indices]
    labels = [domain for idx, domain in enumerate(labels) if idx not in nbc_indices]
    return urls, labels

def _process(path: str, urls: List[str], labels: List[str]) -> None:
    """
    Process the CSV file and fetch additional URLs from web feeds.
    """
    with open(path, "r") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].strip().lower() == "url":
        rows = rows[1:]
    # Fetch additional rows from the web feeds
    rows.extend(_fetch_url_rows())
    # Clean the list of URLS
    rows = _clean(rows)

    # Data validation
    if VALIDATION:
        for u in random.sample(rows, min(100, len(rows))):
            print(u)

    # Creating training dataset
    for row in rows:
        # Skip links from other websites. Links WITHOUT a website name are kept, 
        # with an empty label, so every row gets a prediction.
        if not row or not row[0].strip():
            continue
        parsed = urlparse(row[0].strip())
        label = _label_from_host(parsed.netloc.lower())         # netloc is the domain name, e.g., "www.foxnews.com"
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

    # For evaluation, only use the provided CSV file
    # For training, fetch additional data from the web feeds
    with open(path, "r") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].strip().lower() == "url":
        rows = rows[1:]
    if rows and all(len(row) == 1 and row[0].strip().startswith("/") for row in rows):
        urls = [row[0].strip() for row in rows]
        return urls, [""] * len(urls)

    urls: List[str] = []
    labels: List[str] = []
    _process(path, urls, labels)

    # To ease class imbalance, drop some NBC rows
    urls, labels = drop_nbc(urls, labels, n=1000)
    return urls, labels

def tmp(y):
    counts = Counter(y)
    print(counts)
    print(set(y))

if __name__ == "__main__":
    input_csv = "url_train.csv"
    X, y = prepare_data(input_csv)
    if VALIDATION:
        print(len(X))
        print(len(y))
        tmp(y)
