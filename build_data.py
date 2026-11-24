import os
import csv
import html
import re
from urllib.parse import unquote, urlparse
from typing import List, Tuple
import random
import requests
from xml.etree import ElementTree as ET
from collections import Counter

VALIDATION = False
NON_ARTICLE_PATHS = ["/author/", "/authors/", "/tag/", "/category/", "/newsletters/"]
NON_ARTICLE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js", ".json", ".xml", 
                    ".pdf", ".ico", ".mp4", ".txt")

def _slug_to_path(path: str) -> str:
    """
    Convert the last URL path segment into normalized, lowercase headline tokens.
    """
    # Remove leading/trailing slashes and split the path into segments
    segments = [s for s in path.split("/") if s]
    last = unquote(segments[-1]) if segments else ""

    # Decode URL text before removing markup and normalizing separators.
    last = html.unescape(last)
    last = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", last, flags=re.IGNORECASE | re.DOTALL)
    last = re.sub(r"<[^>]*>", " ", last)
    last = re.sub(r"\.(html?|shtml|php)$", "", last, flags=re.IGNORECASE)

    # Keep words only; dropping digit-bearing terms removes common article IDs.
    words = re.findall(r"[^\W_]+", last.casefold(), flags=re.UNICODE)
    words = [word for word in words if not re.search(r"\d", word)]

    link = "/" + "-".join(words)
    return link

def _label_from_host(host: str) -> str:
    """
    Determine the label based on the host name.

    Parameters:
    host (str): The host name extracted from the URL.
    """
    if "foxnews" in host:
        return "FoxNews"
    elif "nbcnews" in host or "nbc.news" in host or "nbc" in host:
        return "NBC"
    return ""

def _clean(rows: List[List[str]]) -> List[List[str]]:
    """
    Filter scraped URL rows, dropping duplicates, unwanted domains, and irrelevant paths.

    Parameters:
    rows (List[List[str]]): URL rows to be filtered.
    """
    cleaned_rows = []
    seen_urls = set()
    seen_headlines = set()

    for row in rows:
        if not row or not row[0].strip():
            continue
        url = row[0].strip()
        parsed = urlparse(url)

        # Filter out duplicates
        if url in seen_urls:
            continue

        # Filter out unwanted domains
        if not any(domain in parsed.netloc for domain in ["foxnews.com", "nbcnews.com"]):
            continue

        # Filter out irrelevant paths
        path = parsed.path.lower()
        if (path in ["", "/"] or 
            path.endswith(NON_ARTICLE_EXTS) or 
            any(word in path for word in NON_ARTICLE_PATHS)):
            continue

        # Filter out URLs with query parameters or fragments
        segments = [segment for segment in parsed.path.split("/") if segment]
        headline_segment = unquote(segments[-1]) if segments else ""
        if "&" in headline_segment or "=" in headline_segment:
            continue

        # Different URL variants can share the same headline; retain one scraped row.
        headline = _slug_to_path(parsed.path)
        if headline == "/" or headline in seen_headlines:
            continue

        seen_urls.add(url)
        seen_headlines.add(headline)
        cleaned_rows.append([url])

    return cleaned_rows

def _get_urls(sitemap_url: str, domain: str) -> List[str]:
    """
    Fetch URLs from a sitemap XML.

    Parameters:
    sitemap_url (str): The URL of the sitemap XML.
    domain (str): The domain to filter URLs by.
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

def _fetch_url_rows() -> List[List[str]]:
    """
    Fetch Fox News and NBC News URLs from their sitemaps and the Wayback Machine.

    Returns:
    List of one-column URL rows, matching the format used by url_train.csv.
    """
    rows = []
    sites = {"foxnews.com": "https://www.foxnews.com/robots.txt",
             "nbcnews.com": "https://www.nbcnews.com/robots.txt"}

    for domain, robots in sites.items():
        try:
            response = requests.get(robots, timeout=20)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Skipping {domain} robots.txt: {e}")
            continue

        sitemaps = []
        for line in response.text.splitlines():
            if line.lower().startswith("sitemap:"):
                sitemaps.append(line.split(":", 1)[1].strip())

        if domain == "foxnews.com":
            sitemaps.extend(["https://www.foxnews.com/sitemap.xml?type=articles",
                             "https://www.foxnews.com/sitemap.xml?type=fastchanging"])

        for sitemap in dict.fromkeys(sitemaps):
            try:
                rows.extend([[url] for url in _get_urls(sitemap, domain)])
            except (requests.RequestException, ET.ParseError) as e:
                print(f"Skipping sitemap {sitemap}: {e}")

    return rows

def _process(path: str, urls: List[str], labels: List[str], fetch_online: bool = True,) -> None:
    """
    Process the CSV file and fetch additional URLs from web feeds.

    Parameters: 
    path (str): Path to the CSV file containing URLs.
    urls (List[str]): List to store the processed URLs.
    labels (List[str]): List to store the corresponding labels for the URLs.
    fetch_online (bool): Include eligible scraped URLs alongside source CSV rows.
    """
    with open(path, "r", newline="") as f:
        source_rows = list(csv.reader(f))
    if source_rows and source_rows[0] and source_rows[0][0].strip().lower() == "url":
        source_rows = source_rows[1:]

    # Keep supplied rows intact; cleaning and deduplication apply only to new scrape results
    source_urls = {row[0].strip() for row in source_rows if row and row[0].strip()}
    source_headlines = set()
    for row in source_rows:
        if not row or not row[0].strip():
            continue
        parsed = urlparse(row[0].strip())
        label = _label_from_host(parsed.netloc.lower())
        headline = _slug_to_path(parsed.path)
        if headline == "/":
            headline = parsed.netloc.lower()
        source_headlines.add(headline)

    scraped_rows = _clean(_fetch_url_rows()) if fetch_online else []
    scraped_rows = [row for row in scraped_rows if row[0].strip() not in source_urls
                           and _slug_to_path(urlparse(row[0].strip()).path) not in source_headlines]
    
    # Validate scraped URLs
    if VALIDATION:
        for u in random.sample(scraped_rows, min(100, len(scraped_rows))):
            print(u)

    rows = source_rows + scraped_rows

    # Build dataset (feature: URL path; label: FoxNews or NBC)
    for row in rows:
        if not row or not row[0].strip():
            continue
        parsed = urlparse(row[0].strip())
        label = _label_from_host(parsed.netloc.lower())
        if parsed.netloc and not label:
            continue
        link = _slug_to_path(parsed.path)
        if link == "/":
            # Preserve a usable feature for valid URLs that have no headline slug
            link = parsed.netloc.lower()
        urls.append(link)
        labels.append(label)

def build_training_data(path: str, fetch_online: bool = True) -> Tuple[List[str], List[str]]:
    """
    Build labeled data while preserving rows from the supplied CSV.
    Newly scraped URLs are filtered and deduplicated before being appended.

    Parameters:
    path (str): Path to the CSV file containing URLs.
    fetch_only (bool): Fetch and include scraped URLs when True; otherwise use only
        rows from the supplied CSV.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    urls: List[str] = []
    labels: List[str] = []
    _process(path, urls, labels, fetch_online=fetch_online)
    return urls, labels

def tmp(y):
    counts = Counter(y)
    print(counts)
    print(set(y))

if __name__ == "__main__":
    input_csv = "url_train.csv"
    X, y = build_training_data(input_csv)
    print(len(X))
    print(len(y))
    tmp(y)
