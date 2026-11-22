import csv
import os
import re
from typing import List, Tuple
from urllib.parse import unquote, urlparse

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

def prepare_data(path: str) -> Tuple[List[str], List[str]]:
    """
    Return one model input and aligned label for every data row, preserving order.
    Blank and duplicate rows are retained; a header named "url" is not data.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    with open(path, "r") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].strip().lower() == "url":
        rows = rows[1:]

    urls: List[str] = []
    labels: List[str] = []
    for row in rows:
        value = row[0].strip() if row else ""
        parsed = urlparse(value)
        if parsed.netloc:
            link = _slug_to_path(parsed.path)
            if link == "/":
                link = parsed.netloc.lower()
            label = _label_from_host(parsed.netloc.lower())
        else:
            link = _slug_to_path(value) if value else ""
            label = ""
        urls.append(link)
        labels.append(label)

    return urls, labels
