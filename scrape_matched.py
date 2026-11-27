"""Scrape Fox News / NBC News URLs from sitemap shards that match url_train.csv.

Both sites split their sitemaps into shards (Fox: pages of ~5000 articles in
reverse date order; NBC: one file per month and content type). url_train.csv
was clearly sampled from a few specific shards, and scraping other periods or
sections (e.g. Fox "outkick") shifts the training distribution and hurts
accuracy. This script therefore:

  1. downloads every candidate shard (cached under data/raw_sitemaps/, with a
     pause between requests),
  2. measures how many url_train.csv URLs each shard contains,
  3. keeps only shards with at least --min-overlap training URLs,
  4. writes the URLs of the kept shards (minus url_train URLs) to --output.

Only url_train.csv is used to choose shards; no validation/test labels are
involved. Slug-level dedup against url_train (including the local test split)
happens later in the training scripts.

Usage:
    python scrape_matched.py                 # writes data/scraped_matched.csv
"""

import argparse
import csv
import hashlib
import os
import re
import time
from typing import Dict, List
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (academic course project crawler)"}
FOX_INDEX = "https://www.foxnews.com/sitemap.xml"
NBC_INDEX = "https://www.nbcnews.com/sitemap/nbcnews/sitemap-index"
NBC_EXTRA = [
    "https://www.nbcnews.com/sitemap/nbcnews/sitemap-select.xml",
    "https://www.nbcnews.com/sitemap/nbcnews/sitemap-curations",
    "https://www.nbcnews.com/sitemap/nbcnews/sitemap-news",
]


def _fetch(url: str, cache_dir: str, delay: float) -> bytes:
    """GET with an on-disk cache so reruns never hit the sites again."""
    path = os.path.join(cache_dir, hashlib.md5(url.encode()).hexdigest() + ".xml")
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()
    time.sleep(delay)
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)
    return response.content


ARTICLE_HOSTS = {"www.foxnews.com", "www.nbcnews.com", "foxnews.com", "nbcnews.com"}


def _locs(content: bytes) -> List[str]:
    """Page URLs of a sitemap: only the <loc> directly under <url>/<sitemap>.

    Nested tags such as <image:loc> point at image CDNs (a57.foxnews.com,
    media-cldnry.s-nbcnews.com) and must not become training rows.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    out = []
    for entry in root:
        if entry.tag.rsplit("}", 1)[-1] not in ("url", "sitemap"):
            continue
        for child in entry:
            if child.tag.rsplit("}", 1)[-1] == "loc":
                out.append((child.text or "").strip())
    return out


def _candidate_shards(cache_dir: str, delay: float, min_year: int) -> List[str]:
    shards = []
    fox = _locs(_fetch(FOX_INDEX, cache_dir, delay))
    shards += [u for u in fox if "type=articles" in u or "type=liveblogs" in u]

    nbc = _locs(_fetch(NBC_INDEX, cache_dir, delay))
    for u in nbc:
        m = re.search(r"sitemap-(\d{4})-\d{2}-(article|video|slideshow)\.xml$", u)
        if m and int(m.group(1)) >= min_year:
            shards.append(u)
    shards += NBC_EXTRA
    return list(dict.fromkeys(shards))


def _read_urls(path: str) -> List[str]:
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].strip().lower() == "url":
        rows = rows[1:]
    return [row[0].strip() for row in rows if row and row[0].strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape sitemap shards that match url_train.csv")
    parser.add_argument("--csv", default="url_train.csv")
    parser.add_argument("--output", default="data/scraped_matched.csv")
    parser.add_argument("--cache-dir", default="data/raw_sitemaps")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between live requests")
    parser.add_argument("--min-year", type=int, default=2020, help="Oldest NBC monthly shard to consider")
    parser.add_argument("--min-overlap", type=int, default=10,
                        help="Keep a shard only if it contains this many url_train URLs")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    train = set(_read_urls(args.csv))
    shards = _candidate_shards(args.cache_dir, args.delay, args.min_year)
    print(f"candidate shards: {len(shards)}")

    kept: Dict[str, List[str]] = {}
    report = []
    for i, shard in enumerate(shards, 1):
        try:
            urls = list(dict.fromkeys(_locs(_fetch(shard, args.cache_dir, args.delay))))
        except requests.RequestException as e:
            print(f"  skip {shard}: {e}")
            continue
        overlap = sum(u in train for u in urls)
        report.append((shard, len(urls), overlap))
        if overlap >= args.min_overlap:
            kept[shard] = [u for u in urls if u not in train]
        if i % 25 == 0:
            print(f"  {i}/{len(shards)} shards processed")

    report_path = os.path.splitext(args.output)[0] + "_shards.csv"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shard", "n_urls", "url_train_overlap", "kept"])
        for shard, n, ov in sorted(report, key=lambda r: -r[2]):
            writer.writerow([shard, n, ov, shard in kept])

    out = list(dict.fromkeys(u for urls in kept.values() for u in urls
                             if urlparse(u).netloc.lower() in ARTICLE_HOSTS))
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["url"])
        writer.writerows([[u] for u in out])

    covered = sum(ov for s, _, ov in report if s in kept)
    print(f"kept {len(kept)} shards covering {covered} url_train URLs "
          f"(of {len(train)}); wrote {len(out)} new URLs to {args.output}")
    print(f"per-shard report: {report_path}")


if __name__ == "__main__":
    main()
