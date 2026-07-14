"""
recipe_crawler.py
==================
Discovers recipe URLs across configured sites and scrapes them into
recipes.csv, using recipe_scraper.py for the actual page parsing.

Discovery strategy per site:
    1. Primary: crawl the site's main-course category listing page (or
       vegetarian category listing when --vegetarian is set). This is a
       deliberate choice — a category page guarantees the source is
       actually main courses, which a sitemap can't do (sitemaps carry
       no category information).
    2. Fallback: only if the site has no category_urls configured at all,
       fall back to sitemap discovery. Mains-filtering then relies purely
       on the dessert/snack keyword check below, since there's no
       category guarantee from the source.

Filtering (configured in config.yaml):
    - Every candidate is checked against dessert_snack_keywords by URL
      slug before ever fetching the page — applied unconditionally,
      since mains-only is a firm rule regardless of vegetarian mode
      (vegetarian collections often include desserts too).
    - After fetching, ingredients are checked against non_vegetarian_keywords
      when --vegetarian is set.
    - Near-duplicate recipe names (e.g. "Spaghetti Bolognese" vs "Easy Spag
      Bol") are skipped via fuzzy name matching against everything already
      in recipes.csv plus this session's finds.

Stopping criteria:
    - Stops once recipes.csv reaches crawler.target_total_recipes rows.
    - Also gives up per-site after crawler.max_attempts_multiplier * target
      candidate pages have been tried, in case a site runs dry.

Usage:
    python recipe_crawler.py
    python recipe_crawler.py --target 200
    python recipe_crawler.py --vegetarian
    python recipe_crawler.py --site "BBC Good Food"

Dependencies:
    pip install selenium beautifulsoup4 webdriver-manager requests pyyaml
"""

import argparse
import csv
import difflib
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
import yaml
from bs4 import BeautifulSoup

import recipe_scraper as scraper


DEFAULT_CONFIG = Path("config.yaml")

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Existing recipe state (for dedup + stopping criteria)
# ---------------------------------------------------------------------------

def load_existing_recipes(csv_path: Path) -> tuple:
    """Return (set of source_urls, list of recipe names) already scraped."""
    if not csv_path.exists():
        return set(), []
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    urls = {r["source_url"] for r in rows if r.get("source_url")}
    names = [r["name"] for r in rows if r.get("name")]
    return urls, names


def normalise_name(name: str) -> str:
    """Lowercase, strip punctuation/common filler words for comparison."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", "", name)
    filler = {"easy", "simple", "quick", "best", "classic", "the", "a", "an", "recipe"}
    words = [w for w in name.split() if w not in filler]
    return " ".join(words)


def is_near_duplicate(name: str, existing_names: list, threshold: float) -> bool:
    """Check if name is a fuzzy match for anything already collected."""
    norm = normalise_name(name)
    if not norm:
        return False
    for existing in existing_names:
        existing_norm = normalise_name(existing)
        if not existing_norm:
            continue
        ratio = difflib.SequenceMatcher(None, norm, existing_norm).ratio()
        if ratio >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# URL discovery — sitemap
# ---------------------------------------------------------------------------

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def fetch_sitemap_urls(
    sitemap_url: str, recipe_url_contains: str, depth: int = 0,
    non_recipe_segments: list = None,
) -> list:
    """
    Recursively fetch a sitemap (or sitemap index) and return all URLs
    matching recipe_url_contains. Plain requests — sitemaps are static XML,
    no browser rendering needed.
    """
    non_recipe_segments = non_recipe_segments or []
    if depth > 2:  # safety limit on nested sitemap indexes
        return []

    try:
        resp = requests.get(sitemap_url, headers=REQUEST_HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as e:
        print(f"  ⚠ Could not read sitemap {sitemap_url}: {e}")
        return []

    urls = []

    # Sitemap index — contains links to other sitemaps
    sub_sitemaps = root.findall("sm:sitemap/sm:loc", SITEMAP_NS)
    if sub_sitemaps:
        for loc in sub_sitemaps:
            sub_url = loc.text.strip()
            # Only descend into sitemaps that look recipe-related, to save time
            if any(hint in sub_url.lower() for hint in ("recipe", "post", "page")):
                urls.extend(fetch_sitemap_urls(sub_url, recipe_url_contains, depth + 1, non_recipe_segments))
        return urls

    # Regular sitemap — contains actual page URLs
    for loc in root.findall("sm:url/sm:loc", SITEMAP_NS):
        page_url = loc.text.strip()
        if recipe_url_contains not in page_url:
            continue
        if any(seg in page_url for seg in non_recipe_segments):
            continue
        urls.append(page_url)

    return urls


# ---------------------------------------------------------------------------
# URL discovery — category/listing pages (fallback, needs JS rendering)
# ---------------------------------------------------------------------------

def fetch_category_urls(
    driver, category_url: str, base_url: str, recipe_url_contains: str,
    max_pages: int = 40, non_recipe_segments: list = None,
) -> list:
    """
    Render a category/collection page and pull out recipe links, paginating
    with ?page=N until a page yields no new links or max_pages is hit.
    A single listing page rarely has enough recipes to hit a few-hundred
    target, so pagination is required here.

    non_recipe_segments excludes links that technically contain
    recipe_url_contains but are actually other collection/list/article pages
    (e.g. a "Related collections" block linking to another category page).
    Without this, the crawler can wander into scraping collection pages as
    if they were individual recipes.
    """
    non_recipe_segments = non_recipe_segments or []
    urls = []
    seen = set()

    for page_num in range(1, max_pages + 1):
        page_url = category_url if page_num == 1 else f"{category_url}?page={page_num}"
        try:
            html = scraper.fetch_html(driver, page_url)
        except Exception as e:
            print(f"    ✗ Failed to load listing page {page_num}: {e}")
            break

        soup = BeautifulSoup(html, "html.parser")
        new_this_page = 0
        for a in soup.find_all("a", href=True):
            href = urljoin(base_url, a["href"]).split("?")[0].split("#")[0]
            if recipe_url_contains not in href:
                continue
            if any(seg in href for seg in non_recipe_segments):
                continue
            if href.rstrip("/") == category_url.rstrip("/"):
                continue
            if href in seen:
                continue
            seen.add(href)
            urls.append(href)
            new_this_page += 1

        print(f"    Page {page_num}: {new_this_page} new link(s) ({page_url})")

        if new_this_page == 0:
            break
        time.sleep(1)  # light delay between listing pages

    return urls


# ---------------------------------------------------------------------------
# Candidate filtering
# ---------------------------------------------------------------------------

def looks_like_dessert_or_snack(url: str, dessert_keywords: list) -> bool:
    """Cheap pre-fetch check on the URL slug to skip obvious non-mains."""
    lower = url.lower()
    return any(kw in lower for kw in dessert_keywords)


def contains_meat(ingredients: str, non_veg_keywords: list) -> bool:
    lower = ingredients.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in non_veg_keywords)


# ---------------------------------------------------------------------------
# Per-site crawl
# ---------------------------------------------------------------------------

def crawl_site(
    site: dict,
    config: dict,
    driver,
    csv_path: Path,
    existing_urls: set,
    existing_names: list,
    remaining_slots: int,
    vegetarian_only: bool,
) -> int:
    """
    Crawl a single configured site until remaining_slots is filled or
    candidates run out. Returns the number of recipes actually added.
    """
    crawler_cfg = config.get("crawler", {})
    delay = crawler_cfg.get("request_delay_seconds", 2)
    threshold = crawler_cfg.get("name_similarity_threshold", 0.85)
    max_attempts = int(remaining_slots * crawler_cfg.get("max_attempts_multiplier", 4))
    dessert_keywords = config.get("dessert_snack_keywords", [])
    non_veg_keywords = config.get("non_vegetarian_keywords", [])

    site_name = site["name"]
    base_url = site["base_url"]
    recipe_url_contains = site.get("recipe_url_contains", "/recipe")
    category_urls = site.get("category_urls", {})
    non_recipe_segments = site.get("non_recipe_url_segments", [])

    print(f"\n=== {site_name} ===")

    # --- Discover candidate URLs ---
    # Category listing pages are tried first — this is what guarantees a
    # mains-only source. Sitemap is only used as a fallback for a site with
    # no category_urls configured (see module docstring for reasoning).
    candidates = []

    target_category = None
    if vegetarian_only and "vegetarian" in category_urls:
        target_category = category_urls["vegetarian"]
    elif "main" in category_urls:
        target_category = category_urls["main"]

    if target_category:
        print(f"Discovering from category listing (mains-only source): {target_category}")
        max_pages = crawler_cfg.get("max_category_pages", 40)
        candidates = fetch_category_urls(
            driver, target_category, base_url, recipe_url_contains,
            max_pages=max_pages, non_recipe_segments=non_recipe_segments,
        )
        print(f"  Found {len(candidates)} candidate URLs.")
    else:
        sitemap_url = site.get("sitemap_url")
        if sitemap_url:
            print("No suitable category_urls configured — falling back to sitemap.")
            print("  (Mains-only filtering will rely on the dessert/snack keyword check.)")
            candidates = fetch_sitemap_urls(
                sitemap_url, recipe_url_contains, non_recipe_segments=non_recipe_segments
            )
            print(f"  Found {len(candidates)} candidate URLs via sitemap.")
        else:
            print("✗ Site has neither a usable category_urls entry nor sitemap_url — skipping.")
            return 0

    # Dedup candidate list itself, and drop anything already scraped
    seen = set()
    unique_candidates = []
    for url in candidates:
        if url in seen or url in existing_urls:
            continue
        seen.add(url)
        unique_candidates.append(url)

    print(f"{len(unique_candidates)} new candidate URLs to evaluate.")

    added = 0
    attempts = 0

    for url in unique_candidates:
        if added >= remaining_slots:
            print(f"  Reached target for this run ({remaining_slots} slots) — stopping site.")
            break
        if attempts >= max_attempts:
            print(f"  Hit max attempts ({max_attempts}) for this site — moving on.")
            break

        attempts += 1

        # Cheap pre-fetch filter: obvious dessert/snack by URL slug.
        # Applied regardless of vegetarian mode — mains-only is a firm rule,
        # and vegetarian collections often include desserts too.
        if looks_like_dessert_or_snack(url, dessert_keywords):
            print(f"  ⏭ Skipping (looks like dessert/snack): {url}")
            continue

        try:
            recipe = scraper.scrape_recipe(url, driver=driver, csv_path=csv_path)
        except Exception as e:
            print(f"  ✗ Error scraping {url}: {e}")
            time.sleep(delay)
            continue

        if recipe is None:
            time.sleep(delay)
            continue

        name = recipe.get("name", "")
        ingredients = recipe.get("ingredients", "")

        # Vegetarian safety net (post-fetch, ingredient-based)
        if vegetarian_only and contains_meat(ingredients, non_veg_keywords):
            print(f"  ⏭ Discarding (contains meat/fish): {name}")
            _remove_last_csv_row(csv_path)
            time.sleep(delay)
            continue

        # Near-duplicate check
        if is_near_duplicate(name, existing_names, threshold):
            print(f"  ⏭ Discarding (near-duplicate of existing recipe): {name}")
            _remove_last_csv_row(csv_path)
            time.sleep(delay)
            continue

        existing_urls.add(url)
        existing_names.append(name)
        added += 1
        print(f"  ✓ Kept ({added}/{remaining_slots} this run): {name}")

        time.sleep(delay)

    print(f"{site_name}: added {added} recipe(s) this run ({attempts} attempted).")
    return added


def _remove_last_csv_row(csv_path: Path) -> None:
    """
    Remove the most recently appended row from the CSV.
    Used when a recipe is scraped but then rejected by a post-fetch filter
    (vegetarian check, duplicate check) — keeps recipes.csv clean.
    """
    if not csv_path.exists():
        return
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) <= 1:  # header only or empty
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows[:-1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl recipe sites into recipes.csv")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target", type=int, help="Override crawler.target_total_recipes")
    parser.add_argument("--vegetarian", action="store_true", help="Only keep vegetarian recipes")
    parser.add_argument("--site", help="Only crawl the named site (matches config 'name')")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"✗ Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    csv_path = Path(config.get("scraped_csv", "recipes.csv"))
    crawler_cfg = config.get("crawler", {})
    target = args.target or crawler_cfg.get("target_total_recipes", 100)
    sites = config.get("sites", [])

    if args.site:
        sites = [s for s in sites if s["name"] == args.site]
        if not sites:
            print(f"✗ No configured site named '{args.site}'")
            sys.exit(1)

    existing_urls, existing_names = load_existing_recipes(csv_path)
    current_total = len(existing_urls)

    print(f"Current recipe count: {current_total}")
    print(f"Target: {target}")
    print(f"Vegetarian only: {args.vegetarian}")

    if current_total >= target:
        print("Target already reached — nothing to do. Increase --target to fetch more.")
        return

    driver = scraper.create_driver()
    try:
        for site in sites:
            remaining = target - len(existing_urls)
            if remaining <= 0:
                print("\nTarget reached — stopping.")
                break
            crawl_site(
                site,
                config,
                driver,
                csv_path,
                existing_urls,
                existing_names,
                remaining_slots=remaining,
                vegetarian_only=args.vegetarian,
            )
    finally:
        driver.quit()

    final_total = len(load_existing_recipes(csv_path)[0])
    print(f"\nDone. recipes.csv now has {final_total} recipe(s).")


if __name__ == "__main__":
    main()