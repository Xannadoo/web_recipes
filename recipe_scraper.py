"""
recipe_scraper.py
=================
Scrape a recipe from a URL and append it to recipes.csv.

Usage:
    python recipe_scraper.py <url>
    python recipe_scraper.py https://www.example.com/pasta-bake

Dependencies:
    pip install selenium beautifulsoup4 webdriver-manager

Selenium setup (handled automatically via webdriver-manager):
    - Requires Google Chrome to be installed.
    - webdriver-manager downloads the matching chromedriver automatically.
"""

import sys
import csv
import json
import re
import time
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

CSV_FILE = Path("recipes.csv")
CSV_HEADERS = [
    "name", "servings", "prep_time", "cook_time", "total_time",
    "ingredients", "method", "source_url", "scraped_date"
]


# ---------------------------------------------------------------------------
# Browser / page fetching
# ---------------------------------------------------------------------------

def create_driver():
    """
    Create a headless Chrome driver. Caller is responsible for calling
    driver.quit() when done. Reuse one driver across many page fetches
    (e.g. in the crawler) to avoid the overhead of restarting the browser
    for every URL.
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def fetch_html(driver, url: str, wait_seconds: float = 3) -> str:
    """Navigate an existing driver to url and return the rendered HTML."""
    driver.get(url)
    time.sleep(wait_seconds)  # Allow JS to render; increase for slow sites
    return driver.page_source


def get_page_html(url: str) -> str:
    """
    Fetch fully-rendered HTML using a fresh headless Chrome browser.
    Convenience wrapper for one-off single-URL use. For scraping many
    URLs in a loop, use create_driver() + fetch_html() instead to reuse
    the browser session.
    """
    driver = create_driver()
    try:
        return fetch_html(driver, url)
    finally:
        driver.quit()


# ---------------------------------------------------------------------------
# JSON-LD schema extraction (primary strategy)
# ---------------------------------------------------------------------------

def parse_duration(iso: str) -> str:
    """Convert ISO 8601 duration (PT1H30M) to a readable string (1h 30m)."""
    if not iso:
        return ""
    match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?", iso)
    if not match:
        return iso
    hours, minutes = match.group(1), match.group(2)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    return " ".join(parts) if parts else iso


def clean_html_text(raw) -> str:
    """
    Strip any embedded HTML tags/entities from a text value and collapse
    whitespace/newlines to single spaces. Some sites (e.g. HelloFresh) embed
    raw HTML like '<p>a) Preheat...</p>' inside JSON-LD text fields instead
    of plain text — without this, tags and multi-line formatting leak
    straight into the CSV.
    """
    if not raw:
        return ""
    text = BeautifulSoup(str(raw), "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def extract_text(value) -> str:
    """Safely pull a string from a schema value that may be str or dict."""
    if isinstance(value, str):
        return clean_html_text(value)
    if isinstance(value, dict):
        return clean_html_text(value.get("text", ""))
    return ""


def try_json_ld(soup: BeautifulSoup) -> dict | None:
    """
    Look for a <script type="application/ld+json"> block containing a Recipe.
    Returns a normalised dict or None if not found.
    """
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue

        # Handle both a single object and a @graph array
        candidates = []
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            candidates = data.get("@graph", [data])

        for obj in candidates:
            recipe_type = obj.get("@type", "")
            # @type can be a string or a list
            types = recipe_type if isinstance(recipe_type, list) else [recipe_type]
            if "Recipe" not in types:
                continue

            # Ingredients: list of strings or HowToIngredient objects
            raw_ingredients = obj.get("recipeIngredient", [])
            ingredients = " | ".join(
                extract_text(i) for i in raw_ingredients if extract_text(i)
            )

            # Method: recipeInstructions can be a string, list of strings,
            # or list of HowToStep objects. Some sites embed raw HTML
            # inside the text — clean_html_text strips tags/entities.
            raw_steps = obj.get("recipeInstructions", [])
            if isinstance(raw_steps, str):
                method = clean_html_text(raw_steps)
            else:
                steps = []
                for step in raw_steps:
                    if isinstance(step, str):
                        text = clean_html_text(step)
                    elif isinstance(step, dict):
                        text = clean_html_text(step.get("text", "") or step.get("name", ""))
                    else:
                        text = ""
                    if text:
                        steps.append(text)
                method = " | ".join(steps)

            # Servings
            yield_val = obj.get("recipeYield", "")
            if isinstance(yield_val, list):
                yield_val = yield_val[0] if yield_val else ""
            servings = str(yield_val).strip()

            return {
                "name": extract_text(obj.get("name", "")),
                "servings": servings,
                "prep_time": parse_duration(obj.get("prepTime", "")),
                "cook_time": parse_duration(obj.get("cookTime", "")),
                # Some sites (e.g. HelloFresh) only give a combined total
                # time rather than separate prep/cook — kept as its own
                # field so ease scoring can use it as a fallback signal.
                "total_time": parse_duration(obj.get("totalTime", "")),
                "ingredients": ingredients,
                "method": method,
            }

    return None


# ---------------------------------------------------------------------------
# HTML heuristic extraction (fallback strategy)
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def try_html_heuristics(soup: BeautifulSoup) -> dict:
    """
    Best-effort extraction by looking for common CSS class patterns.
    Returns whatever it can find; fields may be empty strings.
    """

    def find_by_patterns(patterns: list[str]) -> list[str]:
        results = []
        for pattern in patterns:
            tags = soup.find_all(class_=re.compile(pattern, re.I))
            for tag in tags:
                text = clean_text(tag.get_text(" ", strip=True))
                if text:
                    results.append(text)
            if results:
                break
        return results

    # Recipe name — try heading tags near the top first
    name = ""
    for tag in soup.find_all(["h1", "h2"]):
        text = clean_text(tag.get_text())
        if text:
            name = text
            break

    # Ingredients
    ingredient_texts = find_by_patterns([
        r"ingredient", r"ingred"
    ])
    # If a single block was returned, split it into items
    if len(ingredient_texts) == 1 and "\n" in ingredient_texts[0]:
        ingredient_texts = [l.strip() for l in ingredient_texts[0].splitlines() if l.strip()]
    ingredients = " | ".join(ingredient_texts)

    # Method / instructions
    method_texts = find_by_patterns([
        r"instruction", r"method", r"direction", r"step"
    ])
    method = " | ".join(method_texts)

    # Servings
    servings = ""
    servings_tag = soup.find(class_=re.compile(r"yield|serving|portion", re.I))
    if servings_tag:
        servings = clean_text(servings_tag.get_text())

    # Times
    prep_time, cook_time, total_time = "", "", ""
    for tag in soup.find_all(class_=re.compile(r"prep.?time|preptime", re.I)):
        prep_time = clean_text(tag.get_text())
        break
    for tag in soup.find_all(class_=re.compile(r"cook.?time|cooktime", re.I)):
        cook_time = clean_text(tag.get_text())
        break
    for tag in soup.find_all(class_=re.compile(r"total.?time|totaltime", re.I)):
        total_time = clean_text(tag.get_text())
        break

    return {
        "name": name,
        "servings": servings,
        "prep_time": prep_time,
        "cook_time": cook_time,
        "total_time": total_time,
        "ingredients": ingredients,
        "method": method,
    }


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def append_to_csv(recipe: dict, url: str, csv_path: Path = CSV_FILE) -> None:
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            **recipe,
            "source_url": url,
            "scraped_date": date.today().isoformat(),
        })
    print(f"  ✓ Saved to {csv_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_recipe(html: str) -> dict | None:
    """
    Parse rendered HTML and return a recipe dict (JSON-LD first, HTML
    heuristics fallback), or None if nothing usable was found.
    Used by both the single-URL CLI and the crawler.
    """
    soup = BeautifulSoup(html, "html.parser")

    recipe = try_json_ld(soup)
    used_fallback = recipe is None
    if used_fallback:
        recipe = try_html_heuristics(soup)

    if not recipe.get("name", "").strip() or not recipe.get("ingredients", "").strip():
        return None

    # Sanity check for the fallback path only: JSON-LD is structured data
    # from the site itself, so it's trusted. HTML heuristics can misfire on
    # pages that aren't actually individual recipes — e.g. a collection/
    # category page's tag-cloud of filter keywords, where each character
    # ends up in its own <span> and gets joined into a stream of single
    # characters like "f | e | t | a | , | c | o | u | s". If most
    # "ingredients" are one or two characters long, this isn't a real
    # ingredient list — reject it rather than saving garbage.
    if used_fallback and recipe.get("ingredients"):
        items = [i for i in recipe["ingredients"].split(" | ") if i]
        if items:
            short_items = [i for i in items if len(i) <= 2]
            if len(short_items) / len(items) > 0.5:
                return None

    recipe["_used_fallback"] = used_fallback
    return recipe


def scrape_recipe(url: str, driver=None, csv_path: Path = CSV_FILE) -> dict | None:
    """
    Scrape a single recipe URL and append it to csv_path.

    Args:
        url: Recipe page URL.
        driver: An existing Selenium driver to reuse (e.g. from the crawler).
                If None, a fresh one is created and closed for this call.
        csv_path: Where to append the result.

    Returns the recipe dict on success, None if extraction failed.
    """
    print(f"\nFetching: {url}")

    owns_driver = driver is None
    if owns_driver:
        driver = create_driver()

    try:
        html = fetch_html(driver, url)
    finally:
        if owns_driver:
            driver.quit()

    print("  Trying JSON-LD schema...")
    recipe = extract_recipe(html)

    if recipe is None:
        print("  ✗ Could not extract recipe — page may require login or block scraping.")
        return None

    print("  ✓ Found structured recipe data (JSON-LD)" if not recipe["_used_fallback"]
          else "  ⚠ No JSON-LD schema found — used HTML heuristics")
    print(f"  Recipe: {recipe.get('name', '(unnamed)')}")
    print(f"  Servings: {recipe.get('servings', '-')}")
    print(f"  Prep: {recipe.get('prep_time', '-')}  Cook: {recipe.get('cook_time', '-')}  Total: {recipe.get('total_time', '-')}")
    print(f"  Ingredients: {len(recipe.get('ingredients', '').split(' | '))} items")

    recipe.pop("_used_fallback", None)
    append_to_csv(recipe, url, csv_path)
    return recipe


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recipe_scraper.py <url>")
        sys.exit(1)

    scrape_recipe(sys.argv[1])