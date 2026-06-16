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
    "name", "servings", "prep_time", "cook_time",
    "ingredients", "method", "source_url", "scraped_date"
]


# ---------------------------------------------------------------------------
# Browser / page fetching
# ---------------------------------------------------------------------------

def get_page_html(url: str) -> str:
    """Fetch fully-rendered HTML using a headless Chrome browser."""
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
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        time.sleep(3)  # Allow JS to render; increase if needed for slow sites
        return driver.page_source
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


def extract_text(value) -> str:
    """Safely pull a string from a schema value that may be str or dict."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return value.get("text", "").strip()
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
            # or list of HowToStep objects
            raw_steps = obj.get("recipeInstructions", [])
            if isinstance(raw_steps, str):
                method = raw_steps.strip()
            else:
                steps = []
                for step in raw_steps:
                    if isinstance(step, str):
                        steps.append(step.strip())
                    elif isinstance(step, dict):
                        text = step.get("text", "") or step.get("name", "")
                        if text:
                            steps.append(text.strip())
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
    prep_time, cook_time = "", ""
    for tag in soup.find_all(class_=re.compile(r"prep.?time|preptime", re.I)):
        prep_time = clean_text(tag.get_text())
        break
    for tag in soup.find_all(class_=re.compile(r"cook.?time|cooktime", re.I)):
        cook_time = clean_text(tag.get_text())
        break

    return {
        "name": name,
        "servings": servings,
        "prep_time": prep_time,
        "cook_time": cook_time,
        "ingredients": ingredients,
        "method": method,
    }


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def append_to_csv(recipe: dict, url: str) -> None:
    file_exists = CSV_FILE.exists()
    with CSV_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            **recipe,
            "source_url": url,
            "scraped_date": date.today().isoformat(),
        })
    print(f"  ✓ Saved to {CSV_FILE.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape_recipe(url: str) -> None:
    print(f"\nFetching: {url}")
    html = get_page_html(url)
    soup = BeautifulSoup(html, "html.parser")

    print("  Trying JSON-LD schema...")
    recipe = try_json_ld(soup)

    if recipe:
        print("  ✓ Found structured recipe data (JSON-LD)")
    else:
        print("  ⚠ No JSON-LD schema found — falling back to HTML heuristics")
        recipe = try_html_heuristics(soup)

    # Validate we got something useful
    if not recipe.get("name") and not recipe.get("ingredients"):
        print("  ✗ Could not extract recipe — page may require login or block scraping.")
        return

    print(f"  Recipe: {recipe.get('name', '(unnamed)')}")
    print(f"  Servings: {recipe.get('servings', '-')}")
    print(f"  Prep: {recipe.get('prep_time', '-')}  Cook: {recipe.get('cook_time', '-')}")
    print(f"  Ingredients: {len(recipe.get('ingredients', '').split(' | '))} items")

    append_to_csv(recipe, url)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recipe_scraper.py <url>")
        sys.exit(1)

    scrape_recipe(sys.argv[1])