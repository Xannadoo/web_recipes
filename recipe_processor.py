"""
recipe_processor.py
===================
Reads recipes.csv (output of recipe_scraper.py), enriches each recipe with:
  - Meal type detection (main / dessert / snack / etc.)
  - Portion scaling to family_size for mains
  - Scaling warnings for non-linear ingredients (spices, eggs, etc.)
  - Key ingredient tags (notable ingredients, excluding common ones)
  - Cuisine / dish type tags (Chinese, soup, bake, etc.)
  - Season tags (Spring / Summer / Autumn / Winter / All)

Skips recipes already present in the output CSV.
Writes enriched rows to recipes_processed.csv.

Tagging is done via a local Ollama model (default: qwen2.5:7b).

Usage:
    python recipe_processor.py
    python recipe_processor.py --config path/to/config.yaml

Dependencies:
    pip install pyyaml requests
"""

import argparse
import csv
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import requests
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = Path("config.yaml")

def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

INPUT_HEADERS = [
    "name", "servings", "prep_time", "cook_time",
    "ingredients", "method", "source_url", "scraped_date"
]

OUTPUT_HEADERS = [
    "name", "meal_type", "servings_original", "servings_scaled",
    "scale_factor", "scaling_warnings",
    "prep_time", "cook_time",
    "ingredients_original", "ingredients_scaled",
    "method", "source_url", "scraped_date",
    "tags_key_ingredients", "tags_cuisine", "tags_season",
]


def load_processed_names(path: Path) -> set[str]:
    """Return the set of source_urls already in the processed CSV."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["source_url"] for row in reader if row.get("source_url")}


def read_input_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"✗ Input file not found: {path}")
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_output_csv(path: Path, row: dict) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Ingredient scaling
# ---------------------------------------------------------------------------

# Matches patterns like:  200g  /  2 tbsp  /  1/2 cup  /  3  /  1½
QUANTITY_RE = re.compile(
    r"""
    (?P<amount>
        \d+\s+\d+/\d+       # mixed number:  1 1/2
        | \d+/\d+            # fraction:       3/4
        | \d*[.,]\d+         # decimal:        0.5 / 1,5
        | [½⅓⅔¼¾⅛⅜⅝⅞]      # unicode fracs
        | \d+                # whole number
    )
    \s*
    (?P<unit>
        tbsp|tbs|tablespoons?|
        tsp|teaspoons?|
        cups?|
        fl\.?\s*oz|fluid\s+ounces?|
        ml|millilitres?|milliliters?|
        [lL](?=\s|$)|litres?|liters?|
        g(?=\s|$)|grams?|
        kg|kilograms?|
        oz(?=\s|$)|ounces?|
        lbs?|pounds?|
        pinch(?:es)?|
        handfuls?|
        slices?|
        cloves?|
        cans?|
        tins?|
        bunches?|
        sprigs?
    )?
    """,
    re.VERBOSE | re.IGNORECASE
)

UNICODE_FRACS = {
    "½": 0.5, "⅓": 1/3, "⅔": 2/3,
    "¼": 0.25, "¾": 0.75,
    "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
}


def parse_amount(text: str) -> float | None:
    """Parse a matched amount string to a float."""
    text = text.strip()
    # Unicode fraction
    if text in UNICODE_FRACS:
        return UNICODE_FRACS[text]
    # Mixed number: "1 1/2"
    if re.match(r"^\d+\s+\d+/\d+$", text):
        whole, frac = text.split()
        return int(whole) + float(Fraction(frac))
    # Standard fraction
    if "/" in text:
        return float(Fraction(text))
    # Decimal (handle comma as separator)
    return float(text.replace(",", "."))


def format_amount(value: float) -> str:
    """Return a clean string for a scaled amount."""
    if value == int(value):
        return str(int(value))
    # Round to 2 dp, strip trailing zeros
    return f"{value:.2f}".rstrip("0").rstrip(".")


def scale_ingredient_line(line: str, factor: float) -> str:
    """Scale all quantities in a single ingredient string."""
    def replacer(m: re.Match) -> str:
        amount_str = m.group("amount")
        unit = m.group("unit") or ""
        try:
            value = parse_amount(amount_str)
        except (ValueError, ZeroDivisionError):
            return m.group(0)
        scaled = value * factor
        return f"{format_amount(scaled)}{' ' + unit if unit else ''}"

    return QUANTITY_RE.sub(replacer, line)


def scale_ingredients(ingredients_str: str, factor: float) -> str:
    """Scale all pipe-separated ingredient lines."""
    if factor == 1.0:
        return ingredients_str
    lines = ingredients_str.split(" | ")
    return " | ".join(scale_ingredient_line(line, factor) for line in lines)


def find_scaling_warnings(ingredients_str: str, non_linear: list[str]) -> list[str]:
    """Return ingredient names that appear in the non-linear list."""
    warnings = []
    lower = ingredients_str.lower()
    for item in non_linear:
        # Match whole word
        if re.search(rf"\b{re.escape(item)}\b", lower):
            warnings.append(item)
    return warnings


def parse_servings(servings_str: str) -> float | None:
    """Extract the first number from a servings string like '4', '4-6', 'Serves 4'."""
    m = re.search(r"\d+", servings_str or "")
    return float(m.group()) if m else None


# ---------------------------------------------------------------------------
# Ollama tagging
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are a recipe tagging assistant. Analyse the recipe below and return ONLY a valid JSON object — no explanation, no markdown, no extra text.

Recipe name: {name}
Ingredients: {ingredients}
Method summary: {method_snippet}

Return this exact JSON structure:
{{
  "meal_type": "<one of: {meal_types}>",
  "key_ingredients": ["ingredient1", "ingredient2"],
  "cuisine_tags": ["tag1", "tag2"],
  "season": ["<one or more of: {seasons}>"]
}}

Rules:
- meal_type: classify as one of the allowed values only.
- key_ingredients: notable/distinctive ingredients only, ie main protein (tofu, egg, chicken, prawns), main carbohydrate (potato, pasta, rice). Exclude common basics like: {common_ingredients} unless they are a major part of the dish, eg onions in french onion soup. Exclude vegetables unless a main part of the dish, ie cauliflower in cauliflower cheese. Max 8 items. Lowercase, no quantities.
- cuisine_tags: cuisine style AND dish format where relevant, e.g. ["Italian", "pasta"] or ["Chinese", "stir-fry"] or ["soup", "vegetarian"]. Max 6 items.
- season: which seasons this dish suits best. Use "All" if it works year-round.
"""


def call_ollama(prompt: str, config: dict) -> str:
    """Send a prompt to Ollama and return the raw text response."""
    url = f"{config['ollama_base_url']}/api/generate"
    payload = {
        "model": config["ollama_model"],
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=config["ollama_timeout"])
    response.raise_for_status()
    return response.json().get("response", "")


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a string (LLMs sometimes add preamble)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def validate_tags(raw: dict, config: dict) -> dict:
    """Validate and clean LLM output against allowed values in config."""
    valid_meal_types = [m.lower() for m in config.get("meal_types", [])]
    valid_seasons = config.get("seasons", [])

    meal_type = (raw.get("meal_type") or "").lower().strip()
    if meal_type not in valid_meal_types:
        meal_type = "main"  # safe default

    key_ingredients = [
        str(i).lower().strip()
        for i in (raw.get("key_ingredients") or [])
    ][:8]

    cuisine_tags = [
        str(t).strip()
        for t in (raw.get("cuisine_tags") or [])
    ][:6]

    raw_seasons = raw.get("season") or ["All"]
    if isinstance(raw_seasons, str):
        raw_seasons = [raw_seasons]
    season = [s for s in raw_seasons if s in valid_seasons] or ["All"]

    return {
        "meal_type": meal_type,
        "key_ingredients": key_ingredients,
        "cuisine_tags": cuisine_tags,
        "season": season,
    }


def tag_recipe(recipe: dict, config: dict) -> dict:
    """Call Ollama to tag a recipe. Returns validated tag dict."""
    # Truncate method to keep prompt short for small models
    method_snippet = (recipe.get("method") or "")[:2000]
    print(f"{len((recipe.get('method') or ''))}, {len(method_snippet)}")

    prompt = PROMPT_TEMPLATE.format(
        name=recipe.get("name", ""),
        ingredients=recipe.get("ingredients", ""),
        method_snippet=method_snippet,
        meal_types=", ".join(config.get("meal_types", [])),
        seasons=", ".join(config.get("seasons", [])),
        common_ingredients=", ".join(config.get("common_ingredients", [])),
    )

    try:
        raw_text = call_ollama(prompt, config)
        raw_json = extract_json(raw_text)
        if raw_json is None:
            print(f"    ⚠ Could not parse JSON from model response. Raw: {raw_text[:200]}")
            return _empty_tags()
        return validate_tags(raw_json, config)
    except requests.exceptions.ConnectionError:
        print("    ✗ Could not connect to Ollama. Is it running? (ollama serve)")
        return _empty_tags()
    except Exception as e:
        print(f"    ✗ Tagging error: {e}")
        return _empty_tags()


def _empty_tags() -> dict:
    return {
        "meal_type": "",
        "key_ingredients": [],
        "cuisine_tags": [],
        "season": [],
    }


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_recipe(recipe: dict, config: dict) -> dict:
    """Enrich a single recipe row and return the output row."""
    family_size = config.get("family_size", 4)
    non_linear = [i.lower() for i in config.get("non_linear_ingredients", [])]

    # --- Tagging via Ollama ---
    print("    Tagging via Ollama...")
    tags = tag_recipe(recipe, config)
    meal_type = tags["meal_type"]

    # --- Portion scaling (mains only) ---
    servings_original_str = recipe.get("servings", "").strip()
    servings_original = parse_servings(servings_original_str)

    if meal_type == "main" and servings_original and servings_original > 0:
        scale_factor = family_size / servings_original
    else:
        scale_factor = 1.0

    servings_scaled = (
        int(round(servings_original * scale_factor))
        if servings_original else ""
    )

    ingredients_original = recipe.get("ingredients", "")
    ingredients_scaled = scale_ingredients(ingredients_original, scale_factor)

    # --- Scaling warnings ---
    if scale_factor != 1.0:
        warnings = find_scaling_warnings(ingredients_original, non_linear)
    else:
        warnings = []

    return {
        "name": recipe.get("name", ""),
        "meal_type": meal_type,
        "servings_original": servings_original_str,
        "servings_scaled": servings_scaled,
        "scale_factor": f"{scale_factor:.3g}" if scale_factor != 1.0 else "1",
        "scaling_warnings": " | ".join(warnings),
        "prep_time": recipe.get("prep_time", ""),
        "cook_time": recipe.get("cook_time", ""),
        "ingredients_original": ingredients_original,
        "ingredients_scaled": ingredients_scaled,
        "method": recipe.get("method", ""),
        "source_url": recipe.get("source_url", ""),
        "scraped_date": recipe.get("scraped_date", ""),
        "tags_key_ingredients": " | ".join(tags["key_ingredients"]),
        "tags_cuisine": " | ".join(tags["cuisine_tags"]),
        "tags_season": " | ".join(tags["season"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Process and tag scraped recipes.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Path to config.yaml (default: ./config.yaml)"
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"✗ Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    input_path = Path(config["input_csv"])
    output_path = Path(config["output_csv"])

    print(f"Reading recipes from:    {input_path.resolve()}")
    print(f"Writing processed to:    {output_path.resolve()}")
    print(f"Model:                   {config['ollama_model']}")
    print(f"Target family size:      {config['family_size']}\n")

    recipes = read_input_csv(input_path)
    already_done = load_processed_names(output_path)

    to_process = [r for r in recipes if r.get("source_url") not in already_done]
    skipped = len(recipes) - len(to_process)

    if skipped:
        print(f"Skipping {skipped} already-processed recipe(s).\n")
    if not to_process:
        print("Nothing new to process.")
        return

    print(f"Processing {len(to_process)} recipe(s)...\n")

    for i, recipe in enumerate(to_process, 1):
        name = recipe.get("name") or recipe.get("source_url") or f"Row {i}"
        print(f"[{i}/{len(to_process)}] {name}")
        try:
            row = process_recipe(recipe, config)
            append_output_csv(output_path, row)
            print(f"    ✓ Done — meal_type: {row['meal_type']}, "
                  f"scale: {row['scale_factor']}x, "
                  f"seasons: {row['tags_season']}")
            if row["scaling_warnings"]:
                print(f"    ⚠ Scaling warnings: {row['scaling_warnings']}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
        print()

    print(f"Complete. Results in: {output_path.resolve()}")


if __name__ == "__main__":
    main()