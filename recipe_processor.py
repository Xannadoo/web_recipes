"""
recipe_processor.py
===================
Reads scraped recipes CSV, enriches each with:
  - Meal type, key ingredient tags, cuisine tags, season tags (via Ollama)
  - Portion scaling to family_size for mains
  - Scaling warnings for non-linear ingredients
  - Ease score derived from prep/cook time (prep weighted more than cook)

Skips recipes already in the output CSV. Appends results to recipes_processed.csv.

Usage:
    python recipe_processor.py
    python recipe_processor.py --config path/to/config.yaml

Dependencies:
    pip install pyyaml requests
"""

import argparse
import csv
import json
import math
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

OUTPUT_HEADERS = [
    "name", "meal_type", "servings_original", "servings_scaled",
    "scale_factor", "scaling_warnings",
    "prep_time", "cook_time", "ease",
    "ingredients_original", "ingredients_scaled",
    "method", "source_url", "scraped_date",
    "tags_key_ingredients", "tags_cuisine", "tags_season",
]


def load_processed_urls(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["source_url"] for row in reader if row.get("source_url")}


def read_input_csv(path: Path) -> list:
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
# Ease score from prep + cook time
# ---------------------------------------------------------------------------

TIME_RE = re.compile(r"(?:(\d+)\s*h(?:r|ours?)?)?\s*(?:(\d+)\s*m(?:in(?:utes?)?)?)?", re.I)


def parse_minutes(time_str: str) -> float | None:
    """Parse a time string like '1h 30m', '45m', '2h' into total minutes."""
    if not time_str or not time_str.strip():
        return None
    m = TIME_RE.search(time_str.strip())
    if not m or not any(m.groups()):
        return None
    hours = int(m.group(1) or 0)
    mins = int(m.group(2) or 0)
    total = hours * 60 + mins
    return float(total) if total > 0 else None


def compute_ease(prep_str: str, cook_str: str, config: dict) -> float:
    """
    Derive an ease score (0.0 hard → 1.0 easy) from prep and cook time.
    Prep is weighted more heavily than cook (active vs passive effort).
    Returns fallback value if times are missing.
    """
    prep_w = config.get("ease_weights", {}).get("prep", 2.0)
    cook_w = config.get("ease_weights", {}).get("cook", 0.5)
    fallback = config.get("ease_fallback", 0.5)

    prep_mins = parse_minutes(prep_str)
    cook_mins = parse_minutes(cook_str)

    if prep_mins is None and cook_mins is None:
        return fallback

    effort = ((prep_mins or 0) * prep_w) + ((cook_mins or 0) * cook_w)
    # log scale: effort=0→log(2)≈0.69, effort=15→log(17)≈2.83, effort=240→log(242)≈5.49
    # We invert so more effort = lower ease, normalised roughly to 0–1
    # Using log(effort+2) mapped from [log(2), log(482)] → [1, 0]
    log_val = math.log(effort + 2)
    log_min = math.log(2)        # ~0 effort
    log_max = math.log(482)      # ~240 min prep equivalent (4h)
    ease = 1.0 - (log_val - log_min) / (log_max - log_min)
    return round(max(0.0, min(1.0, ease)), 4)


# ---------------------------------------------------------------------------
# Ingredient scaling
# ---------------------------------------------------------------------------

QUANTITY_RE = re.compile(
    r"""
    (?P<amount>
        \d+\s+\d+/\d+
        | \d+/\d+
        | \d*[.,]\d+
        | [½⅓⅔¼¾⅛⅜⅝⅞]
        | \d+
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


def parse_amount(text: str) -> float:
    text = text.strip()
    if text in UNICODE_FRACS:
        return UNICODE_FRACS[text]
    if re.match(r"^\d+\s+\d+/\d+$", text):
        whole, frac = text.split()
        return int(whole) + float(Fraction(frac))
    if "/" in text:
        return float(Fraction(text))
    return float(text.replace(",", "."))


def format_amount(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def scale_ingredient_line(line: str, factor: float) -> str:
    def replacer(m):
        try:
            value = parse_amount(m.group("amount"))
        except (ValueError, ZeroDivisionError):
            return m.group(0)
        unit = m.group("unit") or ""
        scaled = value * factor
        return f"{format_amount(scaled)}{' ' + unit if unit else ''}"
    return QUANTITY_RE.sub(replacer, line)


def scale_ingredients(ingredients_str: str, factor: float) -> str:
    if factor == 1.0:
        return ingredients_str
    return " | ".join(
        scale_ingredient_line(line, factor)
        for line in ingredients_str.split(" | ")
    )


def find_scaling_warnings(ingredients_str: str, non_linear: list) -> list:
    lower = ingredients_str.lower()
    return [
        item for item in non_linear
        if re.search(rf"\b{re.escape(item)}\b", lower)
    ]


def parse_servings(servings_str: str) -> float | None:
    m = re.search(r"\d+", servings_str or "")
    return float(m.group()) if m else None


# ---------------------------------------------------------------------------
# Ollama tagging
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are a recipe tagging assistant. Analyse the recipe below and return ONLY a valid JSON object.
No explanation, no markdown fences, no extra text — just the raw JSON.

Recipe name: {name}
Ingredients: {ingredients}
Method: {method_snippet}

Return exactly this JSON structure:
{{
  "meal_type": "<one of: {meal_types}>",
  "key_ingredients": ["ingredient1", "ingredient2"],
  "cuisine_tags": ["tag1", "tag2"],
  "season": ["<one or more of: {seasons}>"]
}}

Rules:
- meal_type: must be one of the listed values. A chilli is a main. A brownie is a dessert.
- key_ingredients: the distinctive, characterful ingredients in this dish. Do NOT include: {common_ingredients}. Max 8 items, lowercase, no quantities. Think: what makes this dish what it is?
- Use categories for the key_ingredients, e.g. "chicken", "beef", "pork", "fish", "tofu", "cheese", "pasta", "rice", "potato", "beans", "lentils". Avoid overly specific terms like "macaroni", "mashed potato".
- cuisine_tags: include BOTH the cuisine origin AND the dish format where relevant. Examples: ["Mexican", "chilli"] or ["Italian", "pasta", "vegetarian"] or ["British", "bake"]. Max 6 items.
- season: which seasons suit this dish. A chilli or stew suits Autumn/Winter. A salad suits Spring/Summer. Use "All" only if it genuinely works year-round.
"""


def call_ollama(prompt: str, config: dict) -> str:
    url = f"{config['ollama_base_url']}/api/generate"
    payload = {
        "model": config["ollama_model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 4096,      # Larger context window for 7b model
            "temperature": 0.2,   # Low temp for consistent structured output
        }
    }
    response = requests.post(url, json=payload, timeout=config["ollama_timeout"])
    response.raise_for_status()
    return response.json().get("response", "")


def extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def validate_tags(raw: dict, config: dict) -> dict:
    valid_meal_types = [m.lower() for m in config.get("meal_types", [])]
    valid_seasons = config.get("seasons", [])

    meal_type = (raw.get("meal_type") or "").lower().strip()
    if meal_type not in valid_meal_types:
        meal_type = "main"

    key_ingredients = [str(i).lower().strip() for i in (raw.get("key_ingredients") or [])][:8]
    cuisine_tags = [str(t).strip() for t in (raw.get("cuisine_tags") or [])][:6]

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
    method_snippet = (recipe.get("method") or "")[:1500]
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
            print(f"    ⚠ Could not parse JSON from model. Raw: {raw_text[:300]}")
            return _empty_tags()
        return validate_tags(raw_json, config)
    except requests.exceptions.ConnectionError:
        print("    ✗ Cannot connect to Ollama. Is it running? (ollama serve)")
        return _empty_tags()
    except Exception as e:
        print(f"    ✗ Tagging error: {e}")
        return _empty_tags()


def _empty_tags() -> dict:
    return {"meal_type": "", "key_ingredients": [], "cuisine_tags": [], "season": []}


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_recipe(recipe: dict, config: dict) -> dict:
    family_size = config.get("family_size", 4)
    non_linear = [i.lower() for i in config.get("non_linear_ingredients", [])]

    print("    Tagging via Ollama...")
    tags = tag_recipe(recipe, config)
    meal_type = tags["meal_type"]

    # Ease from time
    ease = compute_ease(recipe.get("prep_time", ""), recipe.get("cook_time", ""), config)

    # Scaling — mains only
    servings_original_str = recipe.get("servings", "").strip()
    servings_original = parse_servings(servings_original_str)
    if meal_type == "main" and servings_original and servings_original > 0:
        scale_factor = family_size / servings_original
    else:
        scale_factor = 1.0

    servings_scaled = int(round(servings_original * scale_factor)) if servings_original else ""
    ingredients_original = recipe.get("ingredients", "")
    ingredients_scaled = scale_ingredients(ingredients_original, scale_factor)

    warnings = find_scaling_warnings(ingredients_original, non_linear) if scale_factor != 1.0 else []

    return {
        "name": recipe.get("name", ""),
        "meal_type": meal_type,
        "servings_original": servings_original_str,
        "servings_scaled": servings_scaled,
        "scale_factor": f"{scale_factor:.3g}" if scale_factor != 1.0 else "1",
        "scaling_warnings": " | ".join(warnings),
        "prep_time": recipe.get("prep_time", ""),
        "cook_time": recipe.get("cook_time", ""),
        "ease": ease,
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--all", action="store_true",
        help="Force reprocessing of all recipes, ignoring what's already in the "
             "output CSV. This overwrites the output file from scratch."
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"✗ Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    input_path = Path(config["scraped_csv"])
    output_path = Path(config["processed_csv"])

    if args.all and output_path.exists():
        output_path.unlink()
        print(f"--all set: cleared {output_path} — reprocessing everything.\n")

    print(f"Reading from:   {input_path.resolve()}")
    print(f"Writing to:     {output_path.resolve()}")
    print(f"Model:          {config['ollama_model']}")
    print(f"Family size:    {config['family_size']}\n")

    recipes = read_input_csv(input_path)
    already_done = load_processed_urls(output_path)
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
            print(f"    ✓ meal_type={row['meal_type']}  ease={row['ease']}  "
                  f"scale={row['scale_factor']}x  seasons={row['tags_season']}")
            if row["scaling_warnings"]:
                print(f"    ⚠ Scaling warnings: {row['scaling_warnings']}")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
        print()

    print(f"Complete. Results in: {output_path.resolve()}")


if __name__ == "__main__":
    main()