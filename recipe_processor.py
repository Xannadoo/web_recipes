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
    "prep_time", "cook_time", "total_time", "ease",
    "ingredients_original", "ingredients_scaled",
    "method", "source_url", "scraped_date",
    "tags_key_ingredients", "tags_cuisine", "tags_season", "tags_diet"
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


def count_ingredients(ingredients_str: str) -> int:
    """Count pipe-separated ingredient lines."""
    if not ingredients_str or not ingredients_str.strip():
        return 0
    return len([i for i in ingredients_str.split("|") if i.strip()])


def count_steps(method_str: str) -> int:
    """
    Count method steps. Most scraped recipes have pipe-separated steps from
    JSON-LD's recipeInstructions list. If a site gives one big paragraph
    instead (no pipes), fall back to counting sentences as an approximation.
    """
    if not method_str or not method_str.strip():
        return 0
    items = [s for s in method_str.split("|") if s.strip()]
    if len(items) > 1:
        return len(items)
    sentences = re.split(r"(?<=[.!?])\s+", method_str.strip())
    return len([s for s in sentences if s.strip()])


def piecewise_ease(value: float, anchors: list) -> float:
    """
    Linear interpolation through a set of fixed (x, ease) anchor points,
    e.g. [[0, 1.0], [20, 0.85], [45, 0.5], [90, 0.15]] for prep minutes.
    Clamps to the first/last anchor's ease value outside the given range.
    Anchors are absolute, fixed thresholds — NOT relative to the recipe
    pool — so a recipe's ease doesn't shift just because other recipes are
    added or removed later.
    """
    if not anchors:
        return 0.5
    anchors = sorted(anchors, key=lambda p: p[0])
    if value <= anchors[0][0]:
        return anchors[0][1]
    if value >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= value <= x1:
            frac = (value - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + frac * (y1 - y0)
    return anchors[-1][1]


def compute_ease(
    prep_str: str, cook_str: str, ingredients_str: str, method_str: str,
    config: dict, total_str: str = "",
) -> float:
    """
    Derive an ease score (0.0 hard → 1.0 easy) from up to five signals: prep
    time, cook time, total time, ingredient count, and step count. Each
    signal is mapped through its own fixed piecewise curve (config:
    ease_curves), then combined with weights (config: ease_weights).
    Missing signals are simply excluded from the weighted average rather
    than penalised.

    total_time is only used when prep_time and cook_time aren't BOTH
    available — some sites (e.g. HelloFresh) only publish a combined total
    time rather than the two separately. Using it alongside prep/cook when
    both of those are already present would double-count the same effort.

    Fixed absolute thresholds are used deliberately (not pool-relative
    percentile ranking) — a 25-minute prep should always read as "quick",
    regardless of what else is in the recipe database at the time.
    """
    curves = config.get("ease_curves", {})
    weights = config.get("ease_weights", {})
    fallback = config.get("ease_fallback", 0.5)

    prep_mins = parse_minutes(prep_str)
    cook_mins = parse_minutes(cook_str)
    total_mins = parse_minutes(total_str)
    ing_count = count_ingredients(ingredients_str)
    step_count = count_steps(method_str)

    components = {}
    if prep_mins is not None and curves.get("prep_minutes"):
        components["prep"] = piecewise_ease(prep_mins, curves["prep_minutes"])
    if cook_mins is not None and curves.get("cook_minutes"):
        components["cook"] = piecewise_ease(cook_mins, curves["cook_minutes"])
    if (prep_mins is None or cook_mins is None) and total_mins is not None and curves.get("total_minutes"):
        components["total"] = piecewise_ease(total_mins, curves["total_minutes"])
    if ing_count > 0 and curves.get("ingredient_count"):
        components["ingredients"] = piecewise_ease(ing_count, curves["ingredient_count"])
    if step_count > 0 and curves.get("step_count"):
        components["steps"] = piecewise_ease(step_count, curves["step_count"])

    if not components:
        return fallback

    total_weight = sum(weights.get(k, 1.0) for k in components)
    if total_weight <= 0:
        return fallback

    weighted_sum = sum(components[k] * weights.get(k, 1.0) for k in components)
    ease = weighted_sum / total_weight
    return round(max(0.0, min(1.0, ease)), 4)


def recompute_all_ease(output_path: Path, config: dict) -> None:
    """
    Recompute the ease column for every row in the processed CSV using the
    current curves/weights in config.yaml — including rows processed long
    ago under an older formula. No Ollama calls involved, so this is cheap
    to run every time and lets you tune ease_curves/ease_weights and see
    the effect immediately without reprocessing tags.
    """
    if not output_path.exists():
        return
    with output_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    for row in rows:
        row["ease"] = compute_ease(
            row.get("prep_time", ""), row.get("cook_time", ""),
            row.get("ingredients_original", ""), row.get("method", ""),
            config, total_str=row.get("total_time", ""),
        )
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Recomputed ease for {len(rows)} recipe(s) using current ease_curves/ease_weights.")


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
  "diet_type": "<one of: meat, fish, vegetarian, vegan>",
  "season": ["<one or more of: {seasons}>"]
}}

Rules:
- meal_type: must be one of the listed values. A chilli is a main. A brownie is a dessert.
- key_ingredients: the distinctive, characterful ingredients in this dish. Do NOT include: {common_ingredients}, herbs, spices (including chilli) etc or the word 'vegetable'. Max 4 items, lowercase, no quantities, singular form (tomato rather than tomatoes, courgette rather than courgettes). Think: what makes this dish what it is? Typically this will be the main protein/s and carbohydrate/s. Do not include vegetables unless they are the main feature of the dish (eg "tomato" for a tomato soup). 
- key_ingredients: Use categories, e.g. "chicken", "beef", "pork", "fish", "tofu", "cheese", "pasta", "rice", "potato", "bean", "lentil". Do not use specific terms, examples here in the form "specific item":"generic term", "macaroni":"pasta", "mashed potato":"potato", "spaghetti":"pasta", "parmesan":"cheese", "cheddar":"cheese", "hallumi":"cheese", "black beans":"bean", "vegan mince":"mince", "vegetarian mince":"mince". Do not include accompaniments like "mango chutney", "salsa", "sour cream" etc.
- cuisine_tags: include BOTH the cuisine origin AND the dish format where relevant. Examples: ["Mexican", "chilli"] or ["Italian", "pasta"] or ["British", "bake"]. Max 6 items.
- diet_type: one and only one of ["meat", "fish", "vegetarian", "vegan"]
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
    
    diet_type = (raw.get("diet_type") or "").lower().strip()
    if diet_type not in ["meat", "fish", "vegetarian", "vegan"]:
        diet_type = ["meat"]  # Default to meat if not specified
    else: diet_type = [diet_type]

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
        "diet_type": diet_type
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
        diet_types=", ".join(config.get("diet_types", []))
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
    return {"meal_type": "", "key_ingredients": [], "cuisine_tags": [], "season": [], "diet_type": []}


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_recipe(recipe: dict, config: dict) -> dict:
    family_size = config.get("family_size", 4)
    non_linear = [i.lower() for i in config.get("non_linear_ingredients", [])]

    print("    Tagging via Ollama...")
    tags = tag_recipe(recipe, config)
    meal_type = tags["meal_type"]

    # Ease from prep/cook/total time + ingredient count + step count
    ease = compute_ease(
        recipe.get("prep_time", ""), recipe.get("cook_time", ""),
        recipe.get("ingredients", ""), recipe.get("method", ""),
        config, total_str=recipe.get("total_time", ""),
    )

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
        "total_time": recipe.get("total_time", ""),
        "ease": ease,
        "ingredients_original": ingredients_original,
        "ingredients_scaled": ingredients_scaled,
        "method": recipe.get("method", ""),
        "source_url": recipe.get("source_url", ""),
        "scraped_date": recipe.get("scraped_date", ""),
        "tags_key_ingredients": " | ".join(tags["key_ingredients"]),
        "tags_cuisine": " | ".join(tags["cuisine_tags"]),
        "tags_diet": " | ".join(tags["diet_type"]),
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
        print(f"Skipping {skipped} already-processed recipe(s) for tagging.\n")

    if not to_process:
        print("Nothing new to tag.")
    else:
        print(f"Processing {len(to_process)} recipe(s)...\n")
        for i, recipe in enumerate(to_process, 1):
            name = recipe.get("name") or recipe.get("source_url") or f"Row {i}"
            print(f"[{i}/{len(to_process)}] {name}")
            try:
                row = process_recipe(recipe, config)
                #print(row)
                append_output_csv(output_path, row)
                print(f"    ✓ meal_type={row['meal_type']}  key ingredients={row['tags_key_ingredients']}  diet_type={row['tags_diet']}")
                if row["scaling_warnings"]:
                    print(f"    ⚠ Scaling warnings: {row['scaling_warnings']}")
            except Exception as e:
                print(f"    ✗ Failed: {e}")
            print()

    # Ease uses fixed, absolute thresholds (not pool-relative), so this is
    # just a cheap backfill — no Ollama calls — that lets tuning
    # ease_curves/ease_weights in config.yaml take effect on every recipe,
    # including ones tagged long ago, without a full --all reprocess.
    print("Recomputing ease for all recipes...")
    recompute_all_ease(output_path, config)

    print(f"\nComplete. Results in: {output_path.resolve()}")


if __name__ == "__main__":
    main()