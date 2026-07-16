# Meal Planner & Recipe Scraper

A personal meal planning system that scrapes recipes from the web, tags them using a locally hosted LLM, and generates varied weekly meal plans based on season, ease, and family preferences.

---

## Overview

### Pipeline

```
recipe_scraper.py / recipe_crawler.py
        ↓  recipes.csv
recipe_processor.py
        ↓  recipes_processed.csv
meal_planner.py  /  streamlit_app.py
```

| Script | Purpose |
|---|---|
| `recipe_scraper.py` | Scrape a single recipe URL into `recipes.csv` |
| `recipe_crawler.py` | Automatically discover and scrape many recipes from configured sites |
| `recipe_processor.py` | Tag recipes via Ollama, scale portions, compute ease scores |
| `meal_planner.py` | Generate a weekly meal plan from the CLI |
| `streamlit_app.py` | Public-facing web interface for plan generation |

All tuneable parameters live in `config.yaml`.

---

## Setup

### Dependencies

```bash
pip install selenium beautifulsoup4 webdriver-manager pyyaml requests pandas numpy streamlit
```

Requires **Google Chrome** installed. `webdriver-manager` downloads the matching ChromeDriver automatically.

### Ollama (local LLM)

Recipe tagging runs locally via [Ollama](https://ollama.com). By default this uses `qwen2.5:7b`.

```bash
ollama pull qwen2.5:7b
ollama serve
```

Ollama can run on a different machine on your network — set `ollama_base_url` in `config.yaml` accordingly.

---

## Usage

### Scrape a single recipe

```bash
python recipe_scraper.py https://www.bbc.co.uk/food/recipes/example
```

Appends one row to `recipes.csv`. Creates the file with headers if it doesn't exist.

---

### Crawl a site automatically

Discovers and scrapes multiple recipes from configured sites, stopping at `crawler.target_total_recipes` in `config.yaml`.

```bash
python recipe_crawler.py                          # all configured sites
python recipe_crawler.py --site "BBC Food"         # one site only
python recipe_crawler.py --vegetarian              # vegetarian mains only
python recipe_crawler.py --target 50               # override target count
```

Sites are configured in `config.yaml` under `sites:`. Each site entry defines:
- `category_urls.main` / `category_urls.vegetarian` — listing pages to crawl (primary discovery)
- `recipe_url_contains` — path fragment that all real recipe URLs share
- `non_recipe_url_segments` — path fragments that indicate a non-recipe page (collections, articles etc.)
- `recipe_url_pattern` — optional regex for sites where the above isn't enough (e.g. HelloFresh uses a unique ID suffix)

---

### Process recipes

Tags all unprocessed rows in `recipes.csv` using Ollama, scales portions to `family_size`, and computes ease scores. Appends results to `recipes_processed.csv`.

```bash
python recipe_processor.py
python recipe_processor.py --all    # reprocess everything from scratch (clears output file)
```

**Tags applied per recipe:**
- `meal_type` — main / dessert / snack / starter / etc.
- `tags_key_ingredients` — distinctive ingredients (chicken, tofu, pasta…)
- `tags_cuisine` — cuisine and dish format (Italian, soup, stir-fry…)
- `tags_season` — seasons the dish suits (Spring, Winter, All…)
- `tags_diet` — meat / fish / vegetarian / vegan

**Ease scoring** is computed from four signals — prep time, cook time (or total time if that's all the site provides), ingredient count, and method step count — mapped through fixed piecewise curves defined in `config.yaml`. Ease is recomputed for *all* rows every run, so tweaking `ease_curves` or `ease_weights` in config takes effect immediately without re-tagging.

**Portion scaling** targets `family_size` servings (default 4) for main meals. Non-linear ingredients (eggs, spices) are flagged with a warning.

---

### Generate a meal plan

```bash
python meal_planner.py                              # 7 meals, using config defaults
python meal_planner.py 5                            # override number of meals
python meal_planner.py --preset "Pasta Bake"        # lock in a meal, fill the rest
python meal_planner.py --diet vegetarian vegan      # filter by diet type(s)
python meal_planner.py --debug                      # verbose per-recipe scoring output
```

#### Scoring

Each recipe is scored on four components (weights configurable in `config.yaml`):

| Component | Description |
|---|---|
| **Season** | Recipes tagged for the current season score higher |
| **Rating** | Average of per-person ratings (local mode only) |
| **Ease** | Higher ease score = preferred |
| **Recency** | Recipes not eaten recently score higher (local mode only) |

Tag penalties reduce the score of recipes that share key ingredients, cuisine, or season tags with already-chosen meals — applied independently per tag category for maximum variety.

#### Two modes

Set `mode:` in `config.yaml`:

- `local` — uses `last_made` dates and per-person ratings from `planner.csv`
- `web` — ignores personal data; uses random jitter for variety instead

#### Recording meals and ratings

```bash
# Mark a meal as cooked today
python meal_planner.py --made "Veggie Chilli"

# Mark with a specific date
python meal_planner.py --made "Pasta Bake" --made-date 2025-06-10

# Rate a meal (1–5 per person)
python meal_planner.py --rate "Veggie Chilli" --person 1 --score 5
python meal_planner.py --rate "Veggie Chilli" --person 2 --score 3
```

Personal data (ratings, last made dates) is stored in `planner.csv`, kept separate from `recipes_processed.csv` so the pipeline data stays clean.

---

### Web interface

```bash
streamlit run streamlit_app.py
```

Public-facing plan generator. Uses web mode — no personal data read or written. Sidebar controls: number of meals, diet type filter. Each meal card shows cuisine, season, ease label, scaled ingredients, and numbered method steps.

---

## Configuration (`config.yaml`)

Key settings:

| Key | Description |
|---|---|
| `family_size` | Target servings for main meal scaling |
| `family_members` | Number of `person_N_rating` columns in `planner.csv` |
| `mode` | `local` or `web` |
| `meals_per_week` | Default number of meals to plan |
| `tag_penalty` | Score reduction per shared tag (0.0–1.0) |
| `score_weights` | Relative weight of season / rating / ease / recency |
| `ease_curves` | Fixed piecewise curves for each ease signal |
| `ease_weights` | Weight of each ease signal in the final score |
| `ollama_base_url` | Ollama server address (can be a remote machine) |
| `ollama_model` | Model to use for tagging |
| `sites` | Crawler site definitions |
| `common_ingredients` | Ingredients excluded from key ingredient tags |
| `dessert_snack_keywords` | URL keywords used to skip non-main recipes during crawling |
| `non_vegetarian_keywords` | Ingredient keywords used to filter out non-vegetarian recipes |

---

## Data files

| File | Created by | Purpose |
|---|---|---|
| `recipes.csv` | Scraper / crawler | Raw scraped recipes |
| `recipes_processed.csv` | Processor | Tagged, scaled, ease-scored recipes |
| `planner.csv` | Planner (local mode) | Per-recipe ratings and last-made dates |

---

## Planned extensions

- Shopping list generation
- Weather-based suggestions
- Richer web UI with recipe editing and database management