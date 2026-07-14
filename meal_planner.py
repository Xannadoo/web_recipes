"""
meal_planner.py
===============
Generates a weekly meal plan from processed recipes.

Data sources (configured in config.yaml):
  - recipes_processed.csv  — enriched recipes from recipe_processor.py
                             (tags, ease, scaled ingredients etc.)
  - planner.csv            — personal data: last_made, per-person ratings
                             Created automatically on first run.

Two modes (set mode: in config.yaml):
  "local"  — uses last_made recency and per-person ratings in scoring
  "web"    — ignores personal data; uses jitter for variety instead

Usage:
    python meal_planner.py                    # 7 meals, from config
    python meal_planner.py 5                  # override meal count
    python meal_planner.py --config my.yaml   # custom config path
    python meal_planner.py --debug            # verbose scoring output

    # Update personal data after cooking:
    python meal_planner.py --made "Veggie Chilli"
    python meal_planner.py --rate "Veggie Chilli" --person 1 --score 4

Dependencies:
    pip install pandas numpy pyyaml
"""

import argparse
import datetime
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = Path("config.yaml")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def current_season() -> str:
    month = datetime.datetime.now().month
    if month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    elif month in (9, 10, 11):
        return "Autumn"
    else:
        return "Winter"


# ---------------------------------------------------------------------------
# Planner CSV — personal data store
# ---------------------------------------------------------------------------

def planner_columns(n_people: int) -> list:
    rating_cols = [f"person_{i}_rating" for i in range(1, n_people + 1)]
    return ["name", "last_made"] + rating_cols


def load_planner(path: Path, n_people: int) -> pd.DataFrame:
    """Load planner.csv, creating it if absent."""
    cols = planner_columns(n_people)
    if not path.exists():
        df = pd.DataFrame(columns=cols)
        df.to_csv(path, index=False)
        return df
    df = pd.read_csv(path, index_col=False)
    # Add any missing columns (e.g. after increasing family_members)
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def save_planner(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def ensure_planner_row(planner: pd.DataFrame, name: str, n_people: int) -> pd.DataFrame:
    """Add a blank row for a recipe if it doesn't exist in the planner yet."""
    if name not in planner["name"].values:
        row = {"name": name, "last_made": pd.NaT}
        for i in range(1, n_people + 1):
            row[f"person_{i}_rating"] = np.nan
        planner = pd.concat([planner, pd.DataFrame([row])], ignore_index=True)
    return planner


# ---------------------------------------------------------------------------
# Data loading + merging
# ---------------------------------------------------------------------------

def load_and_merge(config: dict) -> pd.DataFrame:
    """
    Load recipes_processed.csv and planner.csv, merge on name,
    and compute all scoring components.
    """
    processed_path = Path(config["processed_csv"])
    planner_path = Path(config["planner_csv"])
    n_people = config.get("family_members", 4)
    mode = config.get("mode", "local")

    if not processed_path.exists():
        print(f"✗ Processed recipes not found: {processed_path}")
        print("  Run recipe_processor.py first.")
        sys.exit(1)

    processed = pd.read_csv(processed_path, index_col=False)

    # Only plan mains for now (could make meal_type filter configurable later)
    mains = processed[processed["meal_type"] == "main"].copy()
    if mains.empty:
        print("✗ No main meals found in processed recipes.")
        sys.exit(1)

    planner = load_planner(planner_path, n_people)

    # Ensure every main has a planner row
    for name in mains["name"]:
        planner = ensure_planner_row(planner, name, n_people)
    save_planner(planner, planner_path)

    # Merge
    df = mains.merge(planner, on="name", how="left")

    # --- Season score ---
    season = current_season()
    season_adjacency = {
        "Spring":  {"Spring": 1.0, "Summer": 0.7, "Winter": 0.7, "Autumn": 0.3},
        "Summer":  {"Summer": 1.0, "Spring": 0.7, "Autumn": 0.7, "Winter": 0.3},
        "Autumn":  {"Autumn": 1.0, "Summer": 0.7, "Winter": 0.7, "Spring": 0.3},
        "Winter":  {"Winter": 1.0, "Autumn": 0.7, "Spring": 0.7, "Summer": 0.3},
    }

    def season_score(tag_str: str) -> float:
        tags = [t.strip() for t in str(tag_str or "").split("|") if t.strip()]
        if not tags or "All" in tags:
            return 0.8
        weights = season_adjacency.get(season, {})
        return max((weights.get(t, 0.1) for t in tags), default=0.1)

    df["score_season"] = df["tags_season"].apply(season_score)

    # --- Rating score ---
    rating_cols = [f"person_{i}_rating" for i in range(1, n_people + 1)]
    existing_rating_cols = [c for c in rating_cols if c in df.columns]

    if mode == "local" and existing_rating_cols:
        df["avg_rating"] = df[existing_rating_cols].mean(axis=1)
        # Fill unrated recipes with the middle of the scale (assume 1-5)
        df["avg_rating"] = df["avg_rating"].fillna(3.0)
        max_rating = df["avg_rating"].max()
        df["score_rating"] = df["avg_rating"] / max_rating if max_rating > 0 else 0.5
    else:
        df["score_rating"] = 0.5  # neutral in web mode

    # --- Ease score (already computed by processor, 0-1) ---
    fallback_ease = config.get("ease_fallback", 0.5)
    df["score_ease"] = pd.to_numeric(df.get("ease"), errors="coerce").fillna(fallback_ease)

    # --- Recency score ---
    if mode == "local":
        today = datetime.date.today()
        df["last_made"] = pd.to_datetime(df["last_made"], errors="coerce")
        df["days_since"] = df["last_made"].apply(
            lambda d: (today - d.date()).days if pd.notna(d) else 30
        )
        max_days = df["days_since"].max()
        df["score_recency"] = df["days_since"] / max_days if max_days > 0 else 0.5
    else:
        df["score_recency"] = 0.0  # replaced by jitter in web mode

    # --- Composite score ---
    w = config.get("score_weights", {})
    w_season   = w.get("season",  0.25)
    w_rating   = w.get("rating",  0.30)
    w_ease     = w.get("ease",    0.20)
    w_recency  = w.get("recency", 0.25)

    if mode == "web":
        # Redistribute recency weight equally across the other three
        extra = w_recency / 3
        w_season += extra
        w_rating += extra
        w_ease   += extra
        w_recency = 0.0

    df["score"] = (
        w_season  * df["score_season"]  +
        w_rating  * df["score_rating"]  +
        w_ease    * df["score_ease"]    +
        w_recency * df["score_recency"]
    )

    return df


# ---------------------------------------------------------------------------
# Tag penalty + selection
# ---------------------------------------------------------------------------

def split_pipe(value: str) -> list:
    """Split a pipe-separated tag string into a clean list."""
    if not value or pd.isna(value):
        return []
    return [t.strip().lower() for t in str(value).split("|") if t.strip()]


def select_with_tag_penalty(
    available: pd.DataFrame,
    selected_tags: dict,   # {"key_ingredients": [...], "cuisine": [...], "season": [...]}
    tag_penalty: float,
    jitter_strength: float,
    rng: np.random.Generator,
) -> tuple:
    """
    Pick one recipe using score-weighted probability, applying tag penalties
    and jitter. Penalties are applied separately per tag category for
    better variety control.

    Returns (chosen_name, dynamic_scores_series, normalised_probs_series)
    """
    scores = available["score"].copy().astype(float)

    tag_fields = {
        "key_ingredients": "tags_key_ingredients",
        "cuisine":         "tags_cuisine",
        "season":          "tags_season",
    }

    for idx in available.index:
        row = available.loc[idx]
        for category, col in tag_fields.items():
            recipe_tags = split_pipe(row.get(col, ""))
            already_used = selected_tags.get(category, [])
            shared = sum(already_used.count(t) for t in recipe_tags)
            if shared > 0:
                scores.loc[idx] *= (1.0 - tag_penalty) ** shared
                scores.loc[idx] = max(0.01, scores.loc[idx])

    # Add jitter
    max_score = scores.max()
    noise = rng.uniform(0, jitter_strength * max_score, size=len(scores))
    scores = scores + noise

    total = scores.sum()
    if total <= 0:
        probs = np.ones(len(available)) / len(available)
    else:
        probs = (scores / total).values

    chosen_name = rng.choice(available["name"].values, p=probs)
    return chosen_name, scores, pd.Series(probs, index=available.index)


# ---------------------------------------------------------------------------
# Meal plan generation
# ---------------------------------------------------------------------------

def get_meals(
    df: pd.DataFrame,
    config: dict,
    n: Optional[int] = None,
    preset_meals: Optional[list] = None,
    debug: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> list:
    """
    Select n meals from df, using weighted probabilistic selection
    with tag penalties for variety.

    Args:
        df:           Merged dataframe with scores.
        config:       Config dict.
        n:            Number of meals to select (defaults to config meals_per_week).
        preset_meals: Meals already chosen (fills remaining slots).
        debug:        Print scoring detail.
        rng:          Random generator (for reproducibility in tests).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = n or config.get("meals_per_week", 7)
    tag_penalty = config.get("tag_penalty", 0.4)
    jitter_strength = config.get("jitter_strength", 0.05)

    preset_meals = preset_meals or []

    # Validate presets exist
    valid_names = set(df["name"].values)
    valid_presets = []
    for meal in preset_meals:
        if meal in valid_names:
            valid_presets.append(meal)
        else:
            print(f"  ⚠ Preset '{meal}' not found in recipes — skipping.")

    n = min(n, len(df))
    selected = valid_presets.copy()

    if len(selected) >= n:
        print(f"  Preset meals already fill all {n} slots.")
        return selected[:n]

    # Initialise tag tracking from presets
    tag_fields = {
        "key_ingredients": "tags_key_ingredients",
        "cuisine":         "tags_cuisine",
        "season":          "tags_season",
    }
    selected_tags: dict = {cat: [] for cat in tag_fields}
    for meal in valid_presets:
        row = df[df["name"] == meal].iloc[0]
        for cat, col in tag_fields.items():
            selected_tags[cat].extend(split_pipe(row.get(col, "")))

    available = df[~df["name"].isin(selected)].copy()

    for i in range(n - len(selected)):
        if available.empty:
            break

        chosen, dyn_scores, probs = select_with_tag_penalty(
            available, selected_tags, tag_penalty, jitter_strength, rng
        )
        selected.append(chosen)

        if debug:
            print(f"\n  [{i+1}] Selected: '{chosen}'")
            print("  Scores (after penalty + jitter):")
            debug_df = available[["name"]].copy()
            debug_df["raw_score"] = available["score"].values
            debug_df["adj_score"] = dyn_scores.values
            debug_df["prob"] = probs.values
            debug_df = debug_df.sort_values("adj_score", ascending=False)
            for _, r in debug_df.head(10).iterrows():
                marker = " ◀" if r["name"] == chosen else ""
                print(f"    {r['name']:<40} raw={r['raw_score']:.3f}  adj={r['adj_score']:.3f}  p={r['prob']:.4f}{marker}")

        # Update tag tracking
        chosen_row = available[available["name"] == chosen].iloc[0]
        for cat, col in tag_fields.items():
            selected_tags[cat].extend(split_pipe(chosen_row.get(col, "")))

        available = available[available["name"] != chosen]

    return selected


# ---------------------------------------------------------------------------
# Personal data updates
# ---------------------------------------------------------------------------

def mark_made(name: str, config: dict, date: Optional[datetime.date] = None) -> None:
    """Update last_made for a recipe in planner.csv."""
    planner_path = Path(config["planner_csv"])
    n_people = config.get("family_members", 4)
    planner = load_planner(planner_path, n_people)
    planner = ensure_planner_row(planner, name, n_people)

    date_str = str(date or datetime.date.today())

    if name not in planner["name"].values:
        print(f"  ✗ '{name}' not found in planner.")
        return

    planner.loc[planner["name"] == name, "last_made"] = date_str
    save_planner(planner, planner_path)
    print(f"  ✓ Marked '{name}' as made on {date_str}.")


def update_rating(name: str, person: int, score: float, config: dict) -> None:
    """Update a single person's rating for a recipe."""
    planner_path = Path(config["planner_csv"])
    n_people = config.get("family_members", 4)

    if not 1 <= person <= n_people:
        print(f"  ✗ Person must be between 1 and {n_people}.")
        return

    planner = load_planner(planner_path, n_people)
    planner = ensure_planner_row(planner, name, n_people)

    col = f"person_{person}_rating"
    planner.loc[planner["name"] == name, col] = score
    save_planner(planner, planner_path)
    print(f"  ✓ Set person_{person}_rating for '{name}' to {score}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a weekly meal plan.")
    parser.add_argument("n", nargs="?", type=int, help="Number of meals (overrides config)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--preset", nargs="+", metavar="MEAL", help="Pre-chosen meals to include")
    parser.add_argument("--debug", action="store_true", help="Show detailed scoring output")
    parser.add_argument("--made", metavar="MEAL", help="Mark a meal as cooked today")
    parser.add_argument("--made-date", metavar="YYYY-MM-DD", help="Date for --made (default: today)")
    parser.add_argument("--rate", metavar="MEAL", help="Rate a meal")
    parser.add_argument("--person", type=int, metavar="N", help="Person number for --rate")
    parser.add_argument("--score", type=float, metavar="1-5", help="Rating score for --rate")
    args = parser.parse_args()

    if not args.config.exists():
        print(f"✗ Config not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    mode = config.get("mode", "local")

    # --- Subcommands ---

    if args.made:
        date = datetime.date.fromisoformat(args.made_date) if args.made_date else None
        mark_made(args.made, config, date)
        return

    if args.rate:
        if args.person is None or args.score is None:
            print("✗ --rate requires --person N and --score VALUE")
            sys.exit(1)
        update_rating(args.rate, args.person, args.score, config)
        return

    # --- Generate plan ---

    print(f"\nMode:    {mode}")
    print(f"Season:  {current_season()}")
    print(f"Meals:   {args.n or config.get('meals_per_week', 7)}\n")

    df = load_and_merge(config)

    meals = get_meals(
        df,
        config,
        n=args.n,
        preset_meals=args.preset,
        debug=args.debug,
    )

    print("This week's meals:")
    for i, meal in enumerate(meals, 1):
        row = df[df["name"] == meal].iloc[0]
        cuisine = row.get("tags_cuisine", "")
        season  = row.get("tags_season", "")
        ease    = row.get("score_ease", "")
        ease_str = f"ease={float(ease):.2f}" if ease != "" else ""
        print(f"  {i}. {meal}")
        print(f"     {cuisine}  |  {season}  |  {ease_str}")

    print()


if __name__ == "__main__":
    main()