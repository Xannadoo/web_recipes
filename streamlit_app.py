"""
streamlit_app.py
=================
Public-facing web interface for the meal planner.

This deliberately runs in "web mode" only — no personal ratings, no
last_made history, no writes to planner.csv. It's meant to be a
self-contained plan generator anyone can use from recipes_processed.csv,
regardless of what your local config.yaml's `mode:` setting says.

Run locally:
    streamlit run streamlit_app.py

Dependencies:
    pip install streamlit pandas numpy pyyaml
"""

import numpy as np
import streamlit as st

from meal_planner import (
    DEFAULT_CONFIG, load_config, load_and_merge, get_meals, current_season,
    split_pipe,
)

st.set_page_config(page_title="Meal Planner", page_icon="🍽️", layout="centered")


# ---------------------------------------------------------------------------
# Cached data loading (the underlying CSV rarely changes within a session)
# ---------------------------------------------------------------------------

@st.cache_data
def get_config():
    if not DEFAULT_CONFIG.exists():
        return None
    return load_config(DEFAULT_CONFIG)


@st.cache_data
def get_all_recipes(_config, diet_key: str):
    """
    _config is prefixed with underscore so Streamlit doesn't try to hash it
    (dicts from YAML are hashable-ish but this avoids edge cases).
    diet_key is a plain string used purely as a cache key for the filter.
    """
    diet_types = diet_key.split(",") if diet_key else None
    return load_and_merge(_config, diet_types=diet_types, use_planner=False)


def ease_label(score: float) -> str:
    if score >= 0.7:
        return "🟢 Easy"
    elif score >= 0.4:
        return "🟡 Medium"
    else:
        return "🔴 Involved"


def split_display(value) -> list:
    """
    Like split_pipe but preserves original casing — used for ingredients
    and method text, which shouldn't be lowercased for display (split_pipe
    is designed for tag comparison, not readable prose).
    """
    if value is None or (isinstance(value, float) and value != value):  # NaN check
        return []
    return [s.strip() for s in str(value).split("|") if s.strip()]


def coalesce(*values):
    """Return the first value that isn't None/NaN/empty — 'or' breaks on
    pandas NaN floats since they're truthy in plain Python."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and v != v:  # NaN
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return ""


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.title("🍽️ Weekly Meal Planner")
st.caption(f"Season: {current_season()}")

config = get_config()
if config is None:
    st.error(
        f"Could not find `{DEFAULT_CONFIG}`. Make sure config.yaml is in the "
        "same folder as this app."
    )
    st.stop()

DIET_OPTIONS = config.get("diet_types", ["meat", "fish", "vegetarian", "vegan"])

with st.sidebar:
    st.header("Preferences")

    n_meals = st.slider("Number of meals", min_value=1, max_value=14, value=7)

    selected_diets = st.multiselect(
        "Diet type",
        options=DIET_OPTIONS,
        default=DIET_OPTIONS,
        help="Pick one or more. Recipes matching any selected type are included.",
    )

    generate_clicked = st.button("Generate meal plan", type="primary", use_container_width=True)

if not selected_diets:
    st.warning("Select at least one diet type in the sidebar to generate a plan.")
    st.stop()

# Load (cached) — only re-filters when the diet selection actually changes
diet_key = ",".join(sorted(selected_diets))
try:
    df = get_all_recipes(config, diet_key)
except SystemExit:
    st.error(
        "No recipes matched your filters, or recipes_processed.csv is missing. "
        "Run recipe_processor.py first."
    )
    st.stop()

if len(df) < n_meals:
    st.info(
        f"Only {len(df)} recipe(s) match your filters — showing all of them "
        f"instead of {n_meals}."
    )

# Regenerating just needs a fresh call; Streamlit reruns the whole script on
# each interaction, and a fresh default_rng() gives new randomness each time.
if generate_clicked or "meals" not in st.session_state:
    rng = np.random.default_rng()
    st.session_state["meals"] = get_meals(df, config, n=n_meals, rng=rng)

meals = st.session_state.get("meals", [])

if not meals:
    st.info("Click **Generate meal plan** in the sidebar to get started.")
    st.stop()

st.subheader("This week's meals")

for i, name in enumerate(meals, 1):
    row = df[df["name"] == name].iloc[0]

    cuisine = split_pipe(row.get("tags_cuisine", ""))
    season_tags = split_pipe(row.get("tags_season", ""))
    ease = ease_label(float(row.get("score_ease", 0.5)))
    servings = coalesce(row.get("servings_scaled"), row.get("servings_original"))
    prep = coalesce(row.get("prep_time"))
    cook = coalesce(row.get("cook_time"))
    total = coalesce(row.get("total_time"))

    with st.expander(f"**{i}. {name}**", expanded=False):
        badge_line = " · ".join(
            filter(None, [
                ", ".join(t.title() for t in cuisine) if cuisine else "",
                ", ".join(t.title() for t in season_tags) if season_tags else "",
                ease,
            ])
        )
        if badge_line:
            st.caption(badge_line)

        meta_cols = st.columns(3)
        meta_cols[0].metric("Servings", servings or "—")
        if prep or cook:
            meta_cols[1].metric("Prep", prep or "—")
            meta_cols[2].metric("Cook", cook or "—")
        else:
            meta_cols[1].metric("Total time", total or "—")

        ingredients = split_display(coalesce(row.get("ingredients_scaled"), row.get("ingredients_original")))
        if ingredients:
            st.markdown("**Ingredients**")
            st.markdown("\n".join(f"- {ing}" for ing in ingredients))

        method_steps = split_display(row.get("method"))
        if method_steps:
            st.markdown("**Method**")
            st.markdown("\n".join(f"{n}. {step}" for n, step in enumerate(method_steps, 1)))

        source_url = coalesce(row.get("source_url"))
        if source_url:
            st.caption(f"Source: {source_url}")

st.button("🔄 Regenerate", on_click=lambda: st.session_state.pop("meals", None))