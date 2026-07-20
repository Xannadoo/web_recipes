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

import difflib

import numpy as np
import pandas as pd
import streamlit as st

from meal_planner import (
    DEFAULT_CONFIG, load_config, load_and_merge, get_meals, current_season,
    split_pipe, select_with_tag_penalty,
)

st.set_page_config(page_title="Meal Planner", page_icon="🍽️", layout="centered")

# Ease score below this is "Involved" (red) — see ease_label(). Easy Week
# excludes these entirely rather than just down-weighting them.
EASY_WEEK_MIN_EASE = 0.4
# Additive boost applied to score_ease during Easy Week, so genuinely quick
# recipes (ease ~0.7+) pull further ahead of merely-medium ones (~0.4-0.7)
# rather than just barely edging them out.
EASY_WEEK_BOOST = 0.4


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
    Passing diet_key="" returns every main recipe, unfiltered — used as the
    lookup table for rendering/search regardless of the active diet filter.
    """
    diet_types = diet_key.split(",") if diet_key else None
    return load_and_merge(_config, diet_types=diet_types, use_planner=False)


def ease_label(score: float) -> str:
    if score >= 0.7:
        return "🟢 Easy"
    elif score >= EASY_WEEK_MIN_EASE:
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


def diet_matches(row, selected_diets: list) -> bool:
    if not selected_diets:
        return True
    tags = split_pipe(row.get("tags_diet", ""))
    allowed = {d.lower() for d in selected_diets}
    return bool(set(tags) & allowed)


def fuzzy_search(query: str, names: list, limit: int = 6) -> list:
    """Substring matches first, then fuzzy ratio for anything close enough —
    recipe names aren't always remembered exactly."""
    q = query.lower().strip()
    if not q:
        return []
    scored = []
    for name in names:
        nl = name.lower()
        if q in nl:
            score = 1.0 + (len(q) / len(nl))
        else:
            score = difflib.SequenceMatcher(None, q, nl).ratio()
            if score < 0.45:
                continue
        scored.append((score, name))
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, name in scored:
        if name not in seen:
            seen.add(name)
            out.append(name)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# Plan generation / regeneration
# ---------------------------------------------------------------------------

def ease_tier(score_ease: float) -> str:
    if score_ease >= 0.7:
        return "easy"
    elif score_ease >= EASY_WEEK_MIN_EASE:
        return "medium"
    return "involved"


def get_meals_easy_week(
    pool: pd.DataFrame, config: dict, n: int, kept: list, medium_cap: int, rng,
) -> list:
    """
    Fill a plan under Easy Week rules. 'Involved' recipes are already
    excluded from `pool` upstream (see build_pool). This additionally caps
    how many freshly-picked 'Medium' recipes (score_ease 0.4-0.7) can make
    it in — once `medium_cap` mediums have been chosen, only 'Easy'
    (score_ease >= 0.7) recipes remain eligible for the rest of the plan.
    A plain score boost alone doesn't guarantee this — with enough medium
    candidates scoring well on season/rating, chance can still let several
    through, which is what was happening before.

    Kept meals count toward the cap if they happen to be Medium, so a
    ticked-medium plus 2 freshly-picked mediums won't sneak past the limit.
    """
    tag_penalty = config.get("tag_penalty", 0.4)
    jitter_strength = config.get("jitter_strength", 0.05)
    tag_fields = {
        "key_ingredients": "tags_key_ingredients",
        "cuisine":         "tags_cuisine",
        "season":          "tags_season",
    }

    valid_names = set(pool["name"])
    selected = [m for m in kept if m in valid_names]
    n = min(n, len(pool))
    if len(selected) >= n:
        return selected[:n]

    selected_tags = {cat: [] for cat in tag_fields}
    medium_used = 0
    for meal in selected:
        row = pool[pool["name"] == meal].iloc[0]
        for cat, col in tag_fields.items():
            selected_tags[cat].extend(split_pipe(row.get(col, "")))
        if ease_tier(float(row.get("score_ease", 0.5))) == "medium":
            medium_used += 1

    remaining = pool[~pool["name"].isin(selected)].copy()

    for _ in range(n - len(selected)):
        if remaining.empty:
            break

        if medium_used < medium_cap:
            candidates = remaining
        else:
            candidates = remaining[remaining["score_ease"] >= 0.7]
            if candidates.empty:
                candidates = remaining  # fall back rather than leaving a slot unfilled

        chosen, _, _ = select_with_tag_penalty(
            candidates, selected_tags, tag_penalty, jitter_strength, rng
        )
        selected.append(chosen)

        chosen_row = remaining[remaining["name"] == chosen].iloc[0]
        if ease_tier(float(chosen_row.get("score_ease", 0.5))) == "medium":
            medium_used += 1

        for cat, col in tag_fields.items():
            selected_tags[cat].extend(split_pipe(chosen_row.get(col, "")))

        remaining = remaining[remaining["name"] != chosen]

    return selected


def build_pool(kept_names: list, filtered_df: pd.DataFrame, full_df: pd.DataFrame, easy_week: bool) -> pd.DataFrame:
    """
    The candidate pool for a (re)roll: diet-filtered recipes, optionally
    restricted + boosted for Easy Week, PLUS full rows for any explicitly
    kept meal that got excluded by those filters — kept meals are honoured
    regardless of filter mismatch, per user's explicit checkbox choice.
    """
    pool = filtered_df.copy()

    if easy_week:
        pool = pool[pool["score_ease"] >= EASY_WEEK_MIN_EASE].copy()
        pool["score"] = pool["score"] + EASY_WEEK_BOOST * pool["score_ease"]

    pool_names = set(pool["name"])
    missing = [n for n in kept_names if n not in pool_names]
    if missing:
        extra = full_df[full_df["name"].isin(missing)].copy()
        if easy_week:
            extra["score"] = extra["score"] + EASY_WEEK_BOOST * extra["score_ease"]
        pool = pd.concat([pool, extra], ignore_index=True)

    return pool


def regenerate() -> None:
    """Reroll every meal in the plan whose 'Keep' checkbox isn't ticked,
    keeping the ticked ones untouched. Used by the Generate/Regenerate
    button and as the on_change handler for diet + Easy Week controls."""
    config = get_config()
    if config is None:
        return

    selected_diets = st.session_state.get("diet_select", [])
    if not selected_diets:
        return  # nothing to plan against; main body shows a warning

    diet_key = ",".join(sorted(selected_diets))
    full_df = get_all_recipes(config, "")
    filtered_df = get_all_recipes(config, diet_key)

    plan = st.session_state.get("plan", [])
    kept = [n for n in plan if st.session_state.get(f"keep__{n}", False)]

    easy_week = st.session_state.get("easy_week", False)
    pool = build_pool(kept, filtered_df, full_df, easy_week)

    if pool.empty:
        return

    n_meals = st.session_state.get("n_meals", 7)
    rng = np.random.default_rng()

    if easy_week:
        # Cap scales with plan size — "1-2 out of 7" is roughly 2/7 of the
        # week, rounded, with at least 1 allowed so small plans aren't
        # forced to be 100% Easy.
        medium_cap = max(1, round(n_meals * 2 / 7))
        st.session_state["plan"] = get_meals_easy_week(
            pool, config, n=n_meals, kept=kept, medium_cap=medium_cap, rng=rng
        )
    else:
        st.session_state["plan"] = get_meals(pool, config, n=n_meals, preset_meals=kept, rng=rng)


def add_manual_recipe(name: str) -> None:
    plan = st.session_state.get("plan", [])
    if name not in plan:
        st.session_state["plan"] = plan + [name]
    st.session_state[f"keep__{name}"] = True


def remove_from_plan(name: str) -> None:
    st.session_state["plan"] = [n for n in st.session_state.get("plan", []) if n != name]


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
full_df = get_all_recipes(config, "")

with st.sidebar:
    st.header("Preferences")

    st.slider("Number of meals", min_value=1, max_value=14, value=7, key="n_meals")

    st.multiselect(
        "Diet type",
        options=DIET_OPTIONS,
        default=DIET_OPTIONS,
        help="Pick one or more. Recipes matching any selected type are included. "
             "Changing this automatically rerolls anything not ticked 'Keep'.",
        key="diet_select",
        on_change=regenerate,
    )

    st.checkbox(
        "🍃 Easy week (favour quick, low-effort meals)",
        value=False,
        key="easy_week",
        help="Excludes 'Involved' recipes entirely and favours 'Easy' over "
             "'Medium' among what's left.",
        on_change=regenerate,
    )

    st.divider()
    st.subheader("Add a specific recipe")
    query = st.text_input("Search recipes", key="manual_search", placeholder="e.g. bolognese")
    if query.strip():
        matches = fuzzy_search(query, full_df["name"].tolist())
        current_plan = st.session_state.get("plan", [])
        if not matches:
            st.caption("No close matches found.")
        for m in matches:
            c1, c2 = st.columns([4, 1])
            c1.write(m)
            if m in current_plan:
                c2.markdown("✓")
            else:
                c2.button("Add", key=f"add__{m}", on_click=add_manual_recipe, args=(m,))

    st.divider()
    st.button(
        "🔄 Generate / Regenerate",
        type="primary",
        use_container_width=True,
        on_click=regenerate,
    )

if not st.session_state.get("diet_select"):
    st.warning("Select at least one diet type in the sidebar to generate a plan.")
    st.stop()

# First load: generate a full plan automatically, same as before.
if "plan" not in st.session_state:
    regenerate()

plan = st.session_state.get("plan", [])
selected_diets = st.session_state.get("diet_select", [])
easy_week = st.session_state.get("easy_week", False)

if not plan:
    st.info("Search for a recipe to add it, or click **Generate / Regenerate** in the sidebar.")
    st.stop()

st.subheader("This week's meals")

for i, name in enumerate(plan, 1):
    match = full_df[full_df["name"] == name]
    if match.empty:
        continue
    row = match.iloc[0]

    ease_score = float(row.get("score_ease", 0.5))
    ease_text = ease_label(ease_score)

    header_col, keep_col, remove_col = st.columns([0.62, 0.2, 0.18])
    header_col.markdown(f"**{i}. {name}** — Ease: {ease_text}")
    keep_col.checkbox("Keep", value=False, key=f"keep__{name}")
    remove_col.button("✕ Remove", key=f"remove__{name}", on_click=remove_from_plan, args=(name,))

    if not diet_matches(row, selected_diets):
        st.caption("⚠️ Doesn't match your current diet filter")
    if easy_week and ease_score < EASY_WEEK_MIN_EASE:
        st.caption("⚠️ Marked 'Involved' — outside this week's Easy Week filter")



    with st.expander("Details", expanded=False):
        cuisine = split_pipe(row.get("tags_cuisine", ""))
        season_tags = split_pipe(row.get("tags_season", ""))
        badge_line = " · ".join(
            filter(None, [
                ", ".join(t.title() for t in cuisine) if cuisine else "",
                ", ".join(t.title() for t in season_tags) if season_tags else "",
            ])
        )
        if badge_line:
            st.caption(badge_line)
        servings = coalesce(row.get("servings_scaled"), row.get("servings_original"))
        prep = coalesce(row.get("prep_time"))
        cook = coalesce(row.get("cook_time"))
        total = coalesce(row.get("total_time"))

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

    st.divider()