# Meal planner and recipe scrape

Project to create a meal planner for my family.

1. Recipe Database
    - Extracted from websites using Selenium and BeautifulSoup.
    - Manual entry
    - Data tagged using a locally hosted LLM via Ollama.

2. Create a meal plan
    - Planned variety in terms of ingredients etc
    - Family favourites come up more often, different family members can provide their own preferences
    - Time since last eaten taken into account
    - Easier dishes preferred over complex
    - Seasonality taken into account so in-season meals prefered
    - Allow user selected meals, taken into account when generating a list too.

3. Create webpage (local hosted?)
    - Allow for easy customisation of recipes etc through an easy interface
    - Pretty generation of a meal plan
    - Local hosted allows customisation, so can be tailored to specific needs
    - Demo on website without deep customisation (no changes to database)

4. Extensions
    - Shopping list generation
    - Weather predictions
    - ...

## Usage:

### Scrape:
```
python recipe_scraper.py [URL]
```

### Process:
```
python recipe_processor.py
```

### Plan:

```
python meal_planner.py
```
```
python meal_planner.py 5                        # 5 meals instead of 7
```
```
python meal_planner.py --preset "Pasta Bake"   # lock in one meal, fill the rest
```
```
python meal_planner.py --debug                 # see scoring breakdown
```

### After cooking
```
python meal_planner.py --made "Veggie Chilli"
```
```
python meal_planner.py --made "Pasta Bake" --made-date 2025-06-10
```
### Rating (local mode)
```
python meal_planner.py --rate "Veggie Chilli" --person 1 --score 5
```
```
python meal_planner.py --rate "Veggie Chilli" --person 2 --score 3
```