import os
import re
import joblib
import pandas as pd
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

package = joblib.load("food_ranker_model.joblib")

model      = package["model"]
columns    = package["columns"]
food_stats = package["food_stats"]
all_foods  = package["all_foods"]


# ── Affinity tables ──────────────────────────────────────────────────────────
#
# Each nested dict maps a food category → affinity score (0.1 – 1.0).
# 1.0  = perfect match for this weather / time
# 0.5  = neutral / no strong preference
# 0.1  = poor match (will rarely surface unless nothing else fits)

WEATHER_AFFINITY = {
    "Rainy": {
        "Soup":        1.0,
        "Rice Meal":   0.9,
        "Beverage":    0.8,   # hot drinks
        "Main Course": 0.7,
        "Fast Food":   0.6,
        "Breakfast":   0.5,
        "Dessert":     0.4,
    },
    "Cold": {
        "Soup":        1.0,
        "Rice Meal":   0.9,
        "Main Course": 0.8,
        "Beverage":    0.7,
        "Breakfast":   0.7,
        "Fast Food":   0.6,
        "Dessert":     0.5,
    },
    "Hot": {
        "Beverage":    1.0,   # cold drinks / smoothies / lassi
        "Dessert":     0.9,   # ice cream, faluda
        "Fast Food":   0.6,
        "Breakfast":   0.5,
        "Rice Meal":   0.5,
        "Main Course": 0.4,
        "Soup":        0.1,
    },
    "Sunny": {
        "Fast Food":   0.9,
        "Beverage":    0.8,
        "Dessert":     0.8,
        "Breakfast":   0.7,
        "Rice Meal":   0.6,
        "Main Course": 0.6,
        "Soup":        0.3,
    },
    "Cloudy": {
        "Main Course": 0.8,
        "Rice Meal":   0.8,
        "Fast Food":   0.7,
        "Soup":        0.7,
        "Breakfast":   0.6,
        "Beverage":    0.6,
        "Dessert":     0.5,
    },
}

TIME_AFFINITY = {
    "Morning": {
        "Breakfast":   1.0,
        "Beverage":    0.9,
        "Dessert":     0.4,
        "Soup":        0.4,
        "Fast Food":   0.3,
        "Rice Meal":   0.3,
        "Main Course": 0.2,
    },
    "Breakfast": {   # alias
        "Breakfast":   1.0,
        "Beverage":    0.9,
        "Dessert":     0.4,
        "Soup":        0.4,
        "Fast Food":   0.3,
        "Rice Meal":   0.3,
        "Main Course": 0.2,
    },
    "Lunch": {
        "Rice Meal":   1.0,
        "Main Course": 0.9,
        "Fast Food":   0.8,
        "Soup":        0.7,
        "Beverage":    0.6,
        "Dessert":     0.5,
        "Breakfast":   0.3,
    },
    "Afternoon": {
        "Beverage":    1.0,
        "Dessert":     0.9,
        "Fast Food":   0.8,
        "Soup":        0.4,
        "Rice Meal":   0.5,
        "Main Course": 0.4,
        "Breakfast":   0.4,
    },
    "Snack": {       # alias
        "Beverage":    1.0,
        "Dessert":     0.9,
        "Fast Food":   0.8,
        "Soup":        0.4,
        "Rice Meal":   0.5,
        "Main Course": 0.4,
        "Breakfast":   0.4,
    },
    "Dinner": {
        "Rice Meal":   1.0,
        "Main Course": 1.0,
        "Soup":        0.8,
        "Fast Food":   0.7,
        "Dessert":     0.6,
        "Beverage":    0.5,
        "Breakfast":   0.1,
    },
    "Evening": {     # alias
        "Rice Meal":   1.0,
        "Main Course": 1.0,
        "Soup":        0.8,
        "Fast Food":   0.7,
        "Dessert":     0.6,
        "Beverage":    0.5,
        "Breakfast":   0.1,
    },
    "Night": {
        "Fast Food":   0.9,
        "Rice Meal":   0.8,
        "Main Course": 0.8,
        "Dessert":     0.7,
        "Soup":        0.7,
        "Beverage":    0.4,
        "Breakfast":   0.2,
    },
}

# Weights must sum to 1.0
WEATHER_WEIGHT = 0.40
TIME_WEIGHT    = 0.35
PRICE_WEIGHT   = 0.25

DEFAULT_AFFINITY = 0.5   # fallback when a key is missing from the table


# ── Helper functions ─────────────────────────────────────────────────────────

def get_category(food_name):
    name = str(food_name).lower()

    if any(x in name for x in ["coffee", "tea", "juice", "milkshake", "shake",
                                 "lassi", "lemonade"]):
        return "Beverage"

    if any(x in name for x in ["cake", "ice cream", "brownie", "cheesecake",
                                 "donut", "muffin", "faluda"]):
        return "Dessert"

    if "soup" in name:
        return "Soup"

    if any(x in name for x in ["biryani", "rice", "khichuri", "khichdi",
                                 "tehari", "pulao"]):
        return "Rice Meal"

    if any(x in name for x in ["pizza", "burger", "sandwich", "shawarma",
                                 "wrap", "fries"]):
        return "Fast Food"

    if any(x in name for x in ["paratha", "toast", "egg", "omelette",
                                 "pancake", "waffle", "porridge", "croissant"]):
        return "Breakfast"

    return "Main Course"


def get_weather_score(category: str, weather: str) -> float:
    table = WEATHER_AFFINITY.get(weather, {})
    return table.get(category, DEFAULT_AFFINITY)


def get_time_score(category: str, day_time: str) -> float:
    table = TIME_AFFINITY.get(day_time, {})
    return table.get(category, DEFAULT_AFFINITY)


def get_best_for_today(score):
    if score >= 0.80:
        return "Best for today"
    if score >= 0.60:
        return "Popular today"
    return "Worth trying"


def safe_filename(food_name):
    name = str(food_name).lower()
    name = name.replace("&", "and")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return f"{name}.jpg"


def generate_food_image(food_name):
    filename = safe_filename(food_name)
    filepath = os.path.join("food_images", filename)
    if os.path.exists(filepath):
        return f"/food_images/{filename}"
    return "/food_images/default_food.jpg"


def reverse_geocode(lat, lon):
    try:
        url = (
            "https://api.bigdatacloud.net/data/reverse-geocode-client"
            f"?latitude={lat}&longitude={lon}&localityLanguage=en"
        )
        data = requests.get(url, timeout=10).json()

        city     = data.get("city") or data.get("locality") or ""
        district = data.get("principalSubdivision") or ""
        country  = data.get("countryName") or ""

        parts = []
        for item in [city, district, country]:
            if item and item not in parts:
                parts.append(item)

        return ", ".join(parts) if parts else "Unknown Location"

    except Exception:
        return "Unknown Location"


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return send_from_directory(".", "website_demo.html")


@app.route("/food_images/<path:filename>")
def food_images(filename):
    return send_from_directory("food_images", filename)


@app.route("/current-weather", methods=["POST"])
def current_weather():
    data = request.get_json()

    lat = data.get("lat")
    lon = data.get("lon")

    if lat is None or lon is None:
        return jsonify({"error": "Latitude and longitude are required"}), 400

    try:
        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
        )
        weather_data = requests.get(weather_url, timeout=10).json()
        current = weather_data.get("current_weather", {})

        temp = current.get("temperature")
        code = current.get("weathercode")

        location = reverse_geocode(lat, lon)

        def map_weather(weather_code, temperature):
            if weather_code in [51, 53, 55, 61, 63, 65, 80, 81, 82]:
                return "Rainy"
            if weather_code in [1, 2, 3, 45, 48]:
                return "Cloudy"
            if temperature is not None and temperature < 15:
                return "Cold"
            if temperature is not None and temperature > 32:
                return "Hot"
            return "Sunny"

        label = map_weather(code, temp)

        return jsonify({
            "city":           location,
            "temp_c":         temp,
            "main_condition": label,
            "weather_label":  label,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    weather  = data.get("weather", "Sunny")
    day_time = data.get("day_time", "Lunch")
    price    = data.get("price")
    top_n    = int(data.get("top_n", 5))

    try:
        budget = float(price)
    except Exception:
        return jsonify({"error": "Invalid budget"}), 400

    # Build candidate list
    rows = []
    for food in all_foods:
        match = food_stats[food_stats["food_item"] == food]
        if match.empty:
            continue
        f = match.iloc[0]
        rows.append({
            "food_item": food,
            "avg_price": float(f["avg_price"]),
            "min_price": float(f["min_price"]),
            "max_price": float(f["max_price"]),
        })

    df = pd.DataFrame(rows)

    # Keep only foods that have at least one price option within budget
    df = df[df["min_price"] <= budget]

    if df.empty:
        return jsonify({
            "recommendations": [],
            "message": "No food available at this budget",
        })

    # ── Three-signal weighted score ──────────────────────────────────────────
    def score_row(row):
        category = get_category(row["food_item"])

        w_score = get_weather_score(category, weather)
        t_score = get_time_score(category, day_time)

        # Price score: peaks at 1.0 when avg_price == budget, falls off smoothly
        p_score = 1 / (abs(row["avg_price"] - budget) + 1)

        return (
            WEATHER_WEIGHT * w_score
            + TIME_WEIGHT  * t_score
            + PRICE_WEIGHT * p_score
        )

    df["score"] = df.apply(score_row, axis=1)
    df = df.sort_values("score", ascending=False).head(top_n)

    recommendations = []
    for _, row in df.iterrows():
        shown_price = min(float(row["avg_price"]), budget)
        score       = float(row["score"])

        recommendations.append({
            "food_item":      row["food_item"],
            "score":          round(score, 4),
            "category":       get_category(row["food_item"]),
            "price":          round(shown_price, 2),
            "price_range":    f"{int(row['min_price'])}-{int(row['max_price'])}",
            "best_for_today": get_best_for_today(score),
            "food_image":     generate_food_image(row["food_item"]),
        })

    return jsonify({"recommendations": recommendations})


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)