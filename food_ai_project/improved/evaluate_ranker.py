import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

package = joblib.load("food_ranker_model.joblib")

model = package["model"]
columns = package["columns"]
food_stats = package["food_stats"]
weather_food_counts = package["weather_food_counts"]
time_food_counts = package["time_food_counts"]
condition_food_counts = package["condition_food_counts"]
weather_totals = package["weather_totals"]
time_totals = package["time_totals"]
condition_totals = package["condition_totals"]
all_foods = package["all_foods"]
df = package["raw_data"]

def get_feature_row(weather, day_time, budget, candidate_food):
    row = {"weather": weather, "day_time": day_time, "budget": budget, "food_item": candidate_food}

    food_row = food_stats[food_stats["food_item"] == candidate_food].iloc[0]
    avg_price = food_row["avg_price"]
    total_count = food_row["total_count"]

    wf = weather_food_counts[
        (weather_food_counts["weather"] == weather) &
        (weather_food_counts["food_item"] == candidate_food)
    ]
    tf = time_food_counts[
        (time_food_counts["day_time"] == day_time) &
        (time_food_counts["food_item"] == candidate_food)
    ]
    cf = condition_food_counts[
        (condition_food_counts["weather"] == weather) &
        (condition_food_counts["day_time"] == day_time) &
        (condition_food_counts["food_item"] == candidate_food)
    ]

    wt = weather_totals[weather_totals["weather"] == weather]
    tt = time_totals[time_totals["day_time"] == day_time]
    ct = condition_totals[
        (condition_totals["weather"] == weather) &
        (condition_totals["day_time"] == day_time)
    ]

    weather_food_count = wf["weather_food_count"].iloc[0] if not wf.empty else 0
    time_food_count = tf["time_food_count"].iloc[0] if not tf.empty else 0
    condition_food_count = cf["condition_food_count"].iloc[0] if not cf.empty else 0

    weather_total = wt["weather_total"].iloc[0] if not wt.empty else 1
    time_total = tt["time_total"].iloc[0] if not tt.empty else 1
    condition_total = ct["condition_total"].iloc[0] if not ct.empty else 1

    row["avg_price"] = avg_price
    row["price_diff"] = abs(avg_price - budget)
    row["price_ratio"] = avg_price / budget if budget > 0 else 0
    row["total_count"] = total_count
    row["weather_food_count"] = weather_food_count
    row["time_food_count"] = time_food_count
    row["condition_food_count"] = condition_food_count
    row["weather_match_rate"] = weather_food_count / weather_total
    row["time_match_rate"] = time_food_count / time_total
    row["condition_match_rate"] = condition_food_count / condition_total

    return row

def rank_foods(weather, day_time, budget, top_n=5):
    rows = []
    for food in all_foods:
        rows.append(get_feature_row(weather, day_time, budget, food))

    candidates = pd.DataFrame(rows)
    candidates = pd.get_dummies(candidates, columns=["weather", "day_time", "food_item"])

    for col in columns:
        if col not in candidates.columns:
            candidates[col] = 0

    candidates = candidates[columns]
    candidates["score"] = model.predict_proba(candidates)[:, 1]

    ranked = candidates.copy()
    ranked["food_item_name"] = all_foods
    ranked = ranked.sort_values("score", ascending=False)

    return ranked["food_item_name"].head(top_n).tolist()

# Use a test split from raw data
_, test_df = train_test_split(df, test_size=0.2, random_state=42)

hit1 = 0
hit3 = 0
hit5 = 0
total = len(test_df)

for _, row in test_df.iterrows():
    actual = row["food_item"]
    rec1 = rank_foods(row["weather"], row["day_time"], row["price"], top_n=1)
    rec3 = rank_foods(row["weather"], row["day_time"], row["price"], top_n=3)
    rec5 = rank_foods(row["weather"], row["day_time"], row["price"], top_n=5)

    if actual in rec1:
        hit1 += 1
    if actual in rec3:
        hit3 += 1
    if actual in rec5:
        hit5 += 1

print("\n===== TRUE ML RANKER PERFORMANCE =====")
print(f"Hit Rate @1: {hit1 / total:.4f}")
print(f"Hit Rate @3: {hit3 / total:.4f}")
print(f"Hit Rate @5: {hit5 / total:.4f}")