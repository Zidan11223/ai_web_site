import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv("restaurant_dataset_realistic_5000.csv")
df = df.dropna()
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna()

# Clean text
df["weather"] = df["weather"].astype(str).str.strip()
df["day_time"] = df["day_time"].astype(str).str.strip()
df["food_item"] = df["food_item"].astype(str).str.strip()

# Food stats
food_stats = df.groupby("food_item").agg(
    avg_price=("price", "mean"),
    min_price=("price", "min"),
    max_price=("price", "max"),
    total_count=("food_item", "count")
).reset_index()

weather_food_counts = df.groupby(["weather", "food_item"]).size().reset_index(name="weather_food_count")
time_food_counts = df.groupby(["day_time", "food_item"]).size().reset_index(name="time_food_count")
condition_food_counts = df.groupby(["weather", "day_time", "food_item"]).size().reset_index(name="condition_food_count")

weather_totals = df.groupby("weather").size().reset_index(name="weather_total")
time_totals = df.groupby("day_time").size().reset_index(name="time_total")
condition_totals = df.groupby(["weather", "day_time"]).size().reset_index(name="condition_total")

all_foods = food_stats["food_item"].tolist()

def get_feature_row(weather, day_time, budget, candidate_food):
    row = {"weather": weather, "day_time": day_time, "budget": budget, "food_item": candidate_food}

    food_row = food_stats[food_stats["food_item"] == candidate_food].iloc[0]
    avg_price = food_row["avg_price"]
    total_count = food_row["total_count"]

    # Counts
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

# Build training pairs
training_rows = []

for _, original in df.iterrows():
    weather = original["weather"]
    day_time = original["day_time"]
    budget = original["price"]
    actual_food = original["food_item"]

    # Positive example
    pos_row = get_feature_row(weather, day_time, budget, actual_food)
    pos_row["label"] = 1
    training_rows.append(pos_row)

    # Negative samples
    negative_foods = [f for f in all_foods if f != actual_food]
    sampled_negatives = np.random.choice(
        negative_foods,
        size=min(10, len(negative_foods)),
        replace=False
    )

    for neg_food in sampled_negatives:
        neg_row = get_feature_row(weather, day_time, budget, neg_food)
        neg_row["label"] = 0
        training_rows.append(neg_row)

rank_df = pd.DataFrame(training_rows)

# One-hot encode weather, day_time, food_item
rank_df = pd.get_dummies(rank_df, columns=["weather", "day_time", "food_item"])

X = rank_df.drop(columns=["label"])
y = rank_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

print(f"Ranking model ROC-AUC: {auc:.4f}")

package = {
    "model": model,
    "columns": X.columns.tolist(),
    "food_stats": food_stats,
    "weather_food_counts": weather_food_counts,
    "time_food_counts": time_food_counts,
    "condition_food_counts": condition_food_counts,
    "weather_totals": weather_totals,
    "time_totals": time_totals,
    "condition_totals": condition_totals,
    "all_foods": all_foods,
    "raw_data": df
}

joblib.dump(package, "food_ranker_model.joblib")
print("Saved as food_ranker_model.joblib")