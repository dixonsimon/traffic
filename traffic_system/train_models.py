"""Train and compare four classifiers that predict congestion (Low / Medium / High).

Features: road attributes, hour, day, weather, road condition, temperature,
visibility, plus *historical* volume for the same road/hour/day and the
previous hour (computed from the training split only, so there is no leakage).
"""
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine

from config import DATABASE_URL, LEVELS, METRICS_PATH, MODEL_DIR, MODEL_PATH

CAT = ["weather", "road_condition", "road_type"]
NUM = ["road_id", "lanes", "speed_limit", "capacity", "hour", "day_of_week", "is_weekend",
       "temperature", "visibility_km", "hist_volume", "hist_prev_volume", "hist_ratio"]
FEATURES = CAT + NUM


def load():
    eng = create_engine(DATABASE_URL)
    q = """SELECT t.*, r.road_type, r.lanes, r.speed_limit, r.capacity
           FROM traffic_records t JOIN roads r ON r.road_id = t.road_id"""
    return pd.read_sql(q, eng)


def build_history(train_df):
    """Mean volume per (road, day_of_week, hour) from the training rows."""
    h = train_df.groupby(["road_id", "day_of_week", "hour"]).traffic_volume.mean().rename("hist_volume").reset_index()
    return h


def add_history(df, hist):
    out = df.merge(hist, on=["road_id", "day_of_week", "hour"], how="left")
    prev = hist.copy()
    prev["hour"] = (prev["hour"] + 1) % 24          # volume of previous hour, shifted onto this hour
    prev = prev.rename(columns={"hist_volume": "hist_prev_volume"})
    out = out.merge(prev, on=["road_id", "day_of_week", "hour"], how="left")
    out["hist_volume"] = out["hist_volume"].fillna(out["hist_volume"].mean())
    out["hist_prev_volume"] = out["hist_prev_volume"].fillna(out["hist_volume"])
    out["hist_ratio"] = out["hist_volume"] / out["capacity"]
    return out


def make_pipe(model, scale):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler() if scale else "passthrough", NUM),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load()
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["congestion"])
    hist = build_history(train)
    train, test = add_history(train, hist), add_history(test, hist)
    Xtr, ytr, Xte, yte = train[FEATURES], train["congestion"], test[FEATURES], test["congestion"]

    models = {
        "Logistic Regression": make_pipe(LogisticRegression(max_iter=2000, C=2.0), True),
        "Decision Tree": make_pipe(DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42), False),
        "Random Forest": make_pipe(RandomForestClassifier(n_estimators=200, max_depth=16, n_jobs=-1, random_state=42), False),
        "K-Nearest Neighbors": make_pipe(KNeighborsClassifier(n_neighbors=15, weights="distance"), True),
    }

    results, fitted = {}, {}
    for name, pipe in models.items():
        t0 = time.time()
        pipe.fit(Xtr, ytr)
        fit_s = time.time() - t0
        pred = pipe.predict(Xte)
        results[name] = {
            "accuracy": accuracy_score(yte, pred),
            "precision": precision_score(yte, pred, average="weighted", zero_division=0),
            "recall": recall_score(yte, pred, average="weighted", zero_division=0),
            "f1": f1_score(yte, pred, average="weighted"),
            "f1_high": f1_score(yte, pred, labels=["High"], average="macro"),
            "train_seconds": fit_s,
            "confusion": confusion_matrix(yte, pred, labels=LEVELS).tolist(),
        }
        fitted[name] = pipe
        print(f"{name:22s} acc={results[name]['accuracy']:.3f}  f1={results[name]['f1']:.3f}  ({fit_s:.1f}s)")

    best = max(results, key=lambda n: results[n]["f1"])
    rf = fitted["Random Forest"]
    names = rf.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False).head(10)

    joblib.dump({"model": fitted[best], "name": best, "features": FEATURES, "hist": hist}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best": best, "results": results, "levels": LEVELS,
                   "n_train": len(train), "n_test": len(test),
                   "importances": imp.round(4).to_dict()}, f, indent=2)
    print(f"\nBest model: {best}  -> saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
