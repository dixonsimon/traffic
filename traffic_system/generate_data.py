"""Generate a realistic synthetic traffic dataset and load it into the database.

Replace this with real data any time: just fill the `roads` and
`traffic_records` tables (see schema.sql) and re-run train_models.py.
"""
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from config import DATABASE_URL, DATA_DIR
import os

RNG = np.random.default_rng(42)
DAYS_OF_DATA = 120

ROADS = [
    # id, name, area, type, lanes, speed_limit, capacity(veh/h)
    (1, "MG Road", "Central Business District", "Arterial", 4, 50, 3200),
    (2, "Anna Salai", "Central Business District", "Arterial", 6, 60, 4800),
    (3, "OMR Tech Corridor", "IT Corridor", "Highway", 6, 80, 5400),
    (4, "Mount Road Flyover", "Central Business District", "Flyover", 4, 60, 3400),
    (5, "East Coast Road", "Coastal Belt", "Highway", 4, 80, 3600),
    (6, "Station Road", "Old Town", "Local", 2, 30, 1200),
    (7, "Market Street", "Old Town", "Local", 2, 30, 1100),
    (8, "University Avenue", "Education Hub", "Collector", 4, 40, 2400),
    (9, "Airport Link Road", "Airport Zone", "Highway", 6, 80, 5200),
    (10, "Ring Road North", "Outer Ring", "Highway", 6, 80, 5600),
    (11, "Lakeview Drive", "Residential East", "Collector", 2, 40, 1500),
    (12, "Industrial Estate Road", "Industrial Zone", "Collector", 4, 50, 2600),
]

# Rush-hour sensitivity by road area (morning, evening, midday weights)
PROFILE = {
    "Central Business District": (1.0, 1.0, 0.7),
    "IT Corridor": (1.0, 1.05, 0.5),
    "Coastal Belt": (0.5, 0.8, 0.5),
    "Old Town": (0.7, 0.9, 0.95),
    "Education Hub": (1.0, 0.7, 0.6),
    "Airport Zone": (0.8, 0.9, 0.8),
    "Outer Ring": (0.9, 1.0, 0.6),
    "Residential East": (0.9, 0.8, 0.4),
    "Industrial Zone": (1.0, 0.8, 0.5),
}

WEATHER_P = {"Clear": 0.45, "Cloudy": 0.25, "Rain": 0.18, "Fog": 0.07, "Storm": 0.05}
WEATHER_FACTOR = {"Clear": 1.0, "Cloudy": 1.0, "Rain": 0.93, "Fog": 0.9, "Storm": 0.8}
COND_FACTOR = {"Dry": 1.0, "Wet": 0.94, "Roadwork": 0.75, "Accident": 0.55}


def hour_shape(h, wd, prof):
    am, pm, mid = prof
    g = lambda x, mu, s: np.exp(-0.5 * ((x - mu) / s) ** 2)
    if wd < 5:
        return 0.12 + 0.85 * am * g(h, 8.5, 1.3) + 0.9 * pm * g(h, 18, 1.5) + 0.45 * mid * g(h, 13, 2.2)
    return 0.12 + 0.5 * mid * g(h, 12, 3.0) + 0.45 * g(h, 18.5, 2.0)


def build():
    roads = pd.DataFrame(ROADS, columns=["road_id", "road_name", "area", "road_type", "lanes", "speed_limit", "capacity"])
    start = pd.Timestamp("2026-06-01")
    ts = pd.date_range(start, periods=DAYS_OF_DATA * 24, freq="h")

    # one weather state per (day, 4h block) with persistence
    blocks = len(ts) // 4 + 1
    states = list(WEATHER_P)
    w = [RNG.choice(states, p=list(WEATHER_P.values()))]
    for _ in range(blocks - 1):
        w.append(w[-1] if RNG.random() < 0.65 else RNG.choice(states, p=list(WEATHER_P.values())))
    weather_by_ts = np.repeat(w, 4)[: len(ts)]

    rows = []
    for _, r in roads.iterrows():
        prof = PROFILE[r.area]
        base_cond = RNG.choice(list(COND_FACTOR), size=len(ts), p=[0.78, 0.15, 0.05, 0.02])
        for i, t in enumerate(ts):
            wx = weather_by_ts[i]
            cond = base_cond[i]
            if wx in ("Rain", "Storm") and cond == "Dry" and RNG.random() < 0.7:
                cond = "Wet"
            shape = hour_shape(t.hour, t.dayofweek, prof)
            demand = r.capacity * shape * RNG.normal(1.0, 0.08)
            # rain pushes some drivers off the road slightly
            demand *= 0.97 if wx in ("Rain", "Storm") else 1.0
            eff_cap = r.capacity * WEATHER_FACTOR[wx] * COND_FACTOR[cond]
            vol = int(max(20, min(demand, r.capacity * 1.05)))
            ratio = demand / eff_cap
            if ratio < 0.42:
                level = "Low"
            elif ratio < 0.68:
                level = "Medium"
            else:
                level = "High"
            speed = max(5.0, r.speed_limit * (1.05 - 0.85 * min(ratio, 1.3)) + RNG.normal(0, 2))
            temp = 29 + 5 * np.sin((t.hour - 9) / 24 * 2 * np.pi) + RNG.normal(0, 1.2) - (3 if wx in ("Rain", "Storm") else 0)
            vis = {"Clear": 10, "Cloudy": 9, "Rain": 5, "Fog": 0.8, "Storm": 2.5}[wx] * RNG.uniform(0.8, 1.1)
            rows.append((r.road_id, t, t.hour, t.dayofweek, int(t.dayofweek >= 5), wx, round(temp, 1),
                         round(vis, 1), cond, vol, round(speed, 1), level))
    rec = pd.DataFrame(rows, columns=["road_id", "ts", "hour", "day_of_week", "is_weekend", "weather", "temperature",
                                      "visibility_km", "road_condition", "traffic_volume", "avg_speed", "congestion"])
    return roads, rec


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    roads, rec = build()
    eng = create_engine(DATABASE_URL)
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS traffic_records"))
        c.execute(text("DROP TABLE IF EXISTS roads"))
    roads.to_sql("roads", eng, index=False)
    rec.to_sql("traffic_records", eng, index=False, chunksize=5000)
    with eng.begin() as c:
        for stmt in ("CREATE INDEX idx_road_time ON traffic_records (road_id, ts)",
                     "CREATE INDEX idx_hour_day ON traffic_records (hour, day_of_week)",
                     "CREATE INDEX idx_road_name ON roads (road_name)"):
            try:
                c.execute(text(stmt))
            except Exception:
                pass
    print(f"Loaded {len(roads)} roads and {len(rec):,} traffic records into {eng.url.render_as_string(hide_password=True)}")
    print(rec.congestion.value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
