"""Flask app: search roads, predict congestion, trends, peak periods, model comparison, dashboard."""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sqlalchemy import create_engine, text

import charts
from config import DATABASE_URL, DAYS, LEVELS, METRICS_PATH, MODEL_PATH, ROAD_CONDITIONS, WEATHER
from train_models import add_history

app = Flask(__name__)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
_cache = {}


def q(sql, **params):
    with engine.connect() as c:
        return pd.read_sql(text(sql), c, params=params)


def cached(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


BUNDLE = load_model()


def roads_df():
    return cached("roads", lambda: q("SELECT * FROM roads ORDER BY road_name"))


# ---------------------------------------------------------------- road search
@app.route("/api/roads")
def api_roads():
    term = request.args.get("q", "").strip().lower()
    df = roads_df()
    if term:
        df = df[df.road_name.str.lower().str.contains(term) | df.area.str.lower().str.contains(term)]
    return jsonify(df.head(15).to_dict("records"))


# ------------------------------------------------------------------ predict
def predict_row(road, hour, dow, weather, cond, temp, vis):
    row = pd.DataFrame([{
        "road_id": road.road_id, "road_type": road.road_type, "lanes": road.lanes,
        "speed_limit": road.speed_limit, "capacity": road.capacity, "hour": hour,
        "day_of_week": dow, "is_weekend": int(dow >= 5), "weather": weather,
        "road_condition": cond, "temperature": temp, "visibility_km": vis,
    }])
    row = add_history(row, BUNDLE["hist"])
    model = BUNDLE["model"]
    proba = model.predict_proba(row[BUNDLE["features"]])[0]
    classes = list(model.classes_)
    probs = {l: float(proba[classes.index(l)]) for l in LEVELS}
    return max(probs, key=probs.get), probs, float(row.hist_volume.iloc[0])


DEFAULT_VIS = {"Clear": 10.0, "Cloudy": 9.0, "Rain": 5.0, "Fog": 0.8, "Storm": 2.5}


@app.route("/predict", methods=["GET", "POST"])
def predict():
    roads = roads_df()
    form = {"road_id": roads.road_id.iloc[0], "hour": 8, "dow": 0, "weather": "Clear", "cond": "Dry", "temp": 30.0}
    result = None
    if BUNDLE is None:
        return render_template("predict.html", roads=roads.to_dict("records"), form=form, result=None,
                               weathers=WEATHER, conds=ROAD_CONDITIONS, days=DAYS, no_model=True)
    if request.method == "POST":
        form = {"road_id": int(request.form["road_id"]), "hour": int(request.form["hour"]),
                "dow": int(request.form["dow"]), "weather": request.form["weather"],
                "cond": request.form["road_condition"], "temp": float(request.form.get("temp") or 30)}
        road = roads[roads.road_id == form["road_id"]].iloc[0]
        vis = DEFAULT_VIS.get(form["weather"], 9.0)
        level, probs, hist_vol = predict_row(road, form["hour"], form["dow"], form["weather"], form["cond"], form["temp"], vis)
        day_levels, day_high = [], []
        for h in range(24):
            lv, pr, _ = predict_row(road, h, form["dow"], form["weather"], form["cond"], form["temp"], vis)
            day_levels.append(lv)
            day_high.append(pr["High"])
        result = {
            "road": road.to_dict(), "level": level, "probs": probs, "hist_vol": round(hist_vol),
            "load_pct": round(hist_vol / road.capacity * 100), "model": BUNDLE["name"],
            "prob_chart": charts.prob_bars(probs),
            "day_chart": charts.forecast_strip(list(range(24)), day_levels, day_high),
            "hour_label": f"{DAYS[form['dow']]} {form['hour']:02d}:00",
        }
    return render_template("predict.html", roads=roads.to_dict("records"), form=form, result=result,
                           weathers=WEATHER, conds=ROAD_CONDITIONS, days=DAYS, no_model=False)


# ------------------------------------------------------------------- trends
def hourly_df(road_id=None):
    where = "WHERE road_id = :rid" if road_id else ""
    df = q(f"""SELECT hour, is_weekend, AVG(traffic_volume) AS vol FROM traffic_records {where}
               GROUP BY hour, is_weekend""", **({"rid": road_id} if road_id else {}))
    wk = df[df.is_weekend == 0].set_index("hour").vol
    we = df[df.is_weekend == 1].set_index("hour").vol
    return pd.DataFrame({"hour": range(24), "weekday_vol": wk.reindex(range(24)).values, "weekend_vol": we.reindex(range(24)).values})


def daily_df(road_id=None):
    where = "WHERE road_id = :rid" if road_id else ""
    return q(f"SELECT day_of_week, AVG(traffic_volume) AS avg_vol FROM traffic_records {where} GROUP BY day_of_week",
             **({"rid": road_id} if road_id else {}))


def heat_pivot(road_id=None):
    where = "WHERE t.road_id = :rid" if road_id else ""
    df = q(f"""SELECT t.day_of_week, t.hour, AVG(t.traffic_volume * 1.0 / r.capacity) AS ratio
               FROM traffic_records t JOIN roads r ON r.road_id = t.road_id {where}
               GROUP BY t.day_of_week, t.hour""", **({"rid": road_id} if road_id else {}))
    return df.pivot(index="day_of_week", columns="hour", values="ratio").reindex(index=range(7), columns=range(24)).fillna(0)


@app.route("/trends")
def trends():
    roads = roads_df()
    rid = request.args.get("road_id", type=int)
    road = roads[roads.road_id == rid].iloc[0].to_dict() if rid in set(roads.road_id) else None
    key = f"trends-{rid}"
    ch = cached(key, lambda: {
        "hourly": charts.hourly_line(hourly_df(rid), "Hourly trend" + (f": {road['road_name']}" if road else ": all roads")),
        "daily": charts.daily_bars(daily_df(rid), "Daily trend" + (f": {road['road_name']}" if road else ": all roads")),
        "heat": charts.heatmap(heat_pivot(rid)),
    })
    return render_template("trends.html", roads=roads.to_dict("records"), sel=road, charts=ch)


# -------------------------------------------------------------------- peaks
def peak_table():
    df = q("""SELECT t.road_id, r.road_name, t.hour, t.is_weekend, AVG(t.traffic_volume) AS vol,
                     AVG(CASE WHEN t.congestion = 'High' THEN 1.0 ELSE 0 END) AS high_share
              FROM traffic_records t JOIN roads r ON r.road_id = t.road_id
              GROUP BY t.road_id, r.road_name, t.hour, t.is_weekend""")
    rows = []
    for (rid, name), g in df.groupby(["road_id", "road_name"]):
        wk = g[g.is_weekend == 0].sort_values("vol", ascending=False)
        we = g[g.is_weekend == 1].sort_values("vol", ascending=False)
        pk = sorted(wk.head(3).hour.tolist())
        rows.append({
            "road": name,
            "weekday_peak": ", ".join(f"{h:02d}:00" for h in pk),
            "weekend_peak": ", ".join(f"{h:02d}:00" for h in sorted(we.head(2).hour.tolist())),
            "peak_volume": int(wk.vol.max()),
            "high_hours_pct": round(g[g.is_weekend == 0].high_share.mean() * 100, 1),
        })
    return sorted(rows, key=lambda r: -r["high_hours_pct"])


@app.route("/peaks")
def peaks():
    hs = cached("highshare", lambda: q("""SELECT hour, AVG(CASE WHEN congestion='High' THEN 1.0 ELSE 0 END) AS high_share
                                         FROM traffic_records GROUP BY hour ORDER BY hour"""))
    top = hs.sort_values("high_share", ascending=False).head(4)
    worst_day = cached("worstday", lambda: q("""SELECT day_of_week, AVG(CASE WHEN congestion='High' THEN 1.0 ELSE 0 END) AS s
                                               FROM traffic_records GROUP BY day_of_week ORDER BY s DESC LIMIT 1"""))
    return render_template("peaks.html", chart=cached("peakchart", lambda: charts.high_share_by_hour(hs)),
                           table=cached("peaktable", peak_table),
                           top_hours=[f"{int(h):02d}:00" for h in sorted(top.hour)],
                           worst_day=DAYS[int(worst_day.day_of_week.iloc[0])])


# ------------------------------------------------------------------- models
@app.route("/models")
def models():
    if not os.path.exists(METRICS_PATH):
        return render_template("models.html", m=None)
    with open(METRICS_PATH) as f:
        m = json.load(f)
    ch = cached("modelcharts", lambda: {
        "compare": charts.model_compare(m["results"]),
        "cms": {n: charts.confusion(r["confusion"], n) for n, r in m["results"].items()},
        "imp": charts.importances(m["importances"]),
    })
    return render_template("models.html", m=m, charts=ch)


# ---------------------------------------------------------------- dashboard
@app.route("/")
def dashboard():
    def build():
        kpi = q("""SELECT COUNT(*) AS n, AVG(traffic_volume) AS vol,
                          AVG(CASE WHEN congestion='High' THEN 1.0 ELSE 0 END) AS high,
                          AVG(avg_speed) AS speed FROM traffic_records""").iloc[0]
        mix = q("""SELECT r.road_name,
                          AVG(CASE WHEN t.congestion='Low' THEN 1.0 ELSE 0 END) AS Low,
                          AVG(CASE WHEN t.congestion='Medium' THEN 1.0 ELSE 0 END) AS Medium,
                          AVG(CASE WHEN t.congestion='High' THEN 1.0 ELSE 0 END) AS High
                   FROM traffic_records t JOIN roads r ON r.road_id = t.road_id GROUP BY r.road_name""")
        wx = q("""SELECT weather, AVG(CASE WHEN congestion='High' THEN 1.0 ELSE 0 END) AS high_share
                  FROM traffic_records GROUP BY weather""")
        worst = mix.sort_values("High", ascending=False).iloc[0]
        return {
            "kpi": {"records": f"{int(kpi.n):,}", "roads": len(roads_df()), "vol": f"{kpi.vol:,.0f}",
                    "high": f"{kpi.high * 100:.1f}%", "speed": f"{kpi.speed:.0f}",
                    "worst": worst.road_name, "worst_pct": f"{worst.High * 100:.0f}%"},
            "hourly": charts.hourly_line(hourly_df()),
            "daily": charts.daily_bars(daily_df()),
            "heat": charts.heatmap(heat_pivot()),
            "mix": charts.level_stack_by_road(mix),
            "weather": charts.weather_bars(wx),
        }
    return render_template("dashboard.html", d=cached("dash", build),
                           best=(json.load(open(METRICS_PATH))["best"] if os.path.exists(METRICS_PATH) else None))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
