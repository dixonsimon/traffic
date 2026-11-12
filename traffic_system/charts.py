"""Matplotlib chart helpers. Every function returns a base64 PNG string for <img src="data:...">."""
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DAYS, LEVELS

LEVEL_COLORS = {"Low": "#2a9d8f", "Medium": "#e9a23b", "High": "#d1495b"}
PRIMARY = "#2f5d8a"
MUTED = "#8a94a3"
INK = "#1f2933"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": "#c9d1db", "axes.labelcolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e8ecf1",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 110,
})


def _b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def hourly_line(df, title="Average traffic volume by hour", series=None):
    """df columns: hour, weekday_vol, weekend_vol (optional)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(df["hour"], df["weekday_vol"], color=PRIMARY, lw=2.4, marker="o", ms=3.5, label="Weekday")
    if "weekend_vol" in df:
        ax.plot(df["hour"], df["weekend_vol"], color="#e9a23b", lw=2.4, marker="o", ms=3.5, label="Weekend")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Vehicles / hour")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left")
    return _b64(fig)


def daily_bars(df, title="Average traffic volume by day"):
    """df columns: day_of_week, avg_vol."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    d = df.sort_values("day_of_week")
    colors = [PRIMARY if i < 5 else "#e9a23b" for i in d["day_of_week"]]
    ax.bar([DAYS[i] for i in d["day_of_week"]], d["avg_vol"], color=colors, width=0.62)
    ax.set_ylabel("Vehicles / hour")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    return _b64(fig)


def heatmap(pivot, title="Congestion load: volume as % of road capacity"):
    """pivot: index day_of_week 0-6, columns hour 0-23, values ratio 0-1."""
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    im = ax.imshow(pivot.values * 100, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_yticks(range(7))
    ax.set_yticklabels(DAYS)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_title(title)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label("% of capacity")
    cb.outline.set_visible(False)
    return _b64(fig)


def level_stack_by_road(df):
    """df columns: road_name, Low, Medium, High (shares)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    d = df.sort_values("High")
    left = np.zeros(len(d))
    for lv in LEVELS:
        ax.barh(d["road_name"], d[lv] * 100, left=left, color=LEVEL_COLORS[lv], label=lv, height=0.65)
        left += d[lv].values * 100
    ax.set_xlabel("% of recorded hours")
    ax.set_xlim(0, 100)
    ax.set_title("Congestion mix by road")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="y", visible=False)
    return _b64(fig)


def weather_bars(df):
    """df columns: weather, high_share."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    d = df.sort_values("high_share", ascending=False)
    ax.bar(d["weather"], d["high_share"] * 100, color=PRIMARY, width=0.6)
    ax.set_ylabel("% of hours with High congestion")
    ax.set_title("Weather impact")
    ax.grid(axis="x", visible=False)
    return _b64(fig)


def high_share_by_hour(df):
    """df columns: hour, high_share."""
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    peak = df["high_share"] >= df["high_share"].max() * 0.7
    ax.bar(df["hour"], df["high_share"] * 100, color=[LEVEL_COLORS["High"] if p else "#c9d1db" for p in peak], width=0.75)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("% of hours High")
    ax.set_title("Peak periods: share of High-congestion hours")
    ax.grid(axis="x", visible=False)
    return _b64(fig)


def forecast_strip(hours, levels, probs_high, hour_now=None):
    """24-hour predicted congestion for one road/day."""
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.bar(hours, [1] * len(hours), color=[LEVEL_COLORS[l] for l in levels], width=0.9)
    for h, p in zip(hours, probs_high):
        ax.text(h, 0.5, f"{p * 100:.0f}", ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    ax.set_yticks([])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (numbers = % chance of High)")
    ax.set_title("Predicted congestion across the day")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=LEVEL_COLORS[l]) for l in LEVELS]
    ax.legend(handles, LEVELS, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.5))
    return _b64(fig)


def prob_bars(probs):
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.barh(LEVELS[::-1], [probs[l] * 100 for l in LEVELS[::-1]], color=[LEVEL_COLORS[l] for l in LEVELS[::-1]], height=0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    for i, l in enumerate(LEVELS[::-1]):
        ax.text(probs[l] * 100 + 1.5, i, f"{probs[l] * 100:.0f}%", va="center", fontsize=9)
    ax.grid(axis="y", visible=False)
    return _b64(fig)


def model_compare(results):
    names = list(results)
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(names))
    w = 0.2
    shades = ["#9db8d3", "#6f98bf", "#4a7aa8", PRIMARY]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    for i, m in enumerate(metrics):
        vals = [results[n][m] * 100 for n in names]
        ax.bar(x + (i - 1.5) * w, vals, w, label=m.capitalize() if m != "f1" else "F1", color=shades[i])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    lo = min(results[n][m] for n in names for m in metrics) * 100
    ax.set_ylim(max(0, lo - 6), 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model comparison on held-out test set")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="x", visible=False)
    return _b64(fig)


def confusion(cm, title):
    cm = np.array(cm)
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LEVELS); ax.set_yticklabels(LEVELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="white" if norm[i, j] > 0.55 else INK, fontsize=9)
    return _b64(fig)


def importances(imp):
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    s = sorted(imp.items(), key=lambda kv: kv[1])
    ax.barh([k.replace("num__", "").replace("cat__", "") for k, _ in s], [v for _, v in s], color=PRIMARY, height=0.6)
    ax.set_title("Top features (Random Forest)")
    ax.set_xlabel("Importance")
    ax.grid(axis="y", visible=False)
    return _b64(fig)
