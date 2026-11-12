# Smart Traffic Congestion Prediction System

Predicts road congestion (Low / Medium / High) from historical traffic volume, time, weather and road conditions.
Stack: Python, Flask, MySQL, Scikit-learn, Matplotlib.

## Quick start

```bash
pip install -r requirements.txt

# 1. (MySQL) create the database and point the app at it
mysql -u root -p < schema.sql
export DATABASE_URL="mysql+pymysql://root:YOUR_PASSWORD@localhost:3306/traffic_db"
#    Skip both lines to use a local SQLite file instead (zero setup).

python generate_data.py     # 12 roads x 120 days x 24 h = 34,560 synthetic records
python train_models.py      # trains 4 models, saves the best + metrics to models/
python app.py               # http://127.0.0.1:5000
```

## Pages

| Page | What it does |
|---|---|
| `/` Dashboard | KPIs, hourly + daily trends, day x hour heatmap, congestion mix per road, weather impact |
| `/predict` | Search a road (live autocomplete), pick day/hour/weather/road condition, get level + probabilities + a 24-hour forecast |
| `/trends` | Hourly and daily congestion trends, for all roads or one selected road |
| `/peaks` | Peak hours network-wide and per road (weekday vs weekend) |
| `/models` | Accuracy / precision / recall / F1 for the four models, confusion matrices, feature importance |

## Models compared

Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors. The best by weighted F1 is saved as `models/best_model.joblib` and used by `/predict`.

## Design notes

- **No leakage.** The current hour's volume is not a model input (it would give the answer away). Instead the model gets the *historical* mean volume for that road/day/hour and the previous hour, computed from the training split only.
- **Congestion label** = demand relative to effective capacity, where weather and road condition (rain, fog, roadwork, accident) reduce capacity. Low < 0.42, Medium < 0.68, High otherwise.
- **Using real data.** Fill the `roads` and `traffic_records` tables (see `schema.sql`) with your own data, then re-run `train_models.py`. Nothing else changes. Adjust the label rule if your data already has a congestion column.
- The synthetic data is smooth, so the simple linear model scores best. Expect Random Forest to pull ahead on messier real data; rerun `train_models.py` and the dashboard picks the winner automatically.

## Files

```
app.py            Flask routes
charts.py         Matplotlib charts (returned as base64 PNGs)
config.py         DB URL, labels
generate_data.py  Synthetic data -> database
train_models.py   Feature building, 4 models, evaluation
schema.sql        MySQL schema
templates/ static/
```
