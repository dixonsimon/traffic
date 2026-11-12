"""Central configuration.

Set DATABASE_URL to use MySQL, e.g.
    export DATABASE_URL="mysql+pymysql://root:yourpassword@localhost:3306/traffic_db"

If DATABASE_URL is not set, the project falls back to a local SQLite file so it
can run anywhere without a database server. Everything else is identical.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "sqlite:///" + os.path.join(DATA_DIR, "traffic.db")
)

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

LEVELS = ["Low", "Medium", "High"]
WEATHER = ["Clear", "Cloudy", "Rain", "Fog", "Storm"]
ROAD_CONDITIONS = ["Dry", "Wet", "Roadwork", "Accident"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
