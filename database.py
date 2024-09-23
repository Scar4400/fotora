import sqlite3
from datetime import datetime
import json
import pandas as pd
from typing import Dict, Any
from config import DB_PATH

class FootballDatabase:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self.cursor = self.conn.cursor()
        self.init_db()

    def init_db(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS cache
                               (endpoint TEXT, params TEXT, data TEXT, timestamp DATETIME)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS predictions
                               (fixture_id INTEGER PRIMARY KEY, league_id INTEGER, home_team TEXT, away_team TEXT,
                                predicted_outcome TEXT, actual_outcome TEXT, probability REAL,
                                accuracy REAL, temperature REAL, wind_speed REAL, precipitation REAL, timestamp DATETIME)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS feature_importance
                               (feature TEXT, importance REAL, timestamp DATETIME)''')
        self.conn.commit()

    def store_prediction(self, fixture_id: int, league_id: int, home_team: str, away_team: str,
                         predicted_outcome: str, probability: float, weather_data: Dict[str, Any]):
        """
        Stores a prediction along with associated weather data (temperature, wind speed, precipitation).
        """
        temperature = weather_data.get('temp_c')  # Temperature in Celsius
        wind_speed = weather_data.get('wind_kph')  # Wind speed in kph
        precipitation = weather_data.get('precip_mm')  # Precipitation in mm

        self.cursor.execute("""INSERT OR REPLACE INTO predictions
                               (fixture_id, league_id, home_team, away_team, predicted_outcome, probability,
                                temperature, wind_speed, precipitation, timestamp)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (fixture_id, league_id, home_team, away_team, predicted_outcome, probability,
                             temperature, wind_speed, precipitation, datetime.now().isoformat()))
        self.conn.commit()

    def update_prediction_accuracy(self, fixture_id: int, actual_outcome: str):
        self.cursor.execute("SELECT predicted_outcome FROM predictions WHERE fixture_id=?", (fixture_id,))
        result = self.cursor.fetchone()
        if result:
            predicted_outcome = result[0]
            accuracy = 1 if predicted_outcome == actual_outcome else 0
            self.cursor.execute("UPDATE predictions SET actual_outcome=?, accuracy=? WHERE fixture_id=?",
                                (actual_outcome, accuracy, fixture_id))
            self.conn.commit()

    def get_prediction_accuracy(self, league_id: int = None):
        if league_id:
            self.cursor.execute("SELECT AVG(accuracy) FROM predictions WHERE league_id=? AND actual_outcome IS NOT NULL",
                                (league_id,))
        else:
            self.cursor.execute("SELECT AVG(accuracy) FROM predictions WHERE actual_outcome IS NOT NULL")
        result = self.cursor.fetchone()
        return result[0] if result[0] is not None else 0

    def get_historical_data(self) -> pd.DataFrame:
        query = "SELECT * FROM predictions WHERE actual_outcome IS NOT NULL"
        return pd.read_sql_query(query, self.conn)

    def cache_data(self, endpoint: str, params: Dict[str, Any], data: Dict[str, Any]):
        self.cursor.execute("INSERT INTO cache VALUES (?, ?, ?, ?)",
                            (endpoint, json.dumps(params), json.dumps(data), datetime.now().isoformat()))
        self.conn.commit()

    def get_cached_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self.cursor.execute("SELECT data FROM cache WHERE endpoint=? AND params=? ORDER BY timestamp DESC LIMIT 1",
                            (endpoint, json.dumps(params)))
        result = self.cursor.fetchone()
        return json.loads(result[0]) if result else None

    def cache_weather_data(self, fixture_id: int, weather_data: Dict[str, Any]):
        """
        Caches weather data for a specific fixture.
        """
        params = {'fixture_id': fixture_id}
        self.cache_data(endpoint='weather', params=params, data=weather_data)

    def get_cached_weather_data(self, fixture_id: int) -> Dict[str, Any]:
        """
        Retrieves cached weather data for a specific fixture if available.
        """
        params = {'fixture_id': fixture_id}
        return self.get_cached_data(endpoint='weather', params=params)

    def store_feature_importance(self, feature_importance: pd.DataFrame):
        for _, row in feature_importance.iterrows():
            self.cursor.execute("INSERT INTO feature_importance VALUES (?, ?, ?)",
                                (row['feature'], row['importance'], datetime.now().isoformat()))
        self.conn.commit()

    def get_feature_importance(self) -> pd.DataFrame:
        query = "SELECT feature, importance FROM feature_importance ORDER BY timestamp DESC LIMIT 1"
        return pd.read_sql_query(query, self.conn)

    def close(self):
        self.conn.close()
