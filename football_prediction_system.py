import asyncio
import logging
import aiohttp
import pandas as pd
import json
import os
from typing import Dict, Any
from config import TOP_LEAGUES, CURRENT_SEASON, MAX_RETRIES
from feature_engineering import engineer_features
from model import PredictionModel
from database import FootballDatabase
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths to the pre-fetched data
DATA_JSON_PATH = 'football_data.json'
DATA_CSV_PATH = 'football_data_team_stats.csv'

class FootballPredictionSystem:
    def __init__(self):
        self.db = FootballDatabase()
        self.model = PredictionModel()
        self.all_data = self.load_data_from_files()
        self.team_stats = self.load_team_stats_from_csv()

    def load_data_from_files(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        if os.path.exists(DATA_JSON_PATH):
            with open(DATA_JSON_PATH, 'r') as f:
                return json.load(f)
        logger.error("No football_data.json file found. Exiting.")
        exit()

    def load_team_stats_from_csv(self) -> pd.DataFrame:
        """Load team statistics from CSV file."""
        if os.path.exists(DATA_CSV_PATH):
            return pd.read_csv(DATA_CSV_PATH)
        logger.error("No football_data_team_stats.csv file found. Exiting.")
        exit()

    async def fetch_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve cached data from the JSON or return data based on pre-fetched data."""
        if endpoint == 'fixtures':
            return self.all_data.get('fixtures', {})
        elif endpoint == 'team_details':
            return self.team_stats[self.team_stats['team_id'] == params['team_id']]
        elif endpoint == 'standings':
            return self.all_data.get('standings', {})
        elif endpoint == 'head_to_head':
            return self.all_data.get('head_to_head', {})
        elif endpoint == 'injuries':
            return self.all_data.get('injuries', {})
        return None

    async def process_fixture(self, fixture: Dict[str, Any], standings: Dict[str, Any]):
        """Process individual fixture and make predictions."""
        home_team_id = fixture['teams']['home']['id']
        away_team_id = fixture['teams']['away']['id']
        league_id = fixture['league']['id']

        match_data = {
            'fixture': fixture,
            'team1_details': await self.fetch_data('team_details', {'team_id': home_team_id}),
            'team2_details': await self.fetch_data('team_details', {'team_id': away_team_id}),
            'standings': standings,
            'head_to_head': await self.fetch_data('head_to_head', {'home_id': home_team_id, 'away_id': away_team_id}),
            'injuries': {
                'home': await self.fetch_data('injuries', {'team_id': home_team_id}),
                'away': await self.fetch_data('injuries', {'team_id': away_team_id})
            }
        }

        features = engineer_features(match_data)
        predicted_outcome, probability = self.model.predict(features)

        self.db.store_prediction(fixture['fixture']['id'], league_id,
                                 fixture['teams']['home']['name'],
                                 fixture['teams']['away']['name'],
                                 predicted_outcome, probability)

        logger.info(f"Prediction for {fixture['teams']['home']['name']} vs "
                    f"{fixture['teams']['away']['name']}: {predicted_outcome} (probability: {probability:.2f})")

    async def process_league(self, league_id: int, season: str):
        """Process fixtures for a league."""
        logger.info(f"Processing league: {TOP_LEAGUES[league_id]}")
        fixtures = await self.fetch_data('fixtures', {'league_id': league_id, 'season': season})
        standings = await self.fetch_data('standings', {'league_id': league_id, 'season': season})

        if fixtures and standings:
            tasks = [self.process_fixture(fixture, standings) for fixture in fixtures['response']]
            await asyncio.gather(*tasks)
        else:
            logger.error(f"Failed to fetch data for league {league_id}")

    async def predict_matches_for_all_leagues(self):
        """Predict matches for all top leagues."""
        tasks = [self.process_league(league_id, CURRENT_SEASON) for league_id in TOP_LEAGUES.keys()]
        await asyncio.gather(*tasks)

    def update_model(self):
        """Update the model based on historical data."""
        historical_data = self.db.get_historical_data()
        if not historical_data.empty:
            X = historical_data.drop(['fixture_id', 'league_id', 'home_team', 'away_team', 'predicted_outcome', 'actual_outcome', 'accuracy', 'timestamp'], axis=1)
            y = historical_data['actual_outcome']
            self.model.train(X, y)
            feature_importance = self.model.get_feature_importance()
            self.db.store_feature_importance(feature_importance)
            logger.info("Model updated with historical data")
        else:
            logger.info("No historical data available for model update")

    def evaluate_predictions(self):
        """Evaluate recent predictions."""
        one_week_ago = datetime.now() - timedelta(days=7)
        recent_predictions = self.db.get_historical_data(timestamp_after=one_week_ago)
        if not recent_predictions.empty:
            X = recent_predictions.drop(['fixture_id', 'league_id', 'home_team', 'away_team', 'predicted_outcome', 'actual_outcome', 'accuracy', 'timestamp'], axis=1)
            y = recent_predictions['actual_outcome']
            evaluation_metrics = self.model.evaluate(X, y)
            logger.info(f"Recent model performance: {evaluation_metrics}")
        else:
            logger.info("No recent predictions available for evaluation")

    async def run(self):
        """Main run loop to predict matches, update the model, and evaluate predictions."""
        while True:
            try:
                await self.predict_matches_for_all_leagues()
                self.update_model()
                self.evaluate_predictions()
            except Exception as e:
                logger.error(f"An error occurred in the prediction cycle: {str(e)}")

            await asyncio.sleep(86400)  # Wait for 24 hours before the next cycle

if __name__ == "__main__":
    prediction_system = FootballPredictionSystem()
    asyncio.run(prediction_system.run())
