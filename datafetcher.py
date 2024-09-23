import requests
import json
import pandas as pd
import time
import os
import logging
from tqdm import tqdm  # For progress bar
from dotenv import load_dotenv  # To load environment variables
from requests.exceptions import HTTPError
from config import MAX_RETRIES, RATE_LIMIT, BASE_URL  # Assuming you have set these in config.py

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the top 6 leagues (league IDs)
leagues = [39, 140, 78, 61, 135, 94]  # Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Primeira Liga
season = 2023

# API credentials (using environment variables)
api_key = os.getenv("API_FOOTBALL_KEY")
if not api_key:
    raise ValueError("API key not found! Make sure you set your API key in the .env file.")

headers = {
    "x-rapidapi-key": api_key,
    "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
}

# Function to fetch data from an endpoint with error handling and pagination support
def get_data(endpoint, params):
    url = f"{BASE_URL}{endpoint}"

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            return data

        except HTTPError as e:
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt + 1 == MAX_RETRIES:
                logging.error(f"Max retries reached for {endpoint}. Returning None.")
                return None
        except requests.RequestException as e:
            logging.error(f"Error occurred: {e}")
        rate_limit_delay()

    return None  # In case all retries fail

# Adding rate limiting with a delay to avoid hitting API limits
def rate_limit_delay():
    time.sleep(RATE_LIMIT)  # Delay to comply with rate limits

# Function to fetch team statistics for a league
def fetch_team_statistics(league_id, season):
    statistics = {}
    for league in tqdm(league_id, desc="Fetching team statistics"):
        params = {"league": league, "season": season}
        data = get_data("teams/statistics", params)
        if data:
            statistics[league] = data
        rate_limit_delay()
    return statistics

# Function to fetch player performance and form
def fetch_player_performance(league_id, season):
    player_data = {}
    for league in tqdm(league_id, desc="Fetching player performance"):
        params = {"league": league, "season": season}
        data = get_data("players", params)
        if data:
            player_data[league] = data
        rate_limit_delay()
    return player_data

# Function to fetch injuries and suspensions
def fetch_injuries(league_ids, season):
    injuries_data = {}
    for league in tqdm(league_ids, desc="Fetching injuries"):
        params = {"league": league, "season": season}
        data = get_data("injuries", params)
        if data:
            injuries_data[league] = data  # Store data by league ID
        rate_limit_delay()
    return injuries_data

# Function to fetch match odds and betting data
def fetch_match_odds(league_id, season, fixture_id):
    odds_data = {}
    for league in tqdm(league_id, desc="Fetching match odds"):
        params = {"league": league, "season": season, "fixture": fixture_id}
        data = get_data("odds", params)
        if data:
            odds_data[league] = data
        rate_limit_delay()
    return odds_data

# Function to fetch team standings for a league
def fetch_team_standings(league_id, season):
    standings_data = {}
    for league in tqdm(league_id, desc="Fetching standings"):
        params = {"league": league, "season": season}
        data = get_data("standings", params)
        if data:
            standings_data[league] = data
        rate_limit_delay()
    return standings_data

# Fetch all data for leagues
def fetch_all_data():
    all_data = {}

    # Fetch team statistics
    all_data['team_statistics'] = fetch_team_statistics(leagues, season)

    # Fetch player performance data
    all_data['player_performance'] = fetch_player_performance(leagues, season)

    # Fetch injuries and suspensions
    all_data['injuries'] = fetch_injuries(leagues, season)

    # Fetch match odds (Example fixture_id: 123456)
    fixture_id = 123456
    all_data['match_odds'] = fetch_match_odds(leagues, season, fixture_id)

    # Fetch team standings
    all_data['standings'] = fetch_team_standings(leagues, season)

    return all_data

# Function to save data to JSON and CSV formats
def save_data_to_file(data, filename_prefix="football_data"):
    # Save as JSON
    json_filename = f"{filename_prefix}.json"
    with open(json_filename, "w") as json_file:
        json.dump(data, json_file, indent=4)
    logging.info(f"Data saved to {json_filename}")

    # Save some data as CSV (team statistics as an example)
    if 'team_statistics' in data:
        stats_df = pd.json_normalize(data['team_statistics'])
        csv_filename = f"{filename_prefix}_team_stats.csv"
        stats_df.to_csv(csv_filename, index=False)
        logging.info(f"Team statistics saved to {csv_filename}")

# Main function to run the script
if __name__ == "__main__":
    logging.info("Starting data fetching process...")
    data = fetch_all_data()

    # Save data to files
    save_data_to_file(data)

    logging.info("Data fetching and saving process complete!")
