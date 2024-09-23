import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load fetched football data
try:
    with open('football_data.json', 'r') as f:
        football_data = json.load(f)
    logging.info(f"Loaded football_data.json. Keys: {football_data.keys()}")
except FileNotFoundError:
    logging.error("football_data.json file not found. Please ensure it's in the correct directory.")
    football_data = {}

def get_team_data(team_id: int, league_id: int) -> Tuple[Dict, Dict]:
    """Get team stats and standings data."""
    team_stats = {}
    standings = {}

    # Get team statistics
    if 'team_statistics' in football_data and str(league_id) in football_data['team_statistics']:
        team_stats_data = football_data['team_statistics'][str(league_id)]
        if str(team_id) in team_stats_data:
            team_stats = team_stats_data[str(team_id)]
        else:
            logging.warning(f"No team statistics found for team {team_id} in league {league_id}")
    else:
        logging.warning(f"No team statistics found for league {league_id}")

    # Get standings
    if 'standings' in football_data and str(league_id) in football_data['standings']:
        standings_data = football_data['standings'][str(league_id)]['response']
        standings = next((team for team in standings_data if team['team']['id'] == team_id), {})
    else:
        logging.warning(f"No standings data found for league {league_id}")

    logging.info(f"Team {team_id} stats keys: {team_stats.keys()}")
    logging.info(f"Team {team_id} standings keys: {standings.keys()}")

    return team_stats, standings

def calculate_form(form_string: str) -> float:
    """Calculate form based on recent results."""
    if not form_string:
        return 0.0
    return sum(1 if result == 'W' else 0.5 if result == 'D' else 0 for result in form_string) / len(form_string)

def get_injuries(team_id: int, league_id: int) -> List[Dict]:
    """Get injuries for a team."""
    if 'injuries' in football_data and str(league_id) in football_data['injuries']:
        return [injury for injury in football_data['injuries'][str(league_id)]['response']
                if injury['team']['id'] == team_id]
    return []

def get_h2h_data(home_team_id: int, away_team_id: int) -> List[Dict]:
    """Get head-to-head data for two teams."""
    h2h_key = f"{home_team_id}-{away_team_id}"
    if 'h2h' in football_data and h2h_key in football_data['h2h']:
        return football_data['h2h'][h2h_key]['response']
    return []

def calculate_recent_performance(h2h_data: List[Dict], team_id: int, num_matches: int = 5) -> float:
    """Calculate recent performance based on last few matches."""
    if not h2h_data:
        return 0.0
    recent_matches = sorted(h2h_data, key=lambda x: x['fixture']['date'], reverse=True)[:num_matches]
    if not recent_matches:
        return 0.0
    performance = sum(3 if (match['teams']['home']['id'] == team_id and match['goals']['home'] > match['goals']['away']) or
                      (match['teams']['away']['id'] == team_id and match['goals']['away'] > match['goals']['home']) else
                      1 if match['goals']['home'] == match['goals']['away'] else 0 for match in recent_matches)
    return performance / (len(recent_matches) * 3)

def feature_engineering(home_team_id: int, away_team_id: int, league_id: int) -> pd.DataFrame:
    """Generate features for match prediction."""
    home_team_stats, home_standings = get_team_data(home_team_id, league_id)
    away_team_stats, away_standings = get_team_data(away_team_id, league_id)

    home_injuries = get_injuries(home_team_id, league_id)
    away_injuries = get_injuries(away_team_id, league_id)

    h2h_data = get_h2h_data(home_team_id, away_team_id)

    features = {
        'home_team_rank': home_standings.get('rank', 0),
        'away_team_rank': away_standings.get('rank', 0),
        'home_team_form': calculate_form(home_standings.get('form', '')),
        'away_team_form': calculate_form(away_standings.get('form', '')),
        'home_team_injuries': len(home_injuries),
        'away_team_injuries': len(away_injuries),
        'home_team_goal_diff': home_standings.get('goalsDiff', 0),
        'away_team_goal_diff': away_standings.get('goalsDiff', 0),
        'home_team_clean_sheets': home_team_stats.get('clean_sheet', {}).get('total', 0),
        'away_team_clean_sheets': away_team_stats.get('clean_sheet', {}).get('total', 0),
        'home_team_attack': home_team_stats.get('goals', {}).get('for', {}).get('average', {}).get('total', 0),
        'away_team_attack': away_team_stats.get('goals', {}).get('for', {}).get('average', {}).get('total', 0),
        'home_team_defense': home_team_stats.get('goals', {}).get('against', {}).get('average', {}).get('total', 0),
        'away_team_defense': away_team_stats.get('goals', {}).get('against', {}).get('average', {}).get('total', 0),
        'home_team_recent_performance': calculate_recent_performance(h2h_data, home_team_id),
        'away_team_recent_performance': calculate_recent_performance(h2h_data, away_team_id),
    }

    return pd.DataFrame([features])

def main():
    try:
        # Sample data - replace with actual team IDs and league ID
        home_team_id = 33  # Example: Manchester United
        away_team_id = 34  # Example: Newcastle
        league_id = 39     # Example: Premier League

        logging.info(f"Processing teams: Home {home_team_id}, Away {away_team_id}, League {league_id}")

        features = feature_engineering(home_team_id, away_team_id, league_id)
        print("Generated features:")
        print(features)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
