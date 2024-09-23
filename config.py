import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# API keys
API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')  # Ensure this is included in your .env file

# Base URLs for APIs
API_FOOTBALL_BASE_URL = 'https://api-football-v1.p.rapidapi.com/v3/'
WEATHER_API_BASE_URL = 'http://api.weatherapi.com/v1/'

# Database path for local storage
DB_PATH = os.path.join(os.getenv('DB_DIRECTORY', 'C:/Users/scar4/fotora'), 'football_data.db')

# API settings
MAX_RETRIES = 3  # Maximum number of retry attempts for API requests
RATE_LIMIT = 1.5  # Rate limit (in seconds) for API requests

# Top 6 football leagues (mapped by their API league IDs)
TOP_LEAGUES = {
    39: "English Premier League",
    140: "La Liga",
    78: "Bundesliga",
    61: "Ligue 1",
    135: "Serie A",
    94: "Primeira Liga"
}

# Current season (update annually)
CURRENT_SEASON = "2023"
