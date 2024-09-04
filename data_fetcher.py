import os
import requests
from dotenv import load_dotenv
import asyncio
import aiohttp

# Load environment variables
load_dotenv()

API_KEY = os.getenv('API_KEY')

PINNACLE_URL = "<https://pinnacle-odds.p.rapidapi.com/kit/v1/special-markets>"
LIVESCORE_URL = "<https://livescore6.p.rapidapi.com/v2/search>"
API_FOOTBALL_URL = "<https://api-football-v1.p.rapidapi.com/v2/odds/league/865927/bookmaker/5>"

headers = {
    "x-rapidapi-key": API_KEY,
    "x-rapidapi-host": "pinnacle-odds.p.rapidapi.com"
}

async def fetch_data(session, url, params=None):
    async with session.get(url, headers=headers, params=params) as response:
        return await response.json()

async def fetch_all_data():
    async with aiohttp.ClientSession() as session:
        pinnacle_task = fetch_data(session, PINNACLE_URL, {"is_have_odds": "true", "sport_id": "1"})
        livescore_task = fetch_data(session, LIVESCORE_URL, {"Category": "soccer", "Query": "chel"})
        api_football_task = fetch_data(session, API_FOOTBALL_URL, {"page": "2"})

        return await asyncio.gather(pinnacle_task, livescore_task, api_football_task)

def get_all_data():
    return asyncio.run(fetch_all_data())

if __name__ == "__main__":
    pinnacle_data, livescore_data, api_football_data = get_all_data()
    print("Pinnacle Data:", pinnacle_data)
    print("Livescore Data:", livescore_data)
    print("API-Football Data:", api_football_data)

