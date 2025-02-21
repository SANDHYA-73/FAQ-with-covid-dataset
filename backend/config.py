import os
from dotenv import load_dotenv

load_dotenv()
DIAL_API_KEY = os.getenv("DIAL_API_KEY")

if not DIAL_API_KEY:
    raise ValueError("Missing EPAM Dial API Key in .env file.")

# Debugging: Print API key to check if it's loading correctly
# print(f"Debug: Loaded API Key - {DIAL_API_KEY}")
