
import requests
import pandas as pd
import sqlite3
from datetime import datetime
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import json

# Load config
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Constants
DEXSCREENER_API_URL = config["dex_api_url"]
POCKET_UNIVERSE_API_URL = config["pocket_universe_api_url"]
POCKET_UNIVERSE_API_KEY = config["pocket_universe_api_key"]
RUGCHECK_API_URL = config["rugcheck_api_url"]
TROJAN_API_URL = config["trojan_api_url"]
TROJAN_API_KEY = config["trojan_api_key"]
TELEGRAM_BOT_TOKEN = config["telegram_bot_token"]
TELEGRAM_CHAT_ID = config["telegram_chat_id"]
DATABASE_NAME = config["database_name"]
UPDATE_INTERVAL = config["update_interval"]
MIN_LIQUIDITY = config["filters"]["min_liquidity"]
MAX_PRICE_CHANGE = config["filters"]["max_price_change"]
MIN_VOLUME = config["filters"]["min_volume"]
COIN_BLACKLIST = config["blacklist"]["coins"]
DEV_BLACKLIST = config["blacklist"]["devs"]

# Initialize database
def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            name TEXT,
            price REAL,
            liquidity REAL,
            volume REAL,
            market_cap REAL,
            timestamp DATETIME,
            event TEXT,  # e.g., "pump", "rug", "cex_listing", "tier1"
            dev_address TEXT,
            is_fake_volume BOOLEAN,
            rugcheck_status TEXT,
            is_bundled_supply BOOLEAN
        )
    ''')
    conn.commit()
    conn.close()

# Fetch data from DexScreener
def fetch_token_data(token_address):
    response = requests.get(f"{DEXSCREENER_API_URL}{token_address}")
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for token {token_address}")
        return None

# Check if token is blacklisted
def is_blacklisted(token_data):
    symbol = token_data['pairs'][0]['baseToken']['symbol']
    dev_address = token_data['pairs'][0]['baseToken']['address']
    return symbol in COIN_BLACKLIST or dev_address in DEV_BLACKLIST

# Apply filters
def apply_filters(token_data):
    token = token_data['pairs'][0]
    liquidity = float(token['liquidity']['usd'])
    price_change = float(token['priceChange']['h24'])
    volume = float(token['volume']['h24'])
    return liquidity >= MIN_LIQUIDITY and abs(price_change) <= MAX_PRICE_CHANGE and volume >= MIN_VOLUME

# Check for fake volume using Pocket Universe API
def is_fake_volume(token_address):
    headers = {"Authorization": f"Bearer {POCKET_UNIVERSE_API_KEY}"}
    response = requests.get(f"{POCKET_UNIVERSE_API_URL}?token={token_address}", headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result.get("is_fake_volume", False)
    else:
        print(f"Failed to check fake volume for token {token_address}")
        return False

# Check contract status on RugCheck.xyz
def check_rugcheck_status(token_address):
    response = requests.get(f"{RUGCHECK_API_URL}?address={token_address}")
    if response.status_code == 200:
        result = response.json()
        return result.get("status", "Unknown"), result.get("is_bundled_supply", False)
    else:
        print(f"Failed to check RugCheck status for token {token_address}")
        return "Unknown", False

# Send Telegram notification
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        print(f"Failed to send Telegram message: {response.text}")

# Place order via Trojan
def place_trojan_order(token_address, side="buy", amount=0.01):
    headers = {"Authorization": f"Bearer {TROJAN_API_KEY}"}
    payload = {
        "token_address": token_address,
        "side": side,
        "amount": amount
    }
    response = requests.post(TROJAN_API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to place {side} order for token {token_address}: {response.text}")
        return None

# Parse and save data
def save_token_data(token_data):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    token = token_data['pairs'][0]
    symbol = token['baseToken']['symbol']
    name = token['baseToken']['name']
    price = float(token['priceUsd'])
    liquidity = float(token['liquidity']['usd'])
    volume = float(token['volume']['h24'])
    market_cap = float(token['fdv']) if 'fdv' in token else 0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dev_address = token['baseToken']['address']
    is_fake_volume_flag = is_fake_volume(dev_address)  # Check for fake volume
    rugcheck_status, is_bundled_supply = check_rugcheck_status(dev_address)  # Check RugCheck status

    # Detect events
    event = None
    if price > 1000:  # Example: Pump detection
        event = "pump"
    elif liquidity < 1000:  # Example: Rug detection
        event = "rug"
    elif "binance" in token['url']:  # Example: CEX listing detection
        event = "cex_listing"

    # Save to database
    cursor.execute('''
        INSERT INTO tokens (symbol, name, price, liquidity, volume, market_cap, timestamp, event, dev_address, is_fake_volume, rugcheck_status, is_bundled_supply)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, name, price, liquidity, volume, market_cap, timestamp, event, dev_address, is_fake_volume_flag, rugcheck_status, is_bundled_supply))
    conn.commit()
    conn.close()

    # Blacklist if bundled supply
    if is_bundled_supply:
        COIN_BLACKLIST.append(symbol)
        DEV_BLACKLIST.append(dev_address)
        print(f"Blacklisted {symbol} and dev {dev_address} due to bundled supply.")

    # Trade if conditions are met
    if rugcheck_status == "Good" and not is_fake_volume_flag and event == "pump":
        place_trojan_order(dev_address, side="buy", amount=0.01)
        send_telegram_message(f"🚀 BUY {symbol} at {price} USD")

# Analyze data for patterns
def analyze_data():
    conn = sqlite3.connect(DATABASE_NAME)
    df = pd.read_sql_query("SELECT * FROM tokens", conn)
    conn.close()

    # Example: Detect anomalies in price changes
    if not df.empty:
        df['price_change'] = df['price'].pct_change()
        model = IsolationForest(contamination=0.1)
        df['anomaly'] = model.fit_predict(df[['price_change']])
        anomalies = df[df['anomaly'] == -1]
        print("Detected anomalies:")
        print(anomalies)

        # Plot price changes
        plt.figure(figsize=(10, 6))
        plt.plot(df['timestamp'], df['price'], label='Price')
        plt.scatter(anomalies['timestamp'], anomalies['price'], color='red', label='Anomalies')
        plt.legend()
        plt.show()

# Main loop
def main():
    init_db()
    token_addresses = ["0x...", "0x..."]  # Add token addresses to monitor

    while True:
        for address in token_addresses:
            token_data = fetch_token_data(address)
            if token_data and not is_blacklisted(token_data) and apply_filters(token_data):
                rugcheck_status, is_bundled_supply = check_rugcheck_status(address)
                if rugcheck_status == "Good" and not is_bundled_supply:  # Only interact with "Good" contracts
                    if not is_fake_volume(address):  # Skip tokens with fake volume
                        save_token_data(token_data)
        analyze_data()
        time.sleep(UPDATE_INTERVAL)

if __name__ == "__main__":
    main()
