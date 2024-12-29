import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from catboost import CatBoostClassifier
import lightgbm as lgb
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import lightgbm as lgb
import streamlit as st


def fetch_yahoo_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data['Company'] = ticker
    return data


def fetch_nse_data(symbol):
    nse = Nse()
    quote = nse.get_quote(symbol)
    return pd.DataFrame([quote])


# --- 2. Data Preprocessing ---
def load_data(file_path, live=False):
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data, live=False):
    """
    Preprocess the data to include technical indicators and ensure numerical values only.
    """
    # Map company names to integers
    company_mapping = {"INFY": 0, "RELIANCE": 1, "TCS": 2}
    data['Company'] = data['Company'].map(company_mapping)
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    if live:
        # Only use the last 40 days of data for calculations
        data = data.groupby("Company").apply(lambda x: x.iloc[-40:]).reset_index(drop=True)

        data['SMA_20'] = data.groupby('Company')['Close'].transform(lambda x: x.rolling(window=20).mean())
        data['EMA_20'] = data.groupby('Company')['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
        data['Daily_Return'] = data.groupby('Company')['Close'].transform(lambda x: x.pct_change())
        data['RSI'] = data.groupby('Company')['Close'].transform(compute_rsi)
        data = data.dropna()  # Remove rows with NaN values
        if live:
            data = data.groupby('Company').tail(1)
        return data

    else:
        # Ensure numeric columns and parse dates
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')
        # Technical indicators
        data['SMA_20'] = data.groupby('Company')['Close'].transform(lambda x: x.rolling(window=20).mean())
        data['EMA_20'] = data.groupby('Company')['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
        data['Daily_Return'] = data.groupby('Company')['Close'].transform(lambda x: x.pct_change())
        data['RSI'] = data.groupby('Company')['Close'].transform(compute_rsi)

        # Drop rows with NaN values after calculations
        data = data.dropna()

        # Drop non-numeric columns (e.g., `Date`) and reset index
        return data.drop(columns=['Date'], errors='ignore').reset_index(drop=True)


def compute_rsi(series, period=14):
    """
    Calculate the Relative Strength Index (RSI) for a given series.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def adjust_date_range(start_date, end_date):
    """
    Ensure there is a minimum of 40 days between start_date and end_date.
    If not, automatically adjust temp_start_date to 40 days before end_date.
    """
    temp_start_date = start_date  # Default to user-defined start_date
    if (end_date - start_date).days < 40:
        temp_start_date = end_date - timedelta(days=40)
    return temp_start_date, end_date


class StockTradingEnv(gym.Env):
    """
    Custom Stock Trading Environment for PPO.
    """

    def __init__(self, data, initial_balance=100000):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.current_step = 0
        self.num_shares = 0
        self.max_steps = len(data) - 1

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns),), dtype=np.float32)

    def reset(self):
        self.current_balance = self.initial_balance
        self.current_step = 0
        self.num_shares = 0
        return self.data.iloc[self.current_step].values

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']

        if action == 0:  # Buy
            self.num_shares += self.current_balance // current_price
            self.current_balance %= current_price
        elif action == 1:  # Sell
            self.current_balance += self.num_shares * current_price
            self.num_shares = 0

        self.current_step += 1
        done = self.current_step >= self.max_steps
        total_value = self.current_balance + (self.num_shares * current_price)
        reward = total_value - self.initial_balance
        next_state = self.data.iloc[self.current_step].values if not done else np.zeros_like(self.data.iloc[0].values)
        return next_state, reward, done, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.current_balance}, Shares: {self.num_shares}")

def train_drl_agent(data):
    env = make_vec_env(lambda: StockTradingEnv(data), n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs/", device="cpu")  # Force CPU usage
    eval_callback = EvalCallback(env, best_model_save_path="./ppo_models/", log_path="./ppo_logs/", eval_freq=500)
    model.learn(total_timesteps=10000, callback=eval_callback)
    model.save("ppo_stock_trading_agent")
    return model


def use_trained_ppo_model(processed_live_data):
    """
    Use the trained PPO agent to make predictions on live data.
    """
    # Remove the signal (target) column from the live data
    processed_live_data.loc[:, 'Signal'] = (processed_live_data['Close'].shift(-1) > processed_live_data['Close']).astype(int)
    processed_live_data = processed_live_data.iloc[:, 1:]

    # Ensure all features are numerical and have the expected shape
    # print(processed_live_data.info())

    # Setup the trading environment with live data
    env = StockTradingEnv(processed_live_data)

    # Load the trained PPO model
    model = PPO.load("ppo_stock_trading_agent")

    # Run predictions
    obs = env.reset()
    actions = []
    for _ in range(len(processed_live_data)):  # Exclude the last step as it will be handled separately
        action, _ = model.predict(obs)  # Predict the action (Buy=0, Sell=1, Hold=2)
        actions.append(action)  # Append only the action, not the entire output
        obs, _, done, _ = env.step(action)
        # if done:
        #     break

    # Map actions to human-readable format
    # print(actions)
    action_mapping = {0: "Buy", 1: "Sell", 2: "Hold"}
    processed_live_data['Action'] = [action_mapping[int(a)] for a in actions]

    # Return the processed live data with actions
    return processed_live_data[['Close', 'Action']]


# Example usage
# --- 6. Streamlit Application ---
st.title("Stock Recommendation System")

# User input for date range
start_date = st.date_input("Select Start Date", datetime(2024, 12, 1))
end_date = st.date_input("Select End Date", datetime.now())

# Verify and adjust date range
temp_start_date, adjusted_end_date = adjust_date_range(start_date, end_date)

# Display information about the adjusted date range
st.write(f"Data will be fetched for the range: {temp_start_date} to {adjusted_end_date}")
st.write(f"Original user-defined range: {start_date} to {end_date}")

# Tickers
tickers = st.text_input("Enter Ticker Symbols (comma-separated)", "INFY")
tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

# Fetch data and train model
if st.button("Fetch Data"):
    # Fetch data using the adjusted date range
    stock_data = pd.concat(
        [fetch_yahoo_data(ticker, temp_start_date, adjusted_end_date) for ticker in tickers], axis=0
    )
    if stock_data.empty:
        st.write("No data fetched. Please check ticker symbols or date range.")
    else:
        st.write("Fetched Data:", stock_data.head())

        # Preprocess and calculate indicators
        processed_data = preprocess_data(stock_data, True)
        st.write("Processed Data with Indicators:", processed_data)

        # Store calculated indicators
        # indicators = processed_data[['Company', 'SMA_20', 'EMA_20', 'Daily_Return', 'RSI']]
        # st.write("Technical Indicators:", indicators.head())

        # Processed data for selected dates
        # Slice data based on user-provided date range
        user_processed_data = processed_data[
            (processed_data['Date'] >= pd.to_datetime(start_date)) &
            (processed_data['Date'] <= pd.to_datetime(end_date))
            ]
        st.write("User Processed Data:", user_processed_data)

        print("USER", user_processed_data.iloc[0:1])

        # Predict
        recommendation = use_trained_ppo_model(user_processed_data)

        st.write("Recommendation:", recommendation)
