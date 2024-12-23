import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import yfinance as yf
from nsetools import Nse
import weaviate
from weaviate.classes.config import Property, Configure, DataType
from weaviate.classes.query import Filter
import joblib
from catboost import CatBoostClassifier
import lightgbm as lgb
import os
import replicate
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# --- 1. Data Extraction ---
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
    data['SMA_20'] = data.groupby('Company')['Close'].transform(lambda x: x.rolling(window=20).mean())
    data['EMA_20'] = data.groupby('Company')['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    data['Daily_Return'] = data.groupby('Company')['Close'].transform(lambda x: x.pct_change())
    data['RSI'] = data.groupby('Company')['Close'].transform(compute_rsi)
    data = data.dropna()
    if live:
        data = data.groupby('Company').tail(1)
    return data

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- 3. Vector Database Integration with Weaviate ---
def setup_weaviate():
    client = weaviate.connect_to_local()
    print(client.is_ready())
    try:
        client.collections.create(
            "StockData",
            description="A collection of stock market data for various companies.",
            properties=[
                Property(name="Company", data_type=DataType.TEXT),
                Property(name="Features", data_type=DataType.TEXT)
            ],
        )
    finally:
        return client

def store_data_in_weaviate(client, data):
    stocks = client.collections.get("StockData")
    for _, row in data.iterrows():
        stocks.data.insert(
            {
                "Company": row['Company'],
                "Features": row[['SMA_20', 'EMA_20', 'Daily_Return', 'RSI', 'Volume']].to_json(),
            }
        )

def retrieve_rag_data(client, company):
    stocks = client.collections.get("StockData")
    result = stocks.query.bm25(
        query=company,
        filters=Filter.by_property("company").equal(company),
        return_properties=["company", "features"]
    )
    return result

# --- 4. Model Training ---
def train_model(data):
    loaded_model = 'final_model.pkl'
    if not os.path.exists(loaded_model):
        print("Training model...")

        features = ['SMA_20', 'EMA_20', 'Daily_Return', 'RSI', 'Volume']
        data['Signal'] = (data['Close'].shift(-1) > data['Close']).astype(int)

        X = data[features]
        y = data['Signal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        lgb_model.fit(X_train, y_train)

        cat_model = CatBoostClassifier(iterations=100, random_seed=42, verbose=0)
        cat_model.fit(X_train, y_train)

        lgb_predictions = lgb_model.predict_proba(X_test)[:, 1]
        cat_predictions = cat_model.predict_proba(X_test)[:, 1]

        ensemble_predictions = (0.5 * lgb_predictions) + (0.5 * cat_predictions)
        final_predictions = (ensemble_predictions >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, final_predictions)
        print(f"Ensemble Model Accuracy: {accuracy:.2f}")

        report = classification_report(y_test, final_predictions, target_names=['Sell', 'Buy'])
        print("Classification Report:\n", report)

        ensemble_model = {'lgb_model': lgb_model, 'cat_model': cat_model, 'weights': [0.5, 0.5]}
        joblib.dump(ensemble_model, 'final_model.pkl')
        return ensemble_model
    else:
        print("Loading saved model...")
        return joblib.load('final_model.pkl')

# --- 5. Real-Time Recommendation ---
def recommend(model, live_data):
    features = ['SMA_20', 'EMA_20', 'Daily_Return', 'RSI', 'Volume']
    live_data = preprocess_data(live_data, True)
    lgb_model = model['lgb_model']
    cat_model = model['cat_model']
    weights = model['weights']

    lgb_predictions = lgb_model.predict_proba(live_data[features])[:, 1]
    cat_predictions = cat_model.predict_proba(live_data[features])[:, 1]

    ensemble_predictions = (weights[0] * lgb_predictions) + (weights[1] * cat_predictions)
    live_data['Prediction'] = (ensemble_predictions >= 0.5).astype(int)

    recommendations = live_data[['Company', 'Prediction']]
    recommendations['Action'] = recommendations['Prediction'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
    return recommendations

# --- 6. DRL Agent with PPO ---
def setup_trading_env(data):
    class StockTradingEnv(gym.Env):
        def __init__(self, data):
            super(StockTradingEnv, self).__init__()
            self.data = data
            self.current_step = 0
            self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(data.columns),), dtype=np.float32)
        
        def reset(self):
            self.current_step = 0
            return self.data.iloc[self.current_step].values

        def step(self, action):
            self.current_step += 1
            if self.current_step >= len(self.data):
                done = True
                reward = 0
            else:
                done = False
                reward = self.calculate_reward(action)
            return self.data.iloc[self.current_step].values, reward, done, {}

        def calculate_reward(self, action):
            if action == 0:  # Buy
                return self.data.iloc[self.current_step]['Daily_Return']
            elif action == 1:  # Sell
                return -self.data.iloc[self.current_step]['Daily_Return']
            else:  # Hold
                return 0

    return StockTradingEnv(data)

def train_drl_agent(data):
    env = make_vec_env(lambda: setup_trading_env(data), n_envs=1)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_trading_logs/")
    eval_callback = EvalCallback(env, best_model_save_path="./ppo_trading_models/", log_path="./ppo_logs/", eval_freq=500)
    model.learn(total_timesteps=10000, callback=eval_callback)
    model.save("ppo_trading_agent")
    return model

# --- DRL Training and Prediction ---
if st.button("Train DRL Agent"):
    uploaded_file = st.file_uploader("Upload Historical Stock Data (CSV) for DRL", type="csv")
    if uploaded_file is not None:
        stock_data = load_data(uploaded_file)
        processed_data = preprocess_data(stock_data)
        st.write("Training DRL Agent with the provided data...")
        drl_model = train_drl_agent(processed_data)
        st.write("DRL Agent trained successfully!")

if st.button("Predict with DRL Agent"):
    live_file = st.file_uploader("Upload Live Stock Data (CSV) for DRL Prediction", type="csv")
    if live_file is not None:
        live_data = load_data(live_file)
        processed_live_data = preprocess_data(live_data, live=True)
        drl_env = setup_trading_env(processed_live_data)
        drl_model = PPO.load("ppo_trading_agent")
        observation = drl_env.reset()
        predictions = []
        for _ in range(len(processed_live_data)):
            action, _states = drl_model.predict(observation)
            predictions.append(action)
            observation, _, done, _ = drl_env.step(action)
            if done:
                break
        actions = ["Buy" if p == 0 else "Sell" if p == 1 else "Hold" for p in predictions]
        processed_live_data["DRL_Action"] = actions
        st.write("DRL Predictions:", processed_live_data[["Company", "Close", "DRL_Action"]])
