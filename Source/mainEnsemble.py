import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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


# --- 1. Data Extraction ---
def fetch_yahoo_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data['Company'] = ticker
    return data


def fetch_nse_data(symbol):
    """
    Fetch live stock data from NSE India.
    """
    nse = Nse()
    quote = nse.get_quote(symbol)
    return pd.DataFrame([quote])


# --- 2. Data Preprocessing ---
def load_data(file_path, live=False):
    """
    Load historical stock data for multiple companies.
    """
    data = pd.read_csv(file_path)  # Assumes CSV file with 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Company'
    return data


def preprocess_data(data, live=False):
    """
    Add technical indicators like SMA, EMA, and RSI for better feature engineering.
    """
    data['SMA_20'] = data.groupby('Company')['Close'].transform(lambda x: x.rolling(window=20).mean())
    data['EMA_20'] = data.groupby('Company')['Close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    data['Daily_Return'] = data.groupby('Company')['Close'].transform(lambda x: x.pct_change())
    data['RSI'] = data.groupby('Company')['Close'].transform(compute_rsi)
    data = data.dropna()  # Remove rows with NaN values
    if live:
        data = data.groupby('Company').tail(1)
    return data


def compute_rsi(series, period=14):
    """
    Compute the Relative Strength Index (RSI) for a given series.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# --- 3. Vector Database Integration with Weaviate ---
def setup_weaviate():
    """
    Set up a Weaviate client and schema.
    """
    client = weaviate.connect_to_local()
    print(client.is_ready())
    # schema = {
    #     "class": "StockData",
    #     "description": "A collection of stock market data for various companies.",
    #     "vectorizer": "text2vec-transformers",
    #     "properties": [
    #         {"name": "Company", "dataType": ["string"], "description": "The company name or ticker symbol."},
    #         {"name": "Features", "dataType": ["string"],
    #          "description": "JSON string of computed technical indicators."},
    #     ],
    # }
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
    """
    Store preprocessed stock data into Weaviate.
    """
    stocks = client.collections.get("StockData")
    for _, row in data.iterrows():
        stocks.data.insert(
            {
                "Company": row['Company'],
                "Features": row[['SMA_20', 'EMA_20', 'Daily_Return', 'RSI', 'Volume']].to_json(),
            }
        )


def retrieve_rag_data(client, company):
    """
    Retrieve relevant data using RAG from Weaviate.
    """
    print(company)
    stocks = client.collections.get("StockData")
    result = stocks.query.bm25(
        query=company,
        filters=Filter.by_property("company").equal(company),
        return_properties=["company", "features"]
    )
    return result


# --- 4. Model Training ---
def train_model(data):
    """
    Train an ensemble model using LightGBM and CatBoost Classifiers to predict Buy/Sell signals.
    """
    loaded_model = 'ensemble_model1.pkl'
    if not os.path.exists(loaded_model):
        print("training")

        # Define the features and target
        features = ['SMA_20', 'EMA_20', 'Daily_Return', 'RSI', 'Volume']
        data['Signal'] = (data['Close'].shift(-1) > data['Close']).astype(int)  # 1 for Buy, 0 for Sell

        X = data[features]
        y = data['Signal']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train LightGBM model
        lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
        lgb_model.fit(X_train, y_train)

        # Train CatBoost model
        cat_model = CatBoostClassifier(iterations=100, random_seed=42, verbose=0)
        cat_model.fit(X_train, y_train)

        # Make predictions with both models
        lgb_predictions = lgb_model.predict_proba(X_test)[:, 1]  # Probability of class 1
        cat_predictions = cat_model.predict_proba(X_test)[:, 1]  # Probability of class 1

        # Ensemble predictions (weighted average)
        ensemble_predictions = (0.5 * lgb_predictions) + (0.5 * cat_predictions)
        final_predictions = (ensemble_predictions >= 0.5).astype(int)  # Threshold at 0.5

        # Evaluate the ensemble model
        accuracy = accuracy_score(y_test, final_predictions)
        print(f"Ensemble Model Accuracy: {accuracy:.2f}")

        # Detailed classification report
        report = classification_report(y_test, final_predictions, target_names=['Sell', 'Buy'])
        print("Classification Report:\n", report)

        # Save the trained models and weights
        ensemble_model = {
            'lgb_model': lgb_model,
            'cat_model': cat_model,
            'weights': [0.5, 0.5]  # Equal weights for now
        }
        joblib.dump(ensemble_model, 'ensemble_model1.pkl')
        return ensemble_model
    else:
        print("loaded")
        return joblib.load('ensemble_model1.pkl')

# --- 5. Real-Time Recommendation ---
def recommend(model, live_data):
    """
    Use the trained ensemble model to recommend Buy/Sell actions for live data.
    """
    features = ['SMA_20', 'EMA_20', 'Daily_Return', 'RSI', 'Volume']
    live_data = preprocess_data(live_data, True)

    # Extract individual models and weights from the ensemble
    lgb_model = model['lgb_model']
    cat_model = model['cat_model']
    weights = model['weights']

    # Get predictions from each model
    lgb_predictions = lgb_model.predict_proba(live_data[features])[:, 1]
    cat_predictions = cat_model.predict_proba(live_data[features])[:, 1]

    # Ensemble predictions (weighted average)
    ensemble_predictions = (weights[0] * lgb_predictions) + (weights[1] * cat_predictions)
    live_data['Prediction'] = (ensemble_predictions >= 0.5).astype(int)  # Threshold at 0.5

    # Add actionable recommendations
    recommendations = live_data[['Company', 'Prediction']]
    recommendations['Action'] = recommendations['Prediction'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
    return recommendations


# --- 6. Streamlit Application ---
# Streamlit UI for user interaction
st.title("Stock Recommendation System")

saved_model = 'ensemble_model1.pkl.pkl'

if not os.path.exists(saved_model):
    uploaded_file = st.file_uploader("Upload Historical Stock Data (CSV)", type="csv")
    if uploaded_file is not None:
        stock_data = load_data(uploaded_file)
        st.write("Loaded Data:", stock_data.head())

        processed_data = preprocess_data(stock_data)
        st.write("Processed Data:", processed_data.head())

        client = setup_weaviate()
        store_data_in_weaviate(client, processed_data)

        model = train_model(processed_data)

        st.write("Model trained successfully! Upload live data for recommendations.")

else:
    model = joblib.load(saved_model)
    st.write("Model loaded successfully! Upload live data for recommendations.")
    client = setup_weaviate()
live_file = st.file_uploader("Upload Live Stock Data (CSV)", type="csv")
if live_file is not None:
    live_data = load_data(live_file)
    print(live_data.dtypes)
    rag_data = retrieve_rag_data(client, live_data['Company'].iloc[0])
    st.write("RAG Data:", rag_data)
    recommendations = recommend(model, live_data)
    st.write("Recommendations:", recommendations)
