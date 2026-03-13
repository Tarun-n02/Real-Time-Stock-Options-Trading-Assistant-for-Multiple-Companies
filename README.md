# Real-Time Stock Options Trading Assistant for Multiple Companies

## Overview

This project is a comprehensive real-time stock options trading assistant designed to provide buy/sell recommendations for multiple companies using machine learning models. It integrates historical and live stock data, employs various ML algorithms for prediction, uses a vector database (Weaviate) for retrieval-augmented generation (RAG), and leverages Large Language Models (LLMs) like Llama 3 for generating insightful explanations of recommendations.
The application features a Streamlit-based web interface for user interaction, allowing users to upload data, train models, and receive real-time recommendations with AI-powered explanations.

## Features

- Data Fetching: Retrieve historical data from Yahoo Finance and live data from NSE India.
- Data Preprocessing: Compute technical indicators such as Simple Moving Average (SMA), Exponential Moving Average (EMA), Relative Strength Index (RSI), and daily returns.
- Machine Learning Models: Supports multiple algorithms including Random Forest, CatBoost, LightGBM (LGBM), Ensemble methods, and Deep Learning Regression (DLR).
- Vector Database Integration: Uses Weaviate for storing and retrieving stock data via RAG.
- LLM Explanations: Integrates Llama 3 (via Replicate API) to provide natural language explanations for trading recommendations.
- Web Interface: Streamlit app for easy data upload, model training, and recommendation viewing.
- Docker Deployment: Includes Docker Compose setup for running Weaviate locally.

## Project Structure

```
/
├── README.md
├── requirement.txt
├── Historic_data_3_comanpies_1yr.csv  # Sample historical data
├── Live.csv                          # Sample live data
└── Source/
    ├── docker-compose.yml            # Docker setup for Weaviate
    ├── main.py                       # Main ensemble model with LLM integration
    ├── mainCatboost.py               # CatBoost model script
    ├── mainDLR.py                    # Deep Learning Regression model script
    ├── mainEnsemble.py               # Ensemble model script
    ├── mainLGBM.py                   # LightGBM model script
    ├── mainLLM.py                    # LLM-focused script
    ├── mainRandomForest.py           # Random Forest model script
    └── LLM.py                        # LLM utilities
```

## Installation and Setup

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for Weaviate)
- Replicate API token for Llama 3 (set as environment variable REPLICATE_API_TOKEN)

### Steps

1. Clone the Repository:  
   `git clone <repository-url>`  
   `cd Real-Time-Stock-Options-Trading-Assistant-for-Multiple-Companies`

2. Install Dependencies:  
   `pip install -r requirement.txt`

3. Set Up Weaviate:  
   Navigate to the Source directory.  
   Run Docker Compose to start Weaviate: `docker-compose up -d`  
   Ensure Weaviate is running on http://localhost:8080.

4. Configure API Keys:  
   Obtain a Replicate API token and set it as an environment variable: `export REPLICATE_API_TOKEN=your_api_token_here`

## Usage

### Running the Application

- Main Ensemble with LLM:  
  `streamlit run Source/main.py`

- Individual Models (e.g., CatBoost):  
  `streamlit run Source/mainCatboost.py`

### Workflow

1. Upload Historical Data: Use the Streamlit interface to upload a CSV file with historical stock data (columns: Date, Open, High, Low, Close, Volume, Company).
2. Train Model: The app preprocesses data, stores it in Weaviate, and trains the ML model.
3. Upload Live Data: Provide live stock data CSV.
4. Get Recommendations: Receive buy/sell signals with LLM-generated explanations.

## Data Format

- Historical Data CSV: Should include columns like Date, Open, High, Low, Close, Volume, Company.
- Sample files: Historic_data_3_comanpies_1yr.csv, Live.csv.

## Models

- Random Forest: mainRandomForest.py
- CatBoost: mainCatboost.py
- LightGBM: mainLGBM.py
- Ensemble: mainEnsemble.py (combines multiple models)
- Deep Learning Regression: mainDLR.py
- LLM Integration: mainLLM.py and LLM.py for explanations

Each model script is standalone with its own Streamlit interface.

## Deployment

The project uses Docker Compose for local Weaviate deployment.  
For production, consider deploying Streamlit apps on platforms like Heroku, AWS, or using Docker for the entire application.

## Requirements

See requirement.txt for a full list of Python dependencies, including:  
pandas, numpy  
scikit-learn, catboost, lightgbm  
streamlit  
yfinance, nsetools  
weaviate-client  
replicate

## Disclaimer

This tool is for educational and informational purposes only. It does not constitute financial advice. Always consult with a qualified financial advisor before making investment decisions.