import replicate
import weaviate
import pandas as pd
import json

# --- 4. LLM for Explanations and RAG Retrieval ---

def setup_weaviate():
    """
    Set up a Weaviate client and schema.
    """
    client = weaviate.Client("http://localhost:8080")  # Replace with your Weaviate server address
    if not client.is_ready():
        raise ConnectionError("Weaviate server is not ready.")

    # Ensure schema exists
    schema = {
        "class": "StockData",
        "description": "A collection of stock market data for various companies.",
        "vectorizer": "text2vec-transformers",
        "properties": [
            {"name": "Company", "dataType": ["string"], "description": "The company name or ticker symbol."},
            {"name": "Features", "dataType": ["string"], "description": "JSON string of computed technical indicators."},
        ],
    }
    try:
        client.schema.create_class(schema)
    except weaviate.exceptions.SchemaValidationException:
        print("Schema already exists.")

    return client

def store_data_in_weaviate(client, data):
    """
    Store preprocessed stock data into Weaviate.
    """
    for _, row in data.iterrows():
        client.data_object.create(
            {
                "Company": row['Company'],
                "Features": json.dumps(row[['SMA_20', 'EMA_20', 'Daily_Return', 'RSI', 'Volume']].to_dict()),
            },
            "StockData"
        )

def retrieve_rag_data(client, company):
    """
    Retrieve relevant data using RAG from Weaviate.
    """
    query = {
        "class": "StockData",
        "filter": {
            "path": ["Company"],
            "operator": "Equal",
            "valueText": company
        },
    }
    results = client.data_object.search(query=query)
    return results

def get_llama3_explanation(prompt):
    """
    Generate explanations using the Llama 3 70B model.
    """
    explanation = ""
    for event in replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input={
            "top_k": 0,
            "top_p": 0.9,
            "prompt": prompt,
            "max_tokens": 512,
            "min_tokens": 0,
            "temperature": 0.6,
            "system_prompt": "You are a helpful assistant",
            "length_penalty": 1,
            "stop_sequences": "<|end_of_text|>,<|eot_id|>",
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15,
            "log_performance_metrics": False
        },
    ):
        explanation += str(event)
    return explanation

def generate_recommendation_explanation(company, rag_data):
    """
    Generate a natural language explanation for a recommendation using Llama 3.
    """
    rag_summary = f"Market trends and history for {company}: {rag_data}"
    prompt = (
        f"Provide a detailed recommendation explanation for the company {company} based on the following market data: {rag_summary}"
    )
    return get_llama3_explanation(prompt)

# Example Usage
if __name__ == "__main__":
    client = setup_weaviate()

    # Simulated data for demonstration
    data = pd.DataFrame({
        "Company": ["AAPL", "GOOG"],
        "SMA_20": [150, 2800],
        "EMA_20": [152, 2820],
        "Daily_Return": [0.01, -0.002],
        "RSI": [45, 55],
        "Volume": [1000000, 1500000],
    })

    store_data_in_weaviate(client, data)
    rag_data = retrieve_rag_data(client, "AAPL")
    explanation = generate_recommendation_explanation("AAPL", rag_data)
    print("Recommendation Explanation:", explanation)
