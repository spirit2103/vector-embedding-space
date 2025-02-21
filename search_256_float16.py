import numpy as np
import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize Qdrant client in memory
qdrant_client = QdrantClient("http://localhost:6333")

# Collection name
collection_name = "vector_256_float16"  # Update collection name to reflect new size

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Outputs 384-dimensional vectors

# Generate a 256-dimensional vector from text
def get_256_dim_vector(text):
    """
    Generate a 256-dimensional vector by truncating or padding the original vector.
    """
    original_vector = embedding_model.encode(text, convert_to_numpy=True)
    if len(original_vector) >= 256:
        return original_vector[:256].astype(np.float16)  # Truncate to 256 dimensions
    else:
        # Pad with zeros if the vector is smaller than 256
        padded_vector = np.zeros(256, dtype=np.float16)
        padded_vector[:len(original_vector)] = original_vector
        return padded_vector

# Define search function
def search(query, top_n=5):
    """
    Search for the most similar entries in the Qdrant collection based on a query.

    :param query: The search query (name and address combined).
    :param top_n: Number of top matches to return.
    :return: DataFrame with top matches and their similarity scores.
    """
    query_vector = get_256_dim_vector(query)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_n,
    )
    # Convert search results into a DataFrame
    results = pd.DataFrame([
        {"name": hit.payload['name'], "address": hit.payload['address'], "score": hit.score * 100}
        for hit in search_result
    ])
    results.index = range(1, len(results) + 1)
    return results

# Streamlit App
st.title("Address Search with Qdrant")
st.write("Enter an address or name below to find the most similar entries in the database.")

# Input field
query = st.text_input("Enter your query:", "")

# Search button
if st.button("Search"):
    with st.spinner("Please wait while we process your request..."):
        if query.strip():
            results = search(query)
            if not results.empty:
                st.success("Search completed! Here are the top matches:")
                st.table(results)
            else:
                st.warning("No matches found for your query.")
        else:
            st.warning("Please enter a valid query.")
