import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import streamlit as st
import numpy as np 

# Initialize Qdrant client
qdrant_client = QdrantClient("http://localhost:6333")

# Define collection name
collection_name = "vector_128_float32"

# Check if the collection exists
if not qdrant_client.collection_exists(collection_name):
    raise ValueError(f"Collection '{collection_name}' does not exist. Please run the vectorization script first.")

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Reduce vector size to 128
def get_128_dim_vector(text):
    original_vector = embedding_model.encode(text, convert_to_numpy=True)
    if len(original_vector) > 128:
        return original_vector[:128].astype(np.float32)
    return original_vector.astype(np.float32)

# Define search function
def search(query, top_n=5):
    query_vector = get_128_dim_vector(query)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_n,
    )
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
    with st.spinner("Searching for matches..."):
        if query.strip():
            results = search(query)
            if not results.empty:
                st.success("Search completed! Here are the top matches:")
                st.table(results)
            else:
                st.warning("No matches found for your query.")
        else:
            st.warning("Please enter a valid query.")
