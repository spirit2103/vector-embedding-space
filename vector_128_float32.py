import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import json

# Initialize Qdrant client (use persistent server)
qdrant_client = QdrantClient("http://localhost:6333")

# Define collection name
collection_name = "vector_128_float32"

# Create the collection if it doesn't exist
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=128, distance=Distance.COSINE),
    )
print(f"Collection '{collection_name}' is ready.")

# Load and clean the dataset
file_path = r"C:\Users\susha\internship\ecourt_1_all_splitzbwa.csv"
raw_data = pd.read_csv(file_path, sep='}', engine='python')

# Parse JSON rows
def parse_json(row):
    try:
        json_data = json.loads(row + '}')
        return pd.Series([json_data.get('name'), json_data.get('address')])
    except Exception:
        return pd.Series([None, None])

# Process data
cleaned_data = raw_data.iloc[:, 0].apply(parse_json)
cleaned_data.columns = ['name', 'address']
cleaned_data['combined'] = cleaned_data['name'] + " " + cleaned_data['address']

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Reduce vector size to 128
def get_128_dim_vector(text):
    original_vector = embedding_model.encode(text, convert_to_numpy=True)
    if len(original_vector) > 128:
        return original_vector[:128].astype(np.float32)
    return original_vector.astype(np.float32)

# Generate and insert vector embeddings
points = []
for i, row in cleaned_data.iterrows():
    if pd.notna(row['combined']):
        vector = get_128_dim_vector(row['combined'])
        payload = {"name": row['name'], "address": row['address']}
        points.append(PointStruct(id=i, vector=vector, payload=payload))

# Upload data to Qdrant
qdrant_client.upsert(collection_name=collection_name, points=points)
print("Data successfully inserted into Qdrant.")
