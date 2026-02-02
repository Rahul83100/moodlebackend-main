from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import os
import glob

from dotenv import load_dotenv

load_dotenv()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = 384
COLLECTION_NAME = "knowledge_base"
URI = os.getenv("MILVUS_URI", "http://localhost:19530")

def ingest_all():
    # Initialize Milvus
    client = MilvusClient(uri=URI)

    # Create collection if not exists
    if client.has_collection(collection_name=COLLECTION_NAME):
        client.drop_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=EMBEDDING_DIM,
        primary_field_name="id",
        id_type="int",
        vector_field_name="vector",
        metric_type="COSINE",
        auto_id=False
    )

    print(f"[INFO] Created Milvus collection: {COLLECTION_NAME}")

    # Load data from text files in 'data' folder
    print("[INFO] Loading data from 'data/' directory...")

    txt_files = glob.glob("data/*.txt")

    if not txt_files:
        print("[WARNING] No text files found in 'data/'")

    data = []

    for file_path in txt_files:
        # Extract ID from filename (e.g. "data/1.txt" -> 1)
        filename = os.path.basename(file_path)
        try:
            idx = int(os.path.splitext(filename)[0])
        except ValueError:
            print(f"[SKIP] Ignoring non-numeric filename: {filename}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        
        print(f" -> Processing Index {idx} ({filename})")
        
        embedding = model.encode(text).tolist()

        data.append({
            "id": idx,
            "vector": embedding,
            "text": text,
            "metadata_source": filename
        })

    if data:
        client.insert(collection_name=COLLECTION_NAME, data=data)
        print(f"[SUCCESS] Inserted {len(data)} documents into Milvus.")
    else:
        print("[WARNING] No data to insert.")

if __name__ == "__main__":
    ingest_all()

