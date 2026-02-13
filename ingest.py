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

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100):
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        if chunk_size <= overlap: break
    return chunks

def ingest_all():
    # Initialize Milvus
    client = MilvusClient(uri=URI)

    # Create collection if not exists
    if not client.has_collection(collection_name=COLLECTION_NAME):
        print(f"[INFO] Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=EMBEDDING_DIM,
            primary_field_name="pk",
            id_type="int",
            vector_field_name="vector",
            metric_type="COSINE",
            auto_id=True
        )
    else:
        print(f"[INFO] Using existing collection: {COLLECTION_NAME}")

    print(f"[INFO] Created Milvus collection: {COLLECTION_NAME}")

    # Load data from text files in 'data' folder
    print("[INFO] Loading data from 'data/' directory...")

    txt_files = glob.glob("data/*.txt")

    if not txt_files:
        print("[WARNING] No text files found in 'data/'")

    data = []

    for file_path in txt_files:
        filename = os.path.basename(file_path)
        try:
            idx = int(os.path.splitext(filename)[0])
        except ValueError:
            print(f"[SKIP] Ignoring non-numeric filename: {filename}")
            continue

        print(f" -> Processing Index {idx} ({filename})")
        
        chunks = chunk_text(text)
        total_chunks = len(chunks)
        if not chunks:
            continue

        print(f"    - Total chunks: {total_chunks}")

        # Batch Embed and Insert in bursts
        BURST_SIZE = 200
        for i in range(0, total_chunks, BURST_SIZE):
            burst_chunks = chunks[i : i + BURST_SIZE]
            
            # Batch Embed this burst
            embeddings = model.encode(burst_chunks).tolist()

            data_to_insert = []
            for j, chunk in enumerate(burst_chunks):
                data_to_insert.append({
                    "course_id": idx,
                    "vector": embeddings[j],
                    "text": chunk,
                    "metadata_source": filename
                })

            # Insert this burst
            client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
            
            completed = min(i + BURST_SIZE, total_chunks)
            print(f"    [PROGRESS] {completed}/{total_chunks} chunks ingested...")
        
        print(f"    [SUCCESS] Finished {filename}")

if __name__ == "__main__":
    ingest_all()

