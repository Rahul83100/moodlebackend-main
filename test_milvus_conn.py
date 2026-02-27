import os
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv()

def test_connection():
    uri = os.getenv("MILVUS_URI", "http://localhost:19530")
    user = os.getenv("MILVUS_USER", "root")
    password = os.getenv("MILVUS_PASSWORD", "Milvus")
    
    print(f"Connecting to {uri} as {user}...")
    try:
        client = MilvusClient(uri=uri, user=user, password=password)
        collections = client.list_collections()
        print(f"Successfully connected! Collections: {collections}")
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_connection()
