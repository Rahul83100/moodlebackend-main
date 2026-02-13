from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import os
from dotenv import load_dotenv

load_dotenv()

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Milvus Config
COLLECTION_NAME = "knowledge_base"
URI = os.getenv("MILVUS_URI", "http://localhost:19530")
client = MilvusClient(uri=URI)

def get_data(user_input: str):
    # 1️⃣ DIRECT COURSE ID LOOKUP
    if user_input.isdigit():
        try:
            results = client.query(
                collection_name=COLLECTION_NAME,
                filter=f"course_id == {user_input}",
                limit=1,
                output_fields=["text"]
            )
            if results:
                return results[0]["text"]
            return "[X] Course ID not found"
        except Exception as e:
            return f"[ERROR] {e}"

    # 2️⃣ SEMANTIC SEARCH
    try:
        query_embedding = model.encode(user_input).tolist()
        query_tokens = user_input.lower().split()

        results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_embedding],
            limit=10, # Retrieve more for re-ranking
            output_fields=["text"]
        )

        if results and results[0]:
            # Re-rank among the 10 candidates using BM25
            chunk_texts = [res['entity']['text'] for res in results[0]]
            tokenized_chunks = [txt.lower().split() for txt in chunk_texts]
            bm25 = BM25Okapi(tokenized_chunks)
            bm25_scores = bm25.get_scores(query_tokens)

            # Find the best hybrid match
            best_idx = 0
            best_hybrid_score = -1
            
            for i, res in enumerate(results[0]):
                dense_score = max(0, res['distance']) * 100
                bm25_score = max(0, bm25_scores[i] * 15)
                hybrid_score = (dense_score * 0.6) + (bm25_score * 0.4)
                
                if hybrid_score > best_hybrid_score:
                    best_hybrid_score = hybrid_score
                    best_idx = i
            
            return results[0][best_idx]["entity"]["text"]
    except Exception as e:
        return f"[ERROR] {e}"
    
    return "[X] No relevant information found."


# -----------------------------
# CLI LOOP
# -----------------------------
print("[SEARCH] Milvus Chunk Search Ready")
print("Enter Course ID or ask a question.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye")
        break

    answer = get_data(user_input)
    print(f"AI: {answer}\n")
