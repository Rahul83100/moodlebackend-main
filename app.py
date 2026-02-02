import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Chroma DB
# Load Chroma DB
client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = client.get_collection(name="knowledge_base")
except Exception as e:
    print("\n[ERROR] Collection 'knowledge_base' not found.")
    print("-> Please run 'python ingest.py' first to initialize the database.\n")
    exit(1)


def get_data(user_input: str):

    # 1️⃣ DIRECT INDEX LOOKUP
    if user_input.isdigit():
        result = collection.get(
            where={"id": int(user_input)}
        )

        if result["documents"]:
            return result["documents"][0]
        return "[X] Index not found"

    # 2️⃣ SEMANTIC SEARCH
    query_embedding = model.encode(user_input).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1
    )

    return results["documents"][0][0]


# -----------------------------
# CLI LOOP
# -----------------------------
print("[SEARCH] Chroma Vector Search Ready")
print("Enter index number or ask a question.")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye")
        break

    answer = get_data(user_input)
    print(f"AI: {answer}\n")
