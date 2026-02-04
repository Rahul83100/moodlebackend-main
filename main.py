import os
from pymilvus import MilvusClient
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from dotenv import load_dotenv
import shutil
from ingest import ingest_all

# Initialize App
app = FastAPI()

# Load Environment Variables
load_dotenv(override=True)

# CORS Configuration
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
if allowed_origins_raw == "*":
    origins = ["*"]
else:
    origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
GENAI_KEY = os.getenv("GEMINI_API_KEY")
if GENAI_KEY:
    print(f"[DEBUG] Loaded GEMINI_API_KEY: {GENAI_KEY[:5]}... (Length: {len(GENAI_KEY)})")
    genai.configure(api_key=GENAI_KEY)
else:
    print("[WARNING] GEMINI_API_KEY not found in .env")

# --------------------------
# Load Model & Database
# --------------------------
print("[INFO] Loading Model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("[INFO] Connecting to MilvusDB...")
uri = os.getenv("MILVUS_URI", "http://localhost:19530")
client = MilvusClient(uri=uri)
COLLECTION_NAME = "knowledge_base"

# Ensure collection exists
# Ensure collection exists
if not client.has_collection(COLLECTION_NAME):
    print(f"[INFO] Collection '{COLLECTION_NAME}' not found. Creating it...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=384,
        primary_field_name="id",
        id_type="int",
        vector_field_name="vector",
        metric_type="COSINE",
        auto_id=False
    )
    print(f"[INFO] Created Milvus collection: {COLLECTION_NAME}")

# --------------------------
# Models
# --------------------------
class SessionRequest(BaseModel):
    index_id: int

class ChatRequest(BaseModel):
    index_id: int
    question: str

# --------------------------
# Endpoints
# --------------------------

@app.post("/api/session")
async def start_session(request: SessionRequest):
    """
    Validate index and return initial content for the session.
    """
    try:
        results = client.get(
            collection_name=COLLECTION_NAME,
            ids=[request.index_id]
        )
    except Exception as e:
         print(e)
         results = []

    if not results:
        raise HTTPException(status_code=404, detail="Index not found")

    # Milvus returns a list of dictionaries
    document_text = results[0]["text"]

    return {
        "message": f"Session started for Index {request.index_id}",
        "content": document_text,
        "index_id": request.index_id
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Compare user question with the specific document's content.
    """
    # 1. Get the target document content
    try:
        doc_result = client.get(
            collection_name=COLLECTION_NAME,
            ids=[request.index_id]
        )
    except Exception:
         doc_result = []
         
    if not doc_result:
        # Instead of 404, we return a generic answer to keep the chatbot alive
        return {
            "relevance_score": "0.0%",
            "breakdown": "Dense: 0.0% | BM25: 0.0%",
            "answer": "I have not been trained on this course yet. Please upload course materials.", 
            "context_used": False
        }
    
    document_text = doc_result[0]["text"]

    # 2. Embed the question
    question_embedding = model.encode(request.question).tolist()

    # 3. Query specifically within this document (Dense Vector Search)
    # Filter by ID using expression
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        data=[question_embedding],
        limit=1,
        # Milvus filter expression: "id == <int>"
        filter=f"id == {request.index_id}", 
        output_fields=["text"]
    )

    dense_score = 0.0
    
    if search_result and len(search_result[0]) > 0:
        # Milvus returns distance/score. Metric is COSINE.
        # Cosine distance: 0 (same) to 2 (opposite).
        # We need similarity.
        match = search_result[0][0]
        distance = match['distance'] # With COSINE metric in MilvusClient, this is usually cosine similarity if normalized, or distance. 
        # Default MilvusClient metric is COSINE, which usually returns Cosine Similarity (1.0 is identical).
        # Wait, auto-created index might depend on exact params. 
        # ingest.py used metric_type="COSINE". Milvus returns distance. 
        # For Cosine, smaller distance is better? No, Cosine Similarity is higher better.
        # But Milvus sometimes returns 'distance'.
        
        # In Milvus, 'COSINE' metric usually means Cosine Similarity (larger is better, max 1.0).
        # Let's assume it returns similarity score directly.
        
        dense_score = max(0, distance) * 100



    # 4. Perform BM25 Search (Sparse/Keyword)
    # Tokenize document (simple whitespace tokenizer for now)
    doc_tokens = document_text.lower().split()
    bm25 = BM25Okapi([doc_tokens])
    
    query_tokens = request.question.lower().split()
    bm25_raw_score = bm25.get_scores(query_tokens)[0]
    
    # 5. Hybrid Combination
    bm25_percentage = min(100, (bm25_raw_score * 20))
    
    hybrid_score = (dense_score * 0.7) + (bm25_percentage * 0.3)
    
    
    # 6. Extract Relevant Snippet (Context for LLM)
    sentences = [s.strip() for s in document_text.replace('\n', ' ').split('.') if s.strip()]
    
    if not sentences:
        context_snippet = "No specific context available."
    else:
        # Tokenize sentences for BM25
        sent_tokens = [sent.lower().split() for sent in sentences]
        sent_bm25 = BM25Okapi(sent_tokens)
        
        # Get top 5 sentences for better context
        top_sentences = sent_bm25.get_top_n(query_tokens, sentences, n=5)
        context_snippet = ". ".join(top_sentences) + "."

    # 7. Generate Answer with Gemini LLM
    final_answer = context_snippet # Fallback
    
    if GENAI_KEY:
        try:
            llm_model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"""
            You are a helpful teaching assistant. Answer the user's question based ONLY on the following context.
            If the answer is not in the context, say "I cannot answer this based on the provided course material."
            
            Context:
            {context_snippet}
            
            Question:
            {request.question}
            
            Answer:
            """
            response = llm_model.generate_content(prompt)
            final_answer = response.text
        except Exception as e:
            print(f"[ERROR] Gemini API Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            final_answer = f"Error generating answer. Fallback context: {context_snippet}"

    return {
        "relevance_score": f"{hybrid_score:.1f}%",
        "breakdown": f"Dense: {dense_score:.1f}% | BM25: {bm25_percentage:.1f}%",
        "answer": final_answer, 
        "context_used": True
    }

@app.post("/api/upload")
async def upload_file(index_id: int = Form(...), file: UploadFile = File(...)):
    """
    Receive file, embed on-the-fly, and upsert into Milvus.
    No local storage, no full re-ingestion.
    """
    try:
        # 1. Read file content from memory
        content_bytes = await file.read()
        text_content = content_bytes.decode("utf-8").strip()
        
        if not text_content:
            return JSONResponse(status_code=400, content={"message": "Empty file"})

        print(f"[INFO] Processing upload for Index {index_id} ({file.filename})")

        # 2. Embed content
        embedding = model.encode(text_content).tolist()

        # 3. Prepare data
        data = [{
            "id": index_id,
            "vector": embedding,
            "text": text_content,
            "metadata_source": file.filename
        }]

        # 4. Upsert (Insert/Update) into Milvus
        # Ensure collection exists
        if not client.has_collection(COLLECTION_NAME):
             client.create_collection(
                collection_name=COLLECTION_NAME,
                dimension=384,
                primary_field_name="id",
                id_type="int",
                vector_field_name="vector",
                metric_type="COSINE",
                auto_id=False
            )

        client.upsert(collection_name=COLLECTION_NAME, data=data)
        
        return JSONResponse(status_code=200, content={"message": f"Successfully ingested context for Course {index_id}"})

    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        # Return 500 but also print invalid characters issue if any
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/{path_param}")
async def route_handler(path_param: str):
    """
    Unified entry point to handle:
    1. Session IDs (e.g. /1, /2) -> serve index.html
    2. Static files (e.g. style.css) -> serve file
    """
    # Case 1: Session ID
    if path_param.isdigit():
        return FileResponse("static/index.html")
    
    # Case 2: Static File
    possible_file = os.path.join("static", path_param)
    if os.path.exists(possible_file) and os.path.isfile(possible_file):
        return FileResponse(possible_file)
    
    raise HTTPException(status_code=404, detail="Not Found")
