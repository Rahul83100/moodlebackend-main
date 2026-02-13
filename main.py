import os
from pymilvus import MilvusClient
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.generativeai as genai
from dotenv import load_dotenv
import shutil
from ingest import ingest_all
from typing import List

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
if not client.has_collection(COLLECTION_NAME):
    print(f"[INFO] Collection '{COLLECTION_NAME}' not found. Creating it...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=384,
        primary_field_name="pk",
        id_type="int",
        vector_field_name="vector",
        metric_type="COSINE",
        auto_id=True
    )
    print(f"[INFO] Created Milvus collection: {COLLECTION_NAME}")
else:
    print(f"[INFO] Using existing Milvus collection: {COLLECTION_NAME}")

# --------------------------
# Utilities
# --------------------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks for better RAG retrieval.
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        
        # Prevent infinite loop if overlap >= chunk_size
        if chunk_size <= overlap:
            break
            
    return chunks

# --------------------------
# Models
# --------------------------
class SessionRequest(BaseModel):
    index_id: int
    session_id: str = "default"

class ChatRequest(BaseModel):
    index_id: int
    question: str
    session_id: str = "default"

# Global In-Memory Session History
# Structure: { session_id: [ {"role": "user", "parts": [...]}, {"role": "model", "parts": [...]} ] }
SESSION_HISTORY = {}

# --------------------------
# Endpoints
# --------------------------

@app.post("/api/session")
async def start_session(request: SessionRequest):
    """
    Validate index and return initial content for the session.
    """
    try:
        # We now query by course_id because one index can have many chunks
        results = client.query(
            collection_name=COLLECTION_NAME,
            filter=f"course_id == {request.index_id}",
            limit=1,
            output_fields=["text"]
        )
    except Exception as e:
         print(f"[ERROR] Session lookup failed: {e}")
         results = []

    if not results:
        raise HTTPException(status_code=404, detail="Course data not found. Please upload materials first.")

    # Return the first chunk text as a sample/initial content
    document_text = results[0]["text"]

    return {
        "message": f"Session started for Course ID {request.index_id}",
        "content": document_text,
        "index_id": request.index_id
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Joint Hybrid Search: 
    1. Retrieve 20 candidates via Vector Search.
    2. Re-rank them using BM25 keyword scores.
    3. Pick the top 5 best-performing chunks for Gemini.
    """
    # 1. Embed the question
    question_embedding = model.encode(request.question).tolist()
    query_tokens = request.question.lower().split()

    # 2. Query a wider window (20 chunks) for re-ranking
    try:
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            data=[question_embedding],
            limit=20, # Wider window for re-ranking
            filter=f"course_id == {request.index_id}", 
            output_fields=["text"]
        )
    except Exception as e:
        print(f"[ERROR] Milvus Search Error: {e}")
        search_results = [[]]

    if not search_results or not search_results[0]:
        return {
            "relevance_score": "0.0%",
            "breakdown": "Dense: 0.0% | BM25: 0.0%",
            "answer": "I have not been trained on this course yet. Please upload course materials.", 
            "context_used": False
        }

    # 3. Re-Ranking Logic
    candidates = []
    chunk_texts = [res['entity']['text'] for res in search_results[0]]
    
    # Initialize BM25 on the candidates
    tokenized_chunks = [txt.lower().split() for txt in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)
    bm25_scores = bm25.get_scores(query_tokens)

    for i, res in enumerate(search_results[0]):
        dense_score = max(0, res['distance']) * 100
        # Normalize BM25 (simplified normalization for re-ranking)
        raw_bm25 = bm25_scores[i]
        bm25_percentage = max(0, min(100, (raw_bm25 * 15))) # Adjusted multiplier
        
        hybrid_score = (dense_score * 0.6) + (bm25_percentage * 0.4) # Slightly more BM25 weight
        
        candidates.append({
            "text": res['entity']['text'],
            "dense": dense_score,
            "bm25": bm25_percentage,
            "hybrid": hybrid_score
        })

    # Sort by hybrid score and take top 5
    candidates.sort(key=lambda x: x['hybrid'], reverse=True)
    top_candidates = candidates[:5]

    # Calculate final display scores (average of top 5)
    avg_hybrid = sum(c['hybrid'] for c in top_candidates) / len(top_candidates)
    avg_dense = sum(c['dense'] for c in top_candidates) / len(top_candidates)
    avg_bm25 = sum(c['bm25'] for c in top_candidates) / len(top_candidates)

    combined_context_text = "\n\n".join([c['text'] for c in top_candidates])
    context_snippet = combined_context_text

    # 7. Manage Conversation History
    session_id = request.session_id
    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = []
    
    # Get last 6 messages for context (3 turns)
    history = SESSION_HISTORY[session_id][-6:]
    history_text = ""
    for msg in history:
        role = "User" if msg["role"] == "user" else "AI"
        history_text += f"{role}: {msg['parts'][0]}\n"

    # 8. Generate Answer with Gemini LLM
    final_answer = context_snippet # Fallback
    
    if GENAI_KEY:
        try:
            llm_model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"""
            Identity: You are "Christ Chatbot", an intelligent teaching assistant for Christ (Deemed to be University).
            
            Conversation History (for context):
            {history_text}
            
            Instructions:
            1. If the user's input is a GREETING (e.g., "hi", "hello", "who are you?"), greet them as Christ Chatbot and provide a one-sentence summary of the topics you can speak on based on the provided context.
            2. If the user's input is a SPECIFIC QUESTION about the course material, answer it DIRECTLY and concisely without introducing yourself or repeating your identity.
            3. Use the Conversation History to handle follow-up questions (e.g., "tell me more" or "who was that?").
            4. Do NOT mention numerical course IDs or system IDs.
            5. If the question is "what is this about?", provide a focused summary of the themes found in the context.
            6. If the context is empty/unrelated, say: "I don't have enough specific material to answer that accurately yet. Please check back as new content is uploaded!"
            
            Available Context:
            {context_snippet}
            
            Current User Question:
            {request.question}
            
            Answer:
            """
            response = llm_model.generate_content(prompt)
            final_answer = response.text

            # Update Session History
            SESSION_HISTORY[session_id].append({"role": "user", "parts": [request.question]})
            SESSION_HISTORY[session_id].append({"role": "model", "parts": [final_answer]})
            
            # Keep history manageable (last 20 messages)
            if len(SESSION_HISTORY[session_id]) > 20:
                SESSION_HISTORY[session_id] = SESSION_HISTORY[session_id][-20:]

        except Exception as e:
            print(f"[ERROR] Gemini API Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            final_answer = f"Error generating answer. Fallback context: {context_snippet}"

    return {
        "relevance_score": f"{avg_hybrid:.1f}%",
        "breakdown": f"Dense: {avg_dense:.1f}% | BM25: {avg_bm25:.1f}%",
        "answer": final_answer, 
        "context_used": True
    }

def process_and_ingest(index_id: int, text_content: str, filename: str):
    """
    Background worker to chunk, batch embed, and insert data in bursts.
    Handles massive files without high memory pressure.
    """
    try:
        print(f"\n[BG-TASK START] Course: {index_id} | File: {filename}")
        
        # 1. Create chunks
        chunks = chunk_text(text_content)
        total_chunks = len(chunks)
        if not chunks:
            print(f"[BG-TASK] No chunks created for {filename}")
            return

        print(f"[BG-TASK] Total chunks to process: {total_chunks}")

        # 2. Process in Bursts (Streaming)
        BURST_SIZE = 200
        for i in range(0, total_chunks, BURST_SIZE):
            burst_chunks = chunks[i : i + BURST_SIZE]
            
            # Batch Embed this burst
            embeddings = model.encode(burst_chunks).tolist()

            # Prepare data for this burst
            data = []
            for j, chunk in enumerate(burst_chunks):
                data.append({
                    "course_id": index_id,
                    "vector": embeddings[j],
                    "text": chunk,
                    "metadata_source": filename
                })

            # Immediate Insert for this burst
            # (Note: client.insert itself can take data in smaller internal batches if needed)
            client.insert(collection_name=COLLECTION_NAME, data=data)
            
            # Log Progress
            completed = min(i + BURST_SIZE, total_chunks)
            percent = (completed / total_chunks) * 100
            print(f"[BG-TASK PROGRESS] {filename}: {completed}/{total_chunks} chunks ({percent:.1f}%)")

        print(f"[BG-TASK SUCCESS] Finished ingestion for {filename} ({total_chunks} chunks)\n")

    except Exception as e:
        print(f"[BG-TASK ERROR] Failed to process {filename}: {e}")
        import traceback
        traceback.print_exc()

@app.post("/api/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    index_id: int = Form(...), 
    file: UploadFile = File(...)
):
    """
    Receive file and trigger background ingestion.
    """
    try:
        content_bytes = await file.read()
        text_content = content_bytes.decode("utf-8").strip()
        
        if not text_content:
            return JSONResponse(status_code=400, content={"message": "Empty file"})

        # Trigger background task
        background_tasks.add_task(process_and_ingest, index_id, text_content, file.filename)
        
        return JSONResponse(status_code=202, content={
            "message": f"Upload received. Processing file in the background for course {index_id}."
        })

    except Exception as e:
        print(f"[ERROR] Upload request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verify-password")
async def verify_password(password: str = Form(...)):
    """
    Verify the admin password.
    """
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_password or password != admin_password:
        raise HTTPException(status_code=403, detail="Invalid admin password")
    return {"status": "success"}

@app.post("/api/bulk-upload")
async def bulk_upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    password: str = Form(...)
):
    """
    Receive multiple files and trigger background ingestion for each.
    """
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_password or password != admin_password:
        raise HTTPException(status_code=403, detail="Invalid admin password")

    summary = []
    for file in files:
        file_name = file.filename
        try:
            base_name = os.path.splitext(file_name)[0]
            index_id = int(base_name)
            
            content_bytes = await file.read()
            text_content = content_bytes.decode("utf-8").strip()
            
            if not text_content:
                summary.append({"filename": file_name, "status": "failed", "reason": "Empty file"})
                continue

            # Queue background task
            background_tasks.add_task(process_and_ingest, index_id, text_content, file_name)
            summary.append({"filename": file_name, "course_id": index_id, "status": "queued"})

        except ValueError:
            summary.append({"filename": file_name, "status": "skipped", "reason": "Filename must be numeric ID"})
        except Exception as e:
            summary.append({"filename": file_name, "status": "failed", "reason": str(e)})

    return JSONResponse(status_code=202, content={
        "message": "Bulk upload received. Processing files in the background.",
        "details": summary
    })


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
        return FileResponse("static/chat.html")
    
    # Case 2: Static File
    possible_file = os.path.join("static", path_param)
    if os.path.exists(possible_file) and os.path.isfile(possible_file):
        return FileResponse(possible_file)
    
    raise HTTPException(status_code=404, detail="Not Found")
