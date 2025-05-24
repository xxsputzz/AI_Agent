"""
FastAPI Web Server for the Document AI Agent
This module provides a REST API for interacting with the Document AI Agent.
"""
import os
import json
import time
import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

# Import the agent functionality
from document_loader_agent import ai_agent
from documents import get_relevant_documents, index_documents, doc_collection
from history import get_relevant_history

# Create the FastAPI app
app = FastAPI(
    title="Document AI Agent API",
    description="API for interacting with an AI agent that answers questions based on documents",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Mount the static files directory once
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(static_dir) and not any(route.name == "static" for route in app.routes):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    print(f"Static files mounted from {static_dir}")
else:
    print(f"Note: Static files already mounted or directory not found at {static_dir}")

# Dictionary to store ongoing conversations
conversations = {}

# Define data models
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = "default_user"
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    conversation_id: str
    user_id: str
    timestamp: float
    relevant_docs: List[str]
    query: str

class IndexRequest(BaseModel):
    reload: bool = False

class IndexResponse(BaseModel):
    status: str
    document_count: int
    message: str

class HealthResponse(BaseModel):
    status: str
    version: str
    documents_indexed: int

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health of the API"""
    from documents import doc_collection
    doc_count = doc_collection.count() if doc_collection else 0
    
    return {
        "status": "ok",
        "version": "1.0.0",
        "documents_indexed": doc_count
    }

# Endpoint to index documents
@app.post("/index", response_model=IndexResponse)
async def index_docs(request: IndexRequest, background_tasks: BackgroundTasks):
    """Index or re-index documents in the document store"""
    from document_loader_agent import doc_collection
    
    if request.reload:
        # Clear the collection if it exists
        if doc_collection:
            doc_collection.delete(where={})
    
    # Run indexing in the background
    background_tasks.add_task(index_documents)
    
    # Return immediately with a status
    return {
        "status": "indexing_started",
        "document_count": 0,
        "message": "Document indexing has started in the background"
    }

# Endpoint to query the agent
@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Query the AI agent with a question"""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Generate conversation ID if not provided
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    try:
        # Get response from agent
        response = ai_agent(request.query, request.user_id)
        
        # Get relevant documents for context
        relevant_docs, _ = get_relevant_documents(request.query)
        
        # Create response object
        timestamp = time.time()
        response_obj = {
            "response": response,
            "conversation_id": conversation_id,
            "user_id": request.user_id,
            "timestamp": timestamp,
            "relevant_docs": relevant_docs[:2] if relevant_docs else [],  # Just return top 2 for brevity
            "query": request.query
        }
        
        # Store conversation for history
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        conversations[conversation_id].append({
            "query": request.query,
            "response": response,
            "timestamp": timestamp,
            "user_id": request.user_id
        })
        
        return response_obj
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Endpoint to get conversation history
@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get the history of a conversation"""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"conversation_id": conversation_id, "history": conversations[conversation_id]}

# Endpoint to list available conversations
@app.get("/conversations")
async def list_conversations():
    """List all active conversations"""
    return {
        "conversations": [
            {
                "id": conv_id,
                "messages": len(history),
                "last_updated": history[-1]["timestamp"] if history else 0
            }
            for conv_id, history in conversations.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    import socket
    
    # Mount the static files directory
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    else:
        print(f"Warning: Static directory not found at {static_dir}")
    
    # Try ports in sequence until we find an available one
    ports = [8000, 8001, 8002, 8003]
    
    for port in ports:
        try:
            # Test if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                s.close()
            
            print(f"Starting server on port {port}")
            uvicorn.run("agent_api:app", host="0.0.0.0", port=port, reload=True)
            break
        except OSError as e:
            if port == ports[-1]:
                print(f"Error: All ports {ports} are in use. Please free up one of these ports and try again.")
                raise e
            print(f"Port {port} is in use, trying next port...")
