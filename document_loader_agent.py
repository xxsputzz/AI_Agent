"""
Document Loader Agent - An AI agent that loads documents from files and uses them for context
"""
import os
import json
import time
import requests
import numpy as np
import chromadb
from datetime import datetime
from sentence_transformers import SentenceTransformer
import PyPDF2
import re

print("Starting Document Loader Agent...")

# Constants and settings
DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
MODEL_NAME = "tinyllama:latest"  # The Ollama model to use
MAX_CHUNK_SIZE = 1000  # Maximum characters in a document chunk
MAX_CHUNKS_CONTEXT = 3  # Maximum chunks to include in context
USER_ID = "default_user"  # Default user ID (could be IP or username)

# Initialize embedding model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Successfully loaded SentenceTransformer model")
except Exception as e:
    print(f"✗ Error loading SentenceTransformer model: {e}")
    model = None

# Initialize ChromaDB
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    # Collection for document chunks
    doc_collection = client.get_or_create_collection(name="document_chunks")
    # Collection for conversation history
    history_collection = client.get_or_create_collection(name="conversation_history")
    print(f"✓ Connected to ChromaDB at {CHROMA_DB_DIR}")
except Exception as e:
    print(f"✗ Error connecting to ChromaDB: {e}")
    client = None
    doc_collection = None
    history_collection = None

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF {file_path}: {e}")
        return f"Error extracting text from {os.path.basename(file_path)}"

def read_file(file_path):
    """Read text from a file based on its extension"""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return extract_text_from_pdf(file_path)
        
        # For text-based files (txt, md, etc.)
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return f"Error reading {os.path.basename(file_path)}"

def chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE, overlap=100):
    """Split text into overlapping chunks of approximately max_chunk_size characters"""
    chunks = []
    
    # Check if text needs to be chunked (if it's short enough, keep it whole)
    if len(text) <= max_chunk_size:
        chunks.append(text)
        return chunks
    
    # Split on paragraph breaks first
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_chunk = ""
    for paragraph in paragraphs:
        # If adding this paragraph would exceed max size, store current chunk and start a new one
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk)
            # Start new chunk with overlap from the end of the previous chunk
            current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
        
        # Add paragraph to current chunk
        if current_chunk:
            current_chunk += "\n\n" + paragraph
        else:
            current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def load_documents_from_directory(directory=DOCUMENTS_DIR, supported_extensions=['.txt', '.md', '.pdf']):
    """Load and chunk documents from the specified directory"""
    document_chunks = []
    metadata_list = []
    
    if not os.path.exists(directory):
        print(f"✗ Documents directory not found: {directory}")
        return document_chunks, metadata_list
    
    print(f"Loading documents from {directory}...")
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                file_path = os.path.join(root, file)
                try:
                    # Get relative path for display
                    rel_path = os.path.relpath(file_path, start=os.path.dirname(directory))
                    
                    # Read the file content
                    content = read_file(file_path)
                    
                    # Chunk the content
                    chunks = chunk_text(content)
                    
                    # Add each chunk with metadata
                    for i, chunk in enumerate(chunks):
                        document_chunks.append(chunk)
                        metadata_list.append({
                            "source": rel_path,
                            "chunk": i+1,
                            "total_chunks": len(chunks),
                            "created_at": os.path.getctime(file_path),
                            "type": os.path.splitext(file)[1][1:].lower()
                        })
                    
                    print(f"  ✓ Loaded {file} ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"  ✗ Error processing {file}: {e}")
    
    return document_chunks, metadata_list

def index_documents():
    """Load documents and index them in ChromaDB"""
    document_chunks, metadata_list = load_documents_from_directory()
    
    if not document_chunks:
        print("No documents found to index.")
        return False
    
    if not model or not doc_collection:
        print("Cannot index documents without embedding model or database.")
        return False
    
    try:
        # Get IDs for all chunks
        ids = [f"chunk_{i}" for i in range(len(document_chunks))]
        
        # Convert metadata to strings for ChromaDB
        str_metadata = []
        for meta in metadata_list:
            str_meta = {k: str(v) for k, v in meta.items()}
            str_metadata.append(str_meta)
        
        # Create embeddings for all chunks
        embeddings = []
        batch_size = 10  # Process in batches to avoid memory issues
        for i in range(0, len(document_chunks), batch_size):
            batch = document_chunks[i:i+batch_size]
            batch_embeddings = model.encode(batch)
            embeddings.extend(batch_embeddings.tolist())
            print(f"  ✓ Embedded chunks {i+1}-{i+len(batch)} of {len(document_chunks)}")
        
        # Add to ChromaDB
        doc_collection.add(
            embeddings=embeddings,
            documents=document_chunks,
            metadatas=str_metadata,
            ids=ids
        )
        
        print(f"✓ Indexed {len(document_chunks)} document chunks")
        return True
    except Exception as e:
        print(f"✗ Error indexing documents: {e}")
        return False

def add_to_history(user_id, query, response):
    """Add a conversation turn to the history"""
    if not model or not history_collection:
        return
    
    try:
        # Create a unique ID for this conversation turn
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        turn_id = f"turn_{user_id}_{timestamp}"
        
        # Create the document to store (query + response)
        document = f"User: {query}\nAssistant: {response}"
        
        # Create embedding
        embedding = model.encode(document).tolist()
        
        # Add to history collection
        history_collection.add(
            embeddings=[embedding],
            documents=[document],
            metadatas=[{
                "user_id": user_id,
                "timestamp": timestamp,
                "query": query
            }],
            ids=[turn_id]
        )
    except Exception as e:
        print(f"✗ Error adding to history: {e}")

def get_relevant_history(user_id, query, n_results=2):
    """Get relevant conversation history for the current user and query"""
    if not model or not history_collection:
        return []
    
    try:
        # Create query embedding
        query_embedding = model.encode(query).tolist()
        
        # Query history collection for this user
        results = history_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"user_id": user_id}
        )
        
        if results and len(results["documents"]) > 0:
            print(f"Found {len(results['documents'][0])} relevant conversation turns")
            return results["documents"][0]
        
        return []
    except Exception as e:
        print(f"✗ Error retrieving conversation history: {e}")
        return []

def get_relevant_documents(query, n_results=MAX_CHUNKS_CONTEXT):
    """Get relevant document chunks for a query"""
    if not model or not doc_collection:
        return []
    
    try:
        # Create query embedding
        query_embedding = model.encode(query).tolist()
        
        # Query document collection
        results = doc_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if results and len(results["documents"]) > 0:
            print(f"Found {len(results['documents'][0])} relevant document chunks")
            return results["documents"][0], results["metadatas"][0]
        
        return [], []
    except Exception as e:
        print(f"✗ Error retrieving relevant documents: {e}")
        return [], []

def get_ollama_response(prompt, model_name=MODEL_NAME):
    """Get response from Ollama API"""
    try:
        # Check if Ollama is available
        version_response = requests.get("http://localhost:11434/api/version", timeout=2)
        if version_response.status_code != 200:
            return None, "Ollama service is not available"
        
        # Get available models
        models_response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if models_response.status_code != 200:
            return None, "Failed to get available models from Ollama"
        
        available_models = models_response.json().get("models", [])
        model_names = [model["name"] for model in available_models] if available_models else []
        
        # Check if our model is available or find an alternative
        if model_name not in model_names and not any(name.startswith("tinyllama") for name in model_names):
            if not model_names:
                return None, "No models available in Ollama"
            # Use the first available model
            model_name = model_names[0]
            print(f"TinyLlama not found, using {model_name} instead")
        elif model_name not in model_names:
            # Find a TinyLlama variant
            for name in model_names:
                if name.startswith("tinyllama"):
                    model_name = name
                    break
        
        print(f"Using Ollama model: {model_name}")
        
        # Generate response
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json().get("response"), None
        else:
            return None, f"Ollama returned error code: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return None, f"Error connecting to Ollama: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"

def create_prompt(query, doc_chunks, doc_metadata, history_chunks):
    """Create a prompt with context from documents and conversation history"""
    # Start with system instructions
    prompt = "You are a helpful AI assistant that answers questions based on the provided documents.\n\n"
    
    # Add document context
    if doc_chunks:
        prompt += "Here are some relevant document excerpts:\n\n"
        for i, (chunk, meta) in enumerate(zip(doc_chunks, doc_metadata)):
            source = meta.get("source", "Unknown")
            prompt += f"[DOCUMENT {i+1}] From {source}:\n{chunk}\n\n"
    
    # Add conversation history context if available
    if history_chunks:
        prompt += "Here is some relevant conversation history:\n\n"
        for i, history in enumerate(history_chunks):
            prompt += f"[HISTORY {i+1}]:\n{history}\n\n"
    
    # Add the current query
    prompt += f"Current question: {query}\n\n"
    prompt += "Please provide a helpful, accurate, and concise answer based on the document context provided. If the information isn't in the documents, say so."
    
    return prompt

def ai_agent(query, user_id=USER_ID):
    """Main agent function that orchestrates the response pipeline"""
    print(f"Processing query: '{query}'")
    
    # 1. Get relevant conversation history
    history_chunks = get_relevant_history(user_id, query)
    
    # 2. Get relevant document chunks
    doc_chunks, doc_metadata = get_relevant_documents(query)
    
    # 3. Create the prompt with context
    prompt = create_prompt(query, doc_chunks, doc_metadata, history_chunks)
    
    # 4. Get AI response from Ollama
    response, error = get_ollama_response(prompt)
    
    if error:
        print(f"AI response error: {error}")
        response = "I'm sorry, but I encountered an error while processing your request. Please try again later."
    
    # 5. Add the interaction to conversation history
    add_to_history(user_id, query, response)
    
    return response

def chat_interface():
    """Simple command-line chat interface"""
    # Check if we need to index documents
    if doc_collection.count() == 0:
        print("No documents indexed. Indexing documents...")
        index_documents()
    
    print("\n" + "="*50)
    print("🤖 Document AI Assistant")
    print("="*50)
    print("Ask me anything about the documents in your collection.")
    print("Type 'exit' to quit, 'reload' to re-index documents.")
    print("-"*50)
    
    while True:
        try:
            query = input("\n🧑 You: ")
            
            if query.lower() in ('exit', 'quit', 'bye'):
                print("🤖 Assistant: Goodbye!")
                break
                
            if query.lower() == 'reload':
                print("🤖 Assistant: Reloading and re-indexing documents...")
                # Clear the collection
                if doc_collection:
                    doc_collection.delete(where={})
                index_documents()
                continue
            
            if not query.strip():
                continue
                
            # Process query and get response
            response = ai_agent(query)
            
            # Display response
            print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n🤖 Assistant: I encountered an error: {e}")

if __name__ == "__main__":
    chat_interface()
