# Advanced Memory Agent with User-Specific History and Document Loading
import os
import json
import requests
import uuid
import datetime
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

# Paths
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
DOCUMENTS_PATH = os.path.join(os.path.dirname(__file__), "documents")

# Ensure documents directory exists
os.makedirs(DOCUMENTS_PATH, exist_ok=True)

print("Starting advanced memory agent...")

# Initialize embedding model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Successfully loaded SentenceTransformer model")
except Exception as e:
    print(f"✗ Error loading SentenceTransformer model: {e}")
    model = None

# Initialize ChromaDB
try:
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # Create collections if they don't exist
    knowledge_collection = client.get_or_create_collection(name="knowledge_base")
    conversation_collection = client.get_or_create_collection(name="conversation_history")
    user_collection = client.get_or_create_collection(name="user_profiles")
    
    print(f"✓ Connected to ChromaDB at {CHROMA_DB_PATH}")
    print("✓ Collections initialized")
except Exception as e:
    print(f"✗ Error connecting to ChromaDB: {e}")
    client = None
    knowledge_collection = None
    conversation_collection = None
    user_collection = None

def load_documents_from_directory(directory_path=DOCUMENTS_PATH, extensions=['.txt', '.md']):
    """Load documents from files in a directory"""
    documents = []
    
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist")
        return documents
        
    print(f"Loading documents from {directory_path}...")
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        document_id = f"doc_{os.path.basename(file_path)}"
                        documents.append({
                            "id": document_id,
                            "content": content,
                            "source": file_path,
                            "created_at": datetime.datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                        })
                    print(f"  ✓ Loaded {file_path}")
                except Exception as e:
                    print(f"  ✗ Error loading {file_path}: {e}")
    
    return documents

def chunk_document(document, chunk_size=500, overlap=100):
    """Split a document into chunks with overlap"""
    content = document["content"]
    chunks = []
    
    # Simple character-based chunking
    start = 0
    while start < len(content):
        end = min(start + chunk_size, len(content))
        
        # Try to find a good break point (period, paragraph)
        if end < len(content):
            # Look for paragraph break
            paragraph_break = content.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + chunk_size / 2:
                end = paragraph_break
            else:
                # Look for sentence break
                sentence_break = content.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + chunk_size / 2:
                    end = sentence_break + 1  # Include the period
        
        chunk_text = content[start:end].strip()
        if chunk_text:
            chunk_id = f"{document['id']}_chunk_{len(chunks)}"
            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "source": document["source"],
                "document_id": document["id"],
                "chunk_index": len(chunks)
            })
        
        # Move the start position, with overlap
        start = end - overlap if end < len(content) else len(content)
    
    return chunks

def add_documents_to_knowledge_base(documents):
    """Process documents and add them to the knowledge base"""
    if not model or not knowledge_collection:
        print("Cannot add documents: model or knowledge collection not available")
        return
    
    # Process documents into chunks
    all_chunks = []
    for document in documents:
        chunks = chunk_document(document)
        all_chunks.extend(chunks)
    
    # Skip if no chunks
    if not all_chunks:
        print("No document chunks to add")
        return
    
    print(f"Adding {len(all_chunks)} document chunks to knowledge base...")
    
    # Create embeddings and add to collection
    chunk_ids = [chunk["id"] for chunk in all_chunks]
    chunk_texts = [chunk["content"] for chunk in all_chunks]
    chunk_metadata = [
        {
            "source": chunk["source"],
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"]
        } 
        for chunk in all_chunks
    ]
    
    # Batch process embeddings to avoid memory issues
    batch_size = 32
    for i in range(0, len(chunk_texts), batch_size):
        batch_ids = chunk_ids[i:i+batch_size]
        batch_texts = chunk_texts[i:i+batch_size]
        batch_metadata = chunk_metadata[i:i+batch_size]
        
        # Create embeddings
        batch_embeddings = model.encode(batch_texts)
        
        # Add to knowledge collection
        knowledge_collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings.tolist(),
            documents=batch_texts,
            metadatas=batch_metadata
        )
    
    print(f"✓ Added {len(all_chunks)} document chunks to knowledge base")

def add_conversation_to_history(user_id, query, response):
    """Add a conversation to the history"""
    if not model or not conversation_collection:
        return
    
    # Create a unique ID for this conversation turn
    conversation_id = f"conv_{user_id}_{uuid.uuid4()}"
    
    # Create the conversation text (both query and response)
    conversation_text = f"User: {query}\nAssistant: {response}"
    
    # Create embedding
    try:
        embedding = model.encode(conversation_text)
        
        # Add to conversation collection
        conversation_collection.add(
            ids=[conversation_id],
            embeddings=[embedding.tolist()],
            documents=[conversation_text],
            metadatas=[{
                "user_id": user_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "query": query,
                "response": response
            }]
        )
    except Exception as e:
        print(f"Error adding conversation to history: {e}")

def get_user_profile(user_id):
    """Get or create a user profile"""
    if not user_collection:
        return {"user_id": user_id}
    
    # Try to find existing profile
    results = user_collection.get(
        ids=[user_id],
        include=["metadatas"]
    )
    
    if results and results.get("metadatas") and len(results["metadatas"]) > 0:
        return results["metadatas"][0]
    
    # Create new profile if not found
    new_profile = {
        "user_id": user_id,
        "created_at": datetime.datetime.now().isoformat(),
        "conversation_count": 0
    }
    
    # No need for embeddings for user profiles, just store metadata
    user_collection.add(
        ids=[user_id],
        embeddings=[[0.0] * 384],  # Dummy embedding
        documents=["User profile"],
        metadatas=[new_profile]
    )
    
    return new_profile

def update_user_profile(user_id, updates):
    """Update a user's profile"""
    if not user_collection:
        return
    
    # Get current profile
    profile = get_user_profile(user_id)
    
    # Update profile
    updated_profile = {**profile, **updates}
    
    # Update in collection
    user_collection.update(
        ids=[user_id],
        embeddings=[[0.0] * 384],  # Dummy embedding
        documents=["User profile"],
        metadatas=[updated_profile]
    )
    
    return updated_profile

def find_relevant_knowledge(query, top_k=3):
    """Find relevant knowledge from the knowledge base"""
    if not model or not knowledge_collection:
        return []
    
    # Create query embedding
    query_embedding = model.encode(query)
    
    # Search knowledge collection
    results = knowledge_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    
    if not results or not results.get("documents"):
        return []
    
    # Flatten the nested list
    documents = results["documents"][0]
    return documents

def find_relevant_conversations(user_id, query, top_k=3):
    """Find relevant past conversations for this user"""
    if not model or not conversation_collection:
        return []
    
    # Create query embedding
    query_embedding = model.encode(query)
    
    # Search conversation collection with user_id filter
    results = conversation_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where={"user_id": user_id}
    )
    
    if not results or not results.get("documents"):
        return []
    
    # Flatten the nested list
    conversations = results["documents"][0]
    return conversations

def find_best_model(available_models, preferred_models=None):
    """Find the best model from available models based on preferences"""
    if not preferred_models:
        preferred_models = ["tinyllama", "llama2", "mistral", "gemma"]
    
    print(f"Available models: {available_models}")
    
    # First try exact matches
    for preferred in preferred_models:
        if preferred in available_models:
            return preferred
    
    # Then try partial matches (for models with version tags like tinyllama:latest)
    for preferred in preferred_models:
        for model_name in available_models:
            if preferred in model_name.lower():
                return model_name
    
    # If no match found, return the first available model if any
    if available_models:
        return available_models[0]
    
    return None

def get_ai_response(user_id, query, context, conversation_history=None):
    """Get response from AI model with context and conversation history"""
    try:
        # Get user profile
        profile = get_user_profile(user_id)
        
        # Build a more sophisticated prompt
        system_message = "You are a helpful AI assistant with memory of past conversations."
        
        # Add user profile info
        profile_info = f"This user has had {profile.get('conversation_count', 0)} conversations with you before."
        
        # Add conversation history if available
        history_text = ""
        if conversation_history and len(conversation_history) > 0:
            history_text = "Here are some relevant past conversations:\n" + "\n".join(conversation_history)
        
        # Combine all parts into a well-structured prompt
        prompt = f"""{system_message}

{profile_info}

{history_text}

Based on this context information:
{context}

Please answer the user's question:
{query}"""
        
        # Check if Ollama is available
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            ollama_available = response.status_code == 200
        except Exception as e:
            print(f"Ollama service check failed: {e}")
            ollama_available = False
            
        if ollama_available:
            print("Ollama is available, checking models...")
            
            # List available models
            try:
                models_response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if models_response.status_code == 200:
                    available_models = models_response.json().get("models", [])
                    model_names = [model["name"] for model in available_models] if available_models else []
                    
                    # Choose a model that's available
                    model_to_use = find_best_model(model_names)
                    
                    if model_to_use:
                        print(f"Using Ollama model: {model_to_use}")
                        
                        # Try to use Ollama with the selected model
                        response = requests.post(
                            "http://localhost:11434/api/generate",
                            json={
                                "model": model_to_use,
                                "prompt": prompt,
                                "stream": False
                            },
                            timeout=15  # Increase timeout for model loading
                        )
                        
                        if response.status_code == 200:
                            return response.json().get("response", "No response from model")
                        else:
                            print(f"Ollama returned error code: {response.status_code}")
                    else:
                        print("No models available in Ollama")
                else:
                    print(f"Failed to list Ollama models: {models_response.status_code}")
            except Exception as e:
                print(f"Error connecting to Ollama: {e}")
        else:
            print("Ollama is not available, using built-in responses")
                
        # Fallback to built-in response generator
        return generate_response_from_context(query, context)
    except Exception as e:
        print(f"Error in AI response generation: {e}")
        return generate_response_from_context(query, context)

def generate_response_from_context(query, context):
    """Generate a simple response based on the context without calling an external API"""
    query_lower = query.lower()
    
    # Extract relevant information from the context
    if "variable" in query_lower and "variable" in context.lower():
        return "A variable in Python is a named location in memory that stores a value. Variables are created when you assign a value to them."
    elif "function" in query_lower and "function" in context.lower():
        return "Python functions are defined using the 'def' keyword, followed by the function name and parameters. Functions allow you to organize and reuse code."
    elif "list" in query_lower and "list" in context.lower():
        return "Python lists are ordered collections that can store items of different types. Lists are mutable, meaning you can change their content without changing their identity."
    elif "dictionary" in query_lower and "dictionary" in context.lower():
        return "Python dictionaries store data as key-value pairs and provide fast lookup by key. They are unordered collections of objects."
    elif "loop" in query_lower and "loop" in context.lower():
        return "Python loops allow you to iterate over sequences like lists, tuples, and strings. Common loop types include 'for' loops and 'while' loops."
    else:
        # If we can't determine a specific answer, return the context itself
        return f"Based on the available information: {context}"

def ai_agent(user_id, query):
    """Main agent function that orchestrates the response pipeline"""
    print(f"Processing query for user {user_id}: '{query}'")
    
    # 1. Find relevant past conversations
    conversations = find_relevant_conversations(user_id, query)
    if conversations:
        print(f"Found {len(conversations)} relevant conversation(s) for user {user_id}")
    else:
        print(f"No relevant conversation history for user {user_id}")
    
    # 2. Find relevant knowledge
    knowledge = find_relevant_knowledge(query)
    if knowledge:
        print(f"Found {len(knowledge)} relevant knowledge document(s)")
        # Use the first knowledge chunk as context
        context = knowledge[0]
        print(f"Using knowledge context: '{context[:50]}...'")
    else:
        context = "No relevant information found in knowledge base."
        print("No relevant knowledge found")
    
    # 3. Get AI response with context and conversation history
    response = get_ai_response(user_id, query, context, conversations)
    
    # 4. Update user profile and add conversation to history
    profile = get_user_profile(user_id)
    conversation_count = profile.get("conversation_count", 0) + 1
    update_user_profile(user_id, {
        "conversation_count": conversation_count,
        "last_conversation": datetime.datetime.now().isoformat(),
        "last_query": query
    })
    
    # 5. Add this conversation to history
    add_conversation_to_history(user_id, query, response)
    
    return response

# Initialize by loading any existing documents
if __name__ == "__main__":
    # Load documents from the documents directory
    documents = load_documents_from_directory()
    if documents:
        add_documents_to_knowledge_base(documents)
    
    print("\n=== Advanced Memory Agent Demo ===")
    
    # Test with multiple users
    users = ["user_1", "user_2"]
    
    for user_id in users:
        print(f"\n--- Testing with user {user_id} ---")
        
        # First conversation
        query = "What is a variable in Python?"
        answer = ai_agent(user_id, query)
        print(f"\nUser {user_id} Question: {query}")
        print(f"Answer: {answer}")
        print("-" * 50)
        
        # Second conversation - should have history context now
        query = "How do I use them in functions?"
        answer = ai_agent(user_id, query)
        print(f"\nUser {user_id} Question: {query}")
        print(f"Answer: {answer}")
        print("-" * 50)
