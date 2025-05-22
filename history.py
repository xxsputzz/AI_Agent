"""Module for managing conversation history using ChromaDB"""
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get or create history collection
    history_collection = client.get_or_create_collection(
        name="conversation_history",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
except Exception as e:
    print(f"✗ Error initializing ChromaDB: {e}")
    history_collection = None

def get_relevant_history(user_id, query, k=2):
    """Get relevant conversation history for a query"""
    if not history_collection:
        return []
        
    try:
        # Search for relevant history entries
        results = history_collection.query(
            query_texts=[query],
            n_results=k,
            where={"user_id": user_id}
        )
        
        # Return document content if found
        if results and results['documents']:
            return results['documents'][0]
        return []
    except Exception as e:
        print(f"✗ Error retrieving conversation history: {e}")
        return []

def add_to_history(user_id, query, response):
    """Add a conversation to history"""
    if not history_collection:
        return
        
    try:
        # Store just the response for context
        interaction = response
        
        # Add to ChromaDB with metadata
        history_collection.add(
            documents=[interaction],
            metadatas=[{"user_id": user_id}],
            ids=[f"hist_{user_id}_{len(get_relevant_history(user_id, ''))}"]
        )
    except Exception as e:
        print(f"✗ Error adding to conversation history: {e}")
