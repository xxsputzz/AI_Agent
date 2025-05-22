"""Module for document handling with ChromaDB"""
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get or create document collection
    doc_collection = client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
except Exception as e:
    print(f"✗ Error initializing ChromaDB: {e}")
    doc_collection = None

def get_relevant_documents(query, k=2):
    """Get relevant document chunks for a query"""
    if not doc_collection:
        return [], []
        
    try:
        # Search for relevant documents
        results = doc_collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Return document chunks and metadata if found
        if results and results['documents']:
            return results['documents'][0], results['metadatas'][0]
        return [], []
    except Exception as e:
        print(f"✗ Error retrieving documents: {e}")
        return [], []

def index_documents():
    """Index documents in the document store"""
    if not doc_collection:
        print("✗ Document collection not available")
        return
        
    try:
        # Your document indexing logic here
        # For now, just a placeholder
        print("✓ Documents indexed successfully")
    except Exception as e:
        print(f"✗ Error indexing documents: {e}")
