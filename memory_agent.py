# Memory-enhanced AI Agent with ChromaDB
import os
import json
import time
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

print("Starting memory-enhanced AI agent...")

# Initialize embedding model
try:
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Successfully loaded SentenceTransformer model")
    
    # Set up embedding function for ChromaDB
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name='all-MiniLM-L6-v2'
    )
except Exception as e:
    print(f"✗ Error loading SentenceTransformer model: {e}")
    model = None
    embedding_function = None

# Initialize ChromaDB
try:
    # Set up a persistent ChromaDB
    db_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    os.makedirs(db_directory, exist_ok=True)
    
    client = chromadb.PersistentClient(path=db_directory)
    print(f"✓ Connected to ChromaDB at {db_directory}")
    
    # Create or get collections
    try:
        # Collection for knowledge base
        knowledge_collection = client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=embedding_function
        )
        
        # Collection for conversation history
        memory_collection = client.get_or_create_collection(
            name="conversation_memory",
            embedding_function=embedding_function
        )
        
        print("✓ Collections initialized")
    except Exception as e:
        print(f"✗ Error creating collections: {e}")
        knowledge_collection = None
        memory_collection = None
except Exception as e:
    print(f"✗ Error connecting to ChromaDB: {e}")
    client = None
    knowledge_collection = None
    memory_collection = None

# Sample documents (for initial knowledge base if empty)
sample_docs = [
    "In Python, a variable is a named location in memory that stores a value. Variables are created when you assign a value to them.",
    "Python functions are defined using the 'def' keyword, followed by the function name and parameters.",
    "Python lists are ordered collections that can store items of different types.",
    "Python dictionaries store data as key-value pairs and provide fast lookup by key.",
    "Python loops allow you to iterate over sequences like lists, tuples, and strings."
]

# Initialize knowledge base if empty
def initialize_knowledge_base():
    if knowledge_collection and knowledge_collection.count() == 0:
        print("Initializing knowledge base with sample documents...")
        
        for i, doc in enumerate(sample_docs):
            knowledge_collection.add(
                documents=[doc],
                ids=[f"sample_doc_{i}"],
                metadatas=[{"source": "sample", "created_at": time.time()}]
            )
        
        print(f"✓ Added {len(sample_docs)} documents to knowledge base")

# Initialize the knowledge base
initialize_knowledge_base()

class ConversationMemory:
    def __init__(self, collection=memory_collection, max_history=10):
        self.collection = collection
        self.max_history = max_history
        self.current_session = f"session_{int(time.time())}"
        
    def add_interaction(self, query, response):
        """Add a query-response pair to memory"""
        if not self.collection:
            print("Memory collection not available")
            return
            
        # Create a combined document to store the interaction
        interaction = f"User: {query}\nAssistant: {response}"
        
        # Create a unique ID for this interaction
        interaction_id = f"{self.current_session}_{int(time.time())}"
        
        # Add to collection
        self.collection.add(
            documents=[interaction],
            ids=[interaction_id],
            metadatas=[{
                "session": self.current_session,
                "timestamp": time.time(),
                "query": query,
                "type": "interaction"
            }]
        )
        
        # Keep history size manageable by checking total count
        self._trim_history_if_needed()
    
    def _trim_history_if_needed(self):
        """Trim history if it exceeds max size"""
        if not self.collection:
            return
            
        # Get current count
        count = self.collection.count()
        
        # If we exceed max history, remove oldest entries
        if count > self.max_history:
            # Get all IDs
            all_data = self.collection.get()
            ids = all_data['ids']
            timestamps = [m.get('timestamp', 0) for m in all_data['metadatas']]
            
            # Sort by timestamp
            id_time_pairs = sorted(zip(ids, timestamps), key=lambda x: x[1])
            
            # Get oldest IDs to remove
            to_remove = id_time_pairs[:count - self.max_history]
            ids_to_remove = [pair[0] for pair in to_remove]
            
            # Remove oldest entries
            if ids_to_remove:
                self.collection.delete(ids=ids_to_remove)
    
    def get_relevant_history(self, query, n_results=3):
        """Get history relevant to the current query"""
        if not self.collection or self.collection.count() == 0:
            return []
            
        # Search for relevant history based on query
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count())
        )
        
        # Return the documents
        return results['documents'][0] if results['documents'] else []
    
    def format_for_context(self, query, n_results=2):
        """Format relevant history as context for the LLM"""
        relevant_history = self.get_relevant_history(query, n_results)
        
        if not relevant_history:
            return ""
            
        formatted = "Relevant conversation history:\n"
        for i, interaction in enumerate(relevant_history):
            formatted += f"{interaction}\n"
            
        return formatted


def find_relevant_docs(query, top_k=2):
    """Find documents relevant to the query"""
    if not knowledge_collection or knowledge_collection.count() == 0:
        return sample_docs[:1]  # Fallback if no knowledge base
    
    # Query the knowledge collection
    results = knowledge_collection.query(
        query_texts=[query],
        n_results=min(top_k, knowledge_collection.count())
    )
    
    # Return the documents
    return results['documents'][0] if results['documents'] else []


def find_best_model(model_names, preferred_models=None):
    """Find the best model from available models based on preferences"""
    if not preferred_models:
        preferred_models = ["tinyllama", "llama2", "mistral", "gemma"]
    
    print(f"Available models: {model_names}")
    
    # First try exact matches
    for preferred in preferred_models:
        if preferred in model_names:
            return preferred
    
    # Then try partial matches (for models with version tags like tinyllama:latest)
    for preferred in preferred_models:
        for model_name in model_names:
            if preferred in model_name.lower():
                return model_name
    
    # If no match found, return the first available model if any
    if model_names:
        return model_names[0]
    
    return None


def get_ai_response(query, context, conversation_history=""):
    """Get response from AI model (using Ollama or fallback)"""
    try:
        # Build the prompt with context and conversation history
        prompt = f"""Answer the question based on the following context and conversation history.

{conversation_history}

Context:
{context}

Question: {query}

Please provide a helpful and accurate response."""
        
        # Check if Ollama is available by trying to connect
        try:
            # First check if Ollama service is running
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
                            timeout=10  # Increase timeout for model loading
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
                
        # If we get here, either Ollama isn't available or returned an error
        # Use our built-in response generator
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


def memory_agent(query, memory=None):
    """Main agent function with memory capabilities"""
    print(f"Processing query: '{query}'")
    
    # Initialize memory if not provided
    if memory is None:
        memory = ConversationMemory()
    
    # 1. Get relevant conversation history
    conversation_history = memory.format_for_context(query)
    if conversation_history:
        print(f"Found relevant conversation history")
    
    # 2. Find relevant documents from knowledge base
    relevant_docs = find_relevant_docs(query)
    context = "\n".join(relevant_docs) if relevant_docs else "No relevant information found."
    print(f"Found relevant context: '{context[:50]}...'")
    
    # 3. Get AI response with context and history
    response = get_ai_response(query, context, conversation_history)
    
    # 4. Save interaction to memory
    memory.add_interaction(query, response)
    
    return response, memory


# Example Usage
if __name__ == "__main__":
    print("\n=== Memory Agent Demo ===")
    
    # Create a memory instance for the session
    memory = ConversationMemory()
    
    # Demo queries to show memory capabilities
    queries = [
        "What is a variable in Python?",
        "How do I create a function?",
        "Can I use variables inside functions?",
        "Tell me more about Python lists",
        "How are lists different from dictionaries?"
    ]
    
    for query in queries:
        answer, memory = memory_agent(query, memory)
        print("\nQuestion:", query)
        print("Answer:", answer)
        print("-" * 50)
