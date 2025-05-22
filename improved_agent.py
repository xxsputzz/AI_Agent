# Improved version of AI Agent with minimal dependencies
import os
import json
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

print("Starting improved AI agent...")

# Use sentence-transformers directly instead of langchain
try:
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Successfully loaded SentenceTransformer model")
except Exception as e:
    print(f"✗ Error loading SentenceTransformer model: {e}")
    # Fallback to dummy embeddings if model fails
    model = None

# Sample documents (in a real scenario, you'd load these from files)
sample_docs = [
    "In Python, a variable is a named location in memory that stores a value. Variables are created when you assign a value to them.",
    "Python functions are defined using the 'def' keyword, followed by the function name and parameters.",
    "Python lists are ordered collections that can store items of different types.",
    "Python dictionaries store data as key-value pairs and provide fast lookup by key.",
    "Python loops allow you to iterate over sequences like lists, tuples, and strings."
]

# Create document embeddings
doc_embeddings = []
if model:
    print("Creating document embeddings...")
    doc_embeddings = model.encode(sample_docs)
    print(f"✓ Created {len(doc_embeddings)} document embeddings")

# Function to find most similar documents
def find_similar_docs(query, top_k=2):
    if not model or len(doc_embeddings) == 0:
        return [sample_docs[0]]  # Fallback if no embeddings
    
    # Create query embedding
    query_embedding = model.encode(query)
    
    # Calculate similarity scores
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        similarity = np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
        similarities.append((i, similarity))
    
    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k documents
    results = [sample_docs[idx] for idx, _ in similarities[:top_k]]
    return results

def get_ai_response(query, context):
    """Get response from AI model (using Ollama or fallback)"""
    try:
        # Build the prompt with context
        prompt = f"""Answer the question based on the following context:
{context}
Question: {query}"""
        
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
                    model_to_use = None
                    for model_name in ["tinyllama", "llama2", "mistral", "gemma"]:
                        if model_name in model_names:
                            model_to_use = model_name
                            break
                    
                    if not model_to_use and model_names:
                        model_to_use = model_names[0]  # Use first available model
                    
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

def ai_agent(query):
    """Main agent function that orchestrates the response pipeline"""
    print(f"Processing query: '{query}'")
    
    # 1. Find relevant documents
    relevant_docs = find_similar_docs(query)
    context = relevant_docs[0] if relevant_docs else "No relevant information found."
    print(f"Found relevant context: '{context[:50]}...'")
    
    # 2. Get AI response
    response = get_ai_response(query, context)
    
    return response

# Example Usage
if __name__ == "__main__":
    print("\n=== AI Agent Demo ===")
    queries = [
        "What is a variable in Python?",
        "How do I create a function?",
        "Tell me about Python lists"
    ]
    
    for query in queries:
        answer = ai_agent(query)
        print("\nQuestion:", query)
        print("Answer:", answer)
        print("-" * 50)
