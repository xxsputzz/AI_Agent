"""
Advanced AI Agent with proper error handling and multiple embedding options
"""
import os
import json
import requests
from pathlib import Path
import numpy as np

# Global variables to track what's available
USING_SENTENCE_TRANSFORMERS = False
USING_HUGGINGFACE = False
USING_CHROMADB = False

# Try to import different embedding models in order of preference
try:
    from sentence_transformers import SentenceTransformer
    print("✓ Using SentenceTransformer for embeddings")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    USING_SENTENCE_TRANSFORMERS = True
except ImportError:
    print("× SentenceTransformer not available, trying HuggingFaceEmbeddings...")
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        print("✓ Using HuggingFaceEmbeddings for embeddings")
        model = HuggingFaceEmbeddings()
        USING_HUGGINGFACE = True
    except ImportError:
        print("× HuggingFaceEmbeddings not available, using fallback embeddings")
        model = None

# Try to import ChromaDB
try:
    import chromadb
    print("✓ ChromaDB available for vector storage")
    USING_CHROMADB = True
except ImportError:
    print("× ChromaDB not available, using in-memory storage")

# Sample documents (in a real scenario, you'd load these from files)
sample_docs = [
    "In Python, a variable is a named location in memory that stores a value. Variables are created when you assign a value to them.",
    "Python functions are defined using the 'def' keyword, followed by the function name and parameters.",
    "Python lists are ordered collections that can store items of different types.",
    "Python dictionaries store data as key-value pairs and provide fast lookup by key.",
    "Python classes are defined using the 'class' keyword and can contain attributes and methods.",
    "Python loops (for and while) allow you to iterate over sequences like lists, tuples, and strings.",
    "Python modules are files containing Python code that can be imported and used in other Python programs.",
    "Exception handling in Python is done using try, except, else, and finally blocks.",
    "Python decorators are functions that modify the behavior of other functions or methods.",
    "Python comprehensions provide a concise way to create lists, dictionaries, and sets."
]

# Create either ChromaDB storage or in-memory storage
class VectorStore:
    """Vector store that can use either ChromaDB or in-memory storage"""
    
    def __init__(self, collection_name="python_docs"):
        self.doc_embeddings = []
        self.documents = []
        self.collection_name = collection_name
        
        if USING_CHROMADB:
            self.client = chromadb.Client()
            try:
                # Try to get existing collection
                self.collection = self.client.get_collection(name=collection_name)
                print(f"✓ Using existing ChromaDB collection '{collection_name}'")
            except:
                # Create new collection if it doesn't exist
                self.collection = self.client.create_collection(name=collection_name)
                print(f"✓ Created new ChromaDB collection '{collection_name}'")
        else:
            print("✓ Using in-memory vector storage")
            self.collection = None
    
    def add_documents(self, documents):
        """Add documents to the vector store"""
        if not documents:
            print("× No documents to add")
            return
            
        if not model:
            print("× No embedding model available, storing documents without embeddings")
            self.documents = documents
            return
            
        print(f"Adding {len(documents)} documents to vector store...")
        
        if USING_CHROMADB:
            # Add documents to ChromaDB
            for i, doc in enumerate(documents):
                if USING_SENTENCE_TRANSFORMERS:
                    embedding = model.encode(doc).tolist()
                elif USING_HUGGINGFACE:
                    embedding = model.embed_query(doc)
                
                self.collection.add(
                    embeddings=[embedding],
                    documents=[doc],
                    ids=[f"doc_{i}"]
                )
        else:
            # Store in memory
            self.documents = documents
            if USING_SENTENCE_TRANSFORMERS:
                self.doc_embeddings = model.encode(documents)
            elif USING_HUGGINGFACE:
                self.doc_embeddings = [model.embed_query(doc) for doc in documents]
                
        print(f"✓ Added {len(documents)} documents to vector store")
    
    def query(self, query_text, n_results=2):
        """Query the vector store for similar documents"""
        if not model:
            # If no embedding model, return random documents
            import random
            results = random.sample(self.documents, min(n_results, len(self.documents)))
            return {"documents": [results], "ids": [["random_1", "random_2"]]}
            
        if USING_CHROMADB:
            # Query ChromaDB
            if USING_SENTENCE_TRANSFORMERS:
                query_embedding = model.encode(query_text).tolist()
            elif USING_HUGGINGFACE:
                query_embedding = model.embed_query(query_text)
                
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        else:
            # Query in-memory store
            if USING_SENTENCE_TRANSFORMERS:
                query_embedding = model.encode(query_text)
            elif USING_HUGGINGFACE:
                query_embedding = model.embed_query(query_text)
                
            # Calculate similarities
            similarities = []
            for i, doc_emb in enumerate(self.doc_embeddings):
                # Normalize vectors and calculate dot product
                doc_emb_array = np.array(doc_emb)
                query_embedding_array = np.array(query_embedding)
                similarity = np.dot(query_embedding_array, doc_emb_array) / (
                    np.linalg.norm(query_embedding_array) * np.linalg.norm(doc_emb_array)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get top results
            top_indices = [idx for idx, _ in similarities[:n_results]]
            top_docs = [[self.documents[i] for i in top_indices]]
            top_ids = [[f"doc_{i}" for i in top_indices]]
            
            return {"documents": top_docs, "ids": top_ids}

# Setup vector store and add documents
vector_store = VectorStore()
vector_store.add_documents(sample_docs)

def get_llm_response(prompt, model_name="tinyllama"):
    """Get response from a language model API"""
    try:
        # Try Ollama API first (http://localhost:11434)
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=5
            )
            
            if response.status_code == 200:
                return response.json()["response"]
        except Exception as e:
            print(f"× Ollama error: {e}")
        
        # Try OpenAI API if configured
        if os.environ.get("OPENAI_API_KEY"):
            try:
                import openai
                openai.api_key = os.environ.get("OPENAI_API_KEY")
                
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                return response.choices[0].message.content
            except Exception as e:
                print(f"× OpenAI error: {e}")
        
        # Fallback to rule-based response
        return get_rule_based_response(prompt)
        
    except Exception as e:
        print(f"× LLM error: {e}")
        return get_rule_based_response(prompt)

def get_rule_based_response(query):
    """Generate a simple response based on keyword matching"""
    query_lower = query.lower()
    
    if "variable" in query_lower:
        return "A variable in Python is a named location in memory that stores a value. Variables are created when you assign a value to them."
    elif "function" in query_lower:
        return "Python functions are defined using the 'def' keyword, followed by the function name and parameters. Functions allow you to organize and reuse code."
    elif "list" in query_lower:
        return "Python lists are ordered collections that can store items of different types. Lists are mutable, meaning you can change their content without changing their identity."
    elif "dictionary" in query_lower or "dict" in query_lower:
        return "Python dictionaries store data as key-value pairs and provide fast lookup by key. They are unordered collections of objects."
    elif "class" in query_lower:
        return "Python classes are defined using the 'class' keyword and can contain attributes and methods. They are used for object-oriented programming."
    elif "loop" in query_lower:
        return "Python loops (for and while) allow you to iterate over sequences like lists, tuples, and strings."
    elif "module" in query_lower or "import" in query_lower:
        return "Python modules are files containing Python code that can be imported and used in other Python programs."
    elif "exception" in query_lower or "error" in query_lower or "try" in query_lower:
        return "Exception handling in Python is done using try, except, else, and finally blocks."
    elif "decorator" in query_lower:
        return "Python decorators are functions that modify the behavior of other functions or methods."
    elif "comprehension" in query_lower:
        return "Python comprehensions provide a concise way to create lists, dictionaries, and sets."
    else:
        return "I don't have specific information about that topic. Could you ask about Python variables, functions, lists, or other core Python concepts?"

def answer_question(query):
    """Main function to answer questions using the AI Agent pipeline"""
    print(f"Processing query: '{query}'")
    
    # 1. Query the vector store
    results = vector_store.query(query)
    
    if not results["documents"] or not results["documents"][0]:
        print("× No relevant documents found")
        return get_rule_based_response(query)
    
    context = results["documents"][0][0]
    print(f"✓ Found relevant context: '{context[:50]}...'")
    
    # 2. Build prompt with context
    prompt = f"""Answer the question based on the following context:
{context}
Question: {query}"""
    
    # 3. Get response from LLM
    return get_llm_response(prompt)

# Command-line interface
if __name__ == "__main__":
    print("\n=== Advanced AI Agent Demo ===")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            break
            
        answer = answer_question(query)
        print("\nAnswer:", answer)
        print("-" * 50)
