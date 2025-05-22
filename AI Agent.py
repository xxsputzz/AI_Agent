from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import requests
import json
import os

# Initialize embedding model
embeddings = HuggingFaceEmbeddings()

# Setup ChromaDB
client = chromadb.Client()
collection = client.create_collection(name="python_docs")

# Sample data to populate ChromaDB (in a real scenario, you'd load documents)
sample_docs = [
    "In Python, a variable is a named location in memory that stores a value. Variables are created when you assign a value to them.",
    "Python functions are defined using the 'def' keyword, followed by the function name and parameters.",
    "Python lists are ordered collections that can store items of different types."
]

# Add sample data to ChromaDB
for i, doc in enumerate(sample_docs):
    embedding = embeddings.embed_query(doc)
    collection.add(
        embeddings=[embedding],
        documents=[doc],
        ids=[f"doc_{i}"]
    )

def get_response(query):
    # 1. Embed the query
    embedding = embeddings.embed_query(query)

    # 2. Search ChromaDB
    results = collection.query(
        query_embeddings=[embedding],
        n_results=2  # Retrieve top 2 most similar chunks
    )

    # 3. Build the prompt
    prompt = f"""Answer the question based on the following context:
{results['documents'][0]}
Question: {query}"""

    # 4. Send to an LLM (Using Ollama API in this example)
    # Change the URL to your Ollama server if needed
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "tinyllama",
            "prompt": prompt,
            "stream": False
        }
    )
    
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.status_code}"

# Example Usage
if __name__ == "__main__":
    query = "What is a variable in Python?"
    answer = get_response(query)
    print("Question:", query)
    print("Answer:", answer)