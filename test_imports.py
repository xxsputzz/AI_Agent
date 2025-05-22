# Simple script to test imports
try:
    from langchain.embeddings import HuggingFaceEmbeddings
    print("Successfully imported HuggingFaceEmbeddings")
except ImportError as e:
    print(f"Error importing Huggiopcon ngFaceEmbeddings: {e}")

try:
    import chromadb
    print("Successfully imported chromadb")
except ImportError as e:
    print(f"Error importing chromadb: {e}")

try:
    import requests
    print("Successfully imported requests")
except ImportError as e:
    print(f"Error importing requests: {e}")
