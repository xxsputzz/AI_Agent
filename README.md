# Document AI Agent

A powerful AI agent that can:
- Load and process documents (PDF, TXT, MD)
- Index them with vector embeddings
- Remember conversation history
- Generate responses using local LLMs via Ollama

## Features

- **Document Processing**: Automatically loads and chunks documents from a folder
- **Vector Search**: Uses SentenceTransformer embeddings and ChromaDB for efficient retrieval
- **Conversation Memory**: Tracks conversation history for contextual responses
- **Local LLM Integration**: Uses TinyLlama through Ollama for AI responses
- **FastAPI Backend**: Provides a RESTful API for accessing the agent
- **Web Interface**: Simple HTML/JavaScript frontend for interacting with the agent

## Installation

1. Make sure you have Python 3.7+ installed
2. Clone this repository
3. Run the server script which will install the necessary dependencies:

```
python run_server.py
```

## Usage

### Web Interface

The easiest way to use the agent is through the web interface:

1. Start the server with `python run_server.py`
2. Open a web browser and go to http://localhost:8000/static/index.html

### API

The agent provides a RESTful API:

- `GET /health` - Check server health
- `POST /index` - Index or re-index documents
- `POST /query` - Send a query to the agent
- `GET /conversations/{conversation_id}` - Get conversation history
- `GET /conversations` - List all conversations

API documentation is available at http://localhost:8000/docs

## Adding Documents

To add documents for the agent to use:

1. Place your files in the `documents` folder (supports PDF, TXT, MD)
2. Click "Reindex Documents" in the web interface
3. Start asking questions!

## Dependencies

- FastAPI - Web framework
- Uvicorn - ASGI server
- SentenceTransformers - For creating embeddings
- ChromaDB - Vector database
- PyPDF2 - PDF processing
- Ollama - Local LLM hosting

## License

MIT
