from simple_responses import is_greeting, get_simple_greeting
from history import get_relevant_history, add_to_history
from documents import get_relevant_documents
from prompt import create_prompt
from ollama import get_ollama_response

USER_ID = "default_user"

def ai_agent(query, user_id=USER_ID):
    """Main agent function that orchestrates the response pipeline"""
    print(f"Processing query: '{query}'")
    
    # Only treat as greeting if it's JUST a greeting without a question
    if is_greeting(query):
        print("Detected simple greeting without question, using simple response")
        return get_simple_greeting()
    
    # Process normally if there's more content than just a greeting
    history_chunks = get_relevant_history(user_id, query)
    doc_chunks, doc_metadata = get_relevant_documents(query)
    
    # Create the prompt with context
    prompt = create_prompt(query, doc_chunks, doc_metadata, history_chunks)
    
    # Get AI response from Ollama
    response, error = get_ollama_response(prompt)
    
    if error:
        print(f"AI response error: {error}")
        return "I apologize, but I encountered an error while processing your request. Please try again."
    
    # Add the interaction to conversation history
    add_to_history(user_id, query, response)
    
    return response