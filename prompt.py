"""Module for creating prompts for the AI model"""

def create_prompt(query, doc_chunks, doc_metadata, history_chunks):
    """Create a prompt with context from documents and conversation history"""
    prompt = "You are a helpful AI assistant. Be direct and natural in your responses.\n\n"
    
    if doc_chunks:
        prompt += "\n".join(doc_chunks) + "\n\n"
    
    prompt += query
    return prompt
