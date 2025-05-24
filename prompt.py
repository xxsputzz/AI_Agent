"""Module for creating prompts for the AI model"""

def create_prompt(query, doc_chunks, doc_metadata, history_chunks):
    """Create a prompt with context from documents and conversation history"""
    # Check if there's a greeting in the query
    greetings = ["hi", "hello", "hey", "hi there", "hello there", 
                "good morning", "good afternoon", "good evening"]
    query_lower = query.lower().strip()
    has_greeting = any(query_lower.startswith(greeting) for greeting in greetings)
    
    # Build the base prompt
    prompt = "You are a helpful AI assistant. "
    
    if has_greeting:
        prompt += "Start your response with a brief greeting (2-3 words), then get straight to answering any question in the message. "
    else:
        prompt += "Answer questions directly and naturally. "
    
    prompt += "Keep responses clear and concise.\n\n"
    
    # Add document context if available
    if doc_chunks:
        prompt += "Based on this information:\n"
        for i, (chunk, meta) in enumerate(zip(doc_chunks, doc_metadata)):
            source = meta.get('source', 'Document')
            prompt += f"[{source}]:\n{chunk}\n\n"
    
    # Add history if available
    if history_chunks:
        prompt += "Previous conversation context:\n"
        for chunk in history_chunks:
            prompt += f"{chunk}\n"
        prompt += "\n"
    
    # Add the query
    prompt += f"Current message: {query}\n\n"
    
    return prompt
