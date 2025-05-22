# Chat interface for the advanced memory agent
import os
import sys
import uuid
import socket
from advanced_memory_agent import ai_agent, load_documents_from_directory, add_documents_to_knowledge_base

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print a nice header for the chat interface"""
    clear_screen()
    print("="*60)
    print("        🤖 Advanced AI Assistant with Memory 🤖        ")
    print("="*60)
    print("This assistant remembers your previous conversations")
    print("and can answer questions based on its knowledge base.")
    print("-"*60)
    print("Commands:")
    print("  'exit' - Quit the chat")
    print("  'clear' - Clear the screen")
    print("  'load <directory>' - Load documents from a directory")
    print("-"*60)

def get_user_id():
    """Generate a unique user ID based on the machine's hostname and IP"""
    try:
        # Try to get a stable identifier for this user
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        user_id = f"user_{hostname}_{ip}"
        return user_id
    except:
        # Fallback to a random UUID if we can't get network info
        return f"user_{uuid.uuid4().hex[:8]}"

def handle_command(command):
    """Handle special commands"""
    if command.startswith('load '):
        # Extract directory path
        parts = command.split(' ', 1)
        if len(parts) < 2:
            print("Please specify a directory path to load documents from")
            return
            
        directory = parts[1].strip()
        
        # Check if directory exists
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return
            
        # Load documents
        print(f"Loading documents from {directory}...")
        documents = load_documents_from_directory(directory)
        
        if documents:
            # Add documents to knowledge base
            add_documents_to_knowledge_base(documents)
            print(f"Added {len(documents)} documents to knowledge base")
        else:
            print("No documents found or loaded")
        
        return True  # Command handled
    
    return False  # Command not handled

def main():
    """Main function to run the chat interface"""
    print_header()
    
    # Get or generate user ID
    user_id = get_user_id()
    print(f"Your user ID: {user_id}")
    
    while True:
        # Get user input
        try:
            user_input = input("\n🧑 You: ")
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
            
        # Check for exit command
        if user_input.lower() in ('exit', 'quit', 'bye'):
            print("🤖 Assistant: Goodbye! Have a great day!")
            break
            
        # Check for clear command
        if user_input.lower() == 'clear':
            print_header()
            continue
            
        # Skip empty inputs
        if not user_input.strip():
            continue
            
        # Check for special commands
        if handle_command(user_input.lower()):
            continue
            
        # Process the query
        try:
            print("\n🤖 Assistant: ", end="")
            
            # Get response from AI agent
            response = ai_agent(user_id, user_input)
            
            # Print the response
            print(response)
            
        except Exception as e:
            print(f"Sorry, I encountered an error: {e}")
    
if __name__ == "__main__":
    main()
