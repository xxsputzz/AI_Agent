# Interactive chat with memory-enhanced AI agent
import os
import sys
from memory_agent import memory_agent, ConversationMemory

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print a nice header for the chat interface"""
    clear_screen()
    print("="*60)
    print("        🤖 Memory-Enhanced Python AI Assistant 🤖        ")
    print("="*60)
    print("I remember our conversation and can provide context-aware answers!")
    print("Ask me anything about Python or type 'exit' to quit")
    print("Type 'clear' to clear the screen")
    print("-"*60)

def main():
    """Main function to run the chat interface"""
    print_header()
    
    # Initialize the memory
    memory = ConversationMemory()
    
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
            
        # Process the query
        try:
            print("\n🤖 Assistant: ", end="")
            
            # Get response from AI agent
            response, memory = memory_agent(user_input, memory)
            
            # Print the response
            print(response)
            
        except Exception as e:
            print(f"Sorry, I encountered an error: {e}")
    
if __name__ == "__main__":
    main()
