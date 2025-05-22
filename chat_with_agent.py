"""
Chat with Agent - Interactive console for the AI Agent
"""
import os
import sys
from improved_agent import ai_agent

def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print a nice header for the chat interface"""
    clear_screen()
    print("="*50)
    print("        🤖 Python AI Assistant 🤖        ")
    print("="*50)
    print("Ask me anything about Python or type 'exit' to quit")
    print("Type 'clear' to clear the screen")
    print("-"*50)

def main():
    """Main function to run the chat interface"""
    print_header()
    
    # Chat history for context
    history = []
    
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
            response = ai_agent(user_input)
            
            # Print the response
            print(response)
            
            # Add to history
            history.append({"user": user_input, "assistant": response})
            
        except Exception as e:
            print(f"Sorry, I encountered an error: {e}")
    
if __name__ == "__main__":
    main()
