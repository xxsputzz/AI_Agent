"""
Run Server - Script to start the FastAPI server for the Document AI Agent
"""
import os
import sys
import subprocess
import webbrowser
from time import sleep

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "sentence-transformers",
        "chromadb",
        "PyPDF2"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements(packages):
    """Install missing packages"""
    print(f"Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing packages: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("Starting the FastAPI server...")
    try:
        # Use a different Python executable to avoid blocking the current process
        server_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "agent_api:app", "--host", "0.0.0.0", "--port", "8000"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Wait for server to start
        print("Waiting for server to start...")
        sleep(3)
        
        # Open browser
        webbrowser.open("http://localhost:8000/docs")  # Open Swagger UI
        webbrowser.open("http://localhost:8000/static/index.html")  # Open our web UI
        
        print("\n" + "="*50)
        print("Document AI Agent Server is running!")
        print("="*50)
        print("API Documentation: http://localhost:8000/docs")
        print("Web Interface: http://localhost:8000/static/index.html")
        print("\nPress Ctrl+C to stop the server")
        
        # Keep the server running until user interrupts
        server_process.wait()
        
    except KeyboardInterrupt:
        print("\nStopping server...")
        server_process.terminate()
        print("Server stopped")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    # Check and install requirements
    missing_packages = check_requirements()
    if missing_packages:
        if not install_requirements(missing_packages):
            print("Failed to install required packages. Please install them manually.")
            sys.exit(1)
    
    # Start the server
    start_server()
