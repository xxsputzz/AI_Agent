"""Module for interacting with Ollama API"""
import requests

MODEL_NAME = "tinyllama:latest"

def get_ollama_response(prompt, model_name=MODEL_NAME):
    """Get response from Ollama API"""
    try:
        # Check if Ollama is available
        version_response = requests.get("http://localhost:11434/api/version", timeout=3)
        if version_response.status_code != 200:
            return None, "Ollama service is not available"
        
        # Get available models
        models_response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if models_response.status_code != 200:
            return None, "Failed to get available models from Ollama"
        
        available_models = models_response.json().get("models", [])
        model_names = [model["name"] for model in available_models] if available_models else []
        
        # Check if our model is available or find an alternative
        if model_name not in model_names and not any(name.startswith("tinyllama") for name in model_names):
            if not model_names:
                return None, "No models available in Ollama"
            model_name = model_names[0]
            print(f"TinyLlama not found, using {model_name} instead")
        elif model_name not in model_names:
            # Find a TinyLlama variant
            for name in model_names:
                if name.startswith("tinyllama"):
                    model_name = name
                    break
        
        print(f"Using Ollama model: {model_name}")
        
        # Generate response
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,                "options": {
                    "temperature": 0.7,  # Allow some creativity while staying focused
                    "top_p": 0.9,
                    "num_predict": 200,  # Allow for complete response with follow-up
                    "stop": ["\n\n\n"]  # Only stop on major paragraph breaks
                }
            },
            timeout=30  # Give it enough time to load model and generate
        )
        
        if response.status_code == 200:
            result = response.json().get("response")
            print(f"Received response from Ollama (length: {len(result) if result else 0})")
            return result, None
        else:
            error_msg = f"Ollama returned error code: {response.status_code}"
            if response.text:
                try:
                    error_details = response.json()
                    error_msg += f" - {error_details.get('error', '')}"
                except:
                    error_msg += f" - {response.text[:100]}"
            return None, error_msg
            
    except requests.exceptions.Timeout:
        return None, "Request to Ollama timed out. The model might be taking too long to respond."
    except requests.exceptions.ConnectionError:
        return None, "Error connecting to Ollama. Make sure the Ollama service is running on localhost:11434."
    except Exception as e:
        return None, f"Error getting response from Ollama: {str(e)}"
