# HTTP library for making requests to APIs
import requests
# Configuration constants for Ollama server and embedding model
from config import OLLAMA_BASE_URL, EMBED_MODEL


# Convert text into vector embeddings using Ollama's embedding model
# This function is used to vectorize both documents and user queries
# for similarity search in the vector database
# Parameters:
#   text: the text string to be converted into an embedding vector
# Returns:
#   embedding: a list of floating-point numbers representing the text vector
def embed_text(text: str):
    # Send a POST request to Ollama's embeddings API
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={
            "model": EMBED_MODEL,  # The embedding model to use (from config)
            "prompt": text  # The text to be embedded
        }
    )
    # Raise an exception if the request failed (non-200 status code)
    response.raise_for_status()
    # Extract and return the embedding vector from the response
    return response.json()["embedding"]


# Generate a text response from Ollama's LLM based on a prompt
# This function is used to generate answers to user questions
# based on retrieved context from the vector database
# Parameters:
#   prompt: the input prompt/question for the language model
#   model: the name of the LLM model to use (e.g., 'llama3.2')
# Returns:
#   response: the generated text response from the LLM
def generate_answer(prompt: str, model: str):
    # Send a POST request to Ollama's generate API
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,  # The language model to use
            "prompt": prompt,  # The input prompt for the model
            "stream": False  # Request non-streaming response (full response at once)
        }
    )
    # Raise an exception if the request failed (non-200 status code)
    response.raise_for_status()
    # Extract and return the generated response text from the API response
    return response.json()["response"]
