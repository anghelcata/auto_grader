# chains.py

import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM, OllamaEmbeddings

# Load environment variables from .env
load_dotenv()

def load_llm():
    """
    Load an LLM from Ollama based on the current environment configuration.
    Reads the LLM_MODEL and OLLAMA_BASE_URL from os.environ.
    """
    model_name = os.getenv("LLM_MODEL", "llama3.1:latest")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"🔁 Loading model: {model_name} from Ollama")

    return OllamaLLM(
        model=model_name,
        base_url=base_url,
        temperature=0.0,
        #top_k=15,
        #top_p=0.3,
        num_ctx=8192
    )

def load_embedding_model(model_name=None, base_url=None):
    """
    Load an embedding model via Ollama.
    Returns (embedding_model, embedding_dimension).
    """
    model_name = model_name or os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
    base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    embedding = OllamaEmbeddings(
        model=model_name,
        base_url=base_url
    )
    return embedding, 1024  # Default embedding dimension
