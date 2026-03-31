import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()  # Load variables from .env if present

CLASS = "MIS372T"
AZURE_INFERENCE_BASE = f"https://aistudio-apim-ai-gateway02.azure-api.net/{CLASS}/v1/models"
CHAT_MODEL_NAME = "gpt-4o-mini" # Using standard model name instead of non-existent gpt-4.1-nano
CHAT_API_VERSION = "2024-05-01-preview"

# We can fall back to the key previously provided, but best practice is env var
AZURE_API_KEY = os.getenv("AZURE_API_KEY", "5aa2f6d151e04a2a92023c0d0a74b355")

APIM_HEADERS = (
    {"Ocp-Apim-Subscription-Key": AZURE_API_KEY}
    if AZURE_API_KEY else None
)

def build_llm():
    """
    Initialize the Azure LLM for use across all techniques.
    """
    if not AZURE_API_KEY:
        raise RuntimeError(
            "Missing AZURE_API_KEY!\n"
            "Set it via environment variable: export AZURE_API_KEY='your-key-here'\n"
            "Or use a .env file."
        )

    llm = init_chat_model(
        model=CHAT_MODEL_NAME,
        model_provider="azure_ai",
        endpoint=AZURE_INFERENCE_BASE,
        credential=AZURE_API_KEY,
        api_version=CHAT_API_VERSION,
        client_kwargs={"headers": APIM_HEADERS},
    )
    return llm
