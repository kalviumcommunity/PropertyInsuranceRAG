"""
Configuration loader for PropertyInsuranceRAG.
Safely loads environment variables from .env and exposes strongly typed settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory for the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from the repository root
load_dotenv(dotenv_path=BASE_DIR / ".env")


class AppConfig:
    """Application and RAG pipeline configuration."""

    # LLM Settings
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")

    # Vector Database Settings
    CHROMA_PERSIST_DIRECTORY: Path = BASE_DIR / os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "property_insurance_corpus")

    # Ingestion & Retrieval Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))

    # Directory Paths
    DATA_DIR: Path = BASE_DIR / "data"
    PROMPTS_DIR: Path = BASE_DIR / "prompts"
    OUTPUTS_DIR: Path = BASE_DIR / "outputs"

    @classmethod
    def validate_api_keys(cls) -> bool:
        """Verify that mandatory API credentials are configured."""
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "your_openai_api_key_here":
            return False
        return True


config = AppConfig()
