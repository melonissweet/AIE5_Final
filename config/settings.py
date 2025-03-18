# app/config/settings.py
import os
from dotenv import load_dotenv

load_dotenv()

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
UI_PORT = int(os.getenv("UI_PORT", 8501))

# LLM settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1000))

# Vector database settings
QDRANT_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_CLOUD_KEY")

# Collections
TICKETS_COLLECTION = "tickets"
TICKETS_FEEDBACK_COLLECTION = "tickets_with_feedback"
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# # Langsmith settings (for debugging and tracing)
# LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true").lower() == "true"
# LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
# LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
# LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "customer-support-agent")
# LANGCHAIN_TAGS = os.getenv("LANGCHAIN_TAGS", "production").split(",")

# Agent settings
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", 10))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 10))

# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB_EM = int(os.getenv("REDIS_DB_EM", 1))
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")