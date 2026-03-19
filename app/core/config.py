import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

load_dotenv()

_logger = logging.getLogger(__name__)

# --- Required ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Optional features ---
ENABLE_MCL_IMAGE_VALIDATION = os.getenv("ENABLE_MCL_IMAGE_VALIDATION", "false").lower() == "true"
MCL_VALIDATION_CONFIDENCE_THRESHOLD = float(os.getenv("MCL_VALIDATION_CONFIDENCE_THRESHOLD", "0.5"))

# --- Vector store ---
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "mcl_vector_store")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

# --- Reranking ---
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# --- RAG search tuning ---
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "25"))
SEARCH_ALPHA = float(os.getenv("SEARCH_ALPHA", "0.5"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "10"))
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.7"))
MIN_SEARCH_SCORE = float(os.getenv("MIN_SEARCH_SCORE", "0.0"))  # 0 = no filter; tune after testing
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "24000"))  # ~6000 tokens

# --- CORS ---
# Comma-separated list of allowed origins, e.g. "https://myapp.azurewebsites.net,http://localhost:5001"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5000,http://localhost:5001,https://localhost:5001,https://localhost:7001"
    ).split(",")
    if origin.strip()
]

# --- Admin security ---
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# --- Database ---
# Azure sometimes prefixes env vars with APPSETTING_
DATABASE_CONNECTION_STRING = os.getenv("DATABASE_CONNECTION_STRING", "") or \
                             os.getenv("APPSETTING_DATABASE_CONNECTION_STRING", "")

engine = None
AsyncSessionLocal = None

if DATABASE_CONNECTION_STRING:
    try:
        engine = create_async_engine(
            DATABASE_CONNECTION_STRING,
            echo=False,
            future=True
        )
        AsyncSessionLocal = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        _logger.info("Database connection initialized")
    except Exception as e:
        _logger.error(f"Database connection failed — feedback system disabled: {e}")
        engine = None
        AsyncSessionLocal = None
else:
    _logger.warning("DATABASE_CONNECTION_STRING not set — feedback system disabled")


async def get_db():
    if not AsyncSessionLocal:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Feedback service unavailable — database not configured")
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
