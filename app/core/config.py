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

# --- Ragie retrieval (Layer 5) ---
RAGIE_API_KEY = os.getenv("RAGIE_API_KEY", "")
RAGIE_PARTITION = os.getenv("RAGIE_PARTITION", "mcl_spike")
RAGIE_TOP_K = int(os.getenv("RAGIE_TOP_K", "6"))


def _resolve_public_url() -> str:
    base = os.getenv("API_PUBLIC_URL") or os.getenv("WEBSITE_HOSTNAME") or "http://127.0.0.1:8001"
    return base if base.startswith("http") else f"https://{base}"


API_PUBLIC_URL = _resolve_public_url()

# --- CORS ---
# Comma-separated list of allowed origins, e.g. "https://myapp.azurewebsites.net,http://localhost:5001"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5000,http://localhost:5001,https://localhost:5001,https://localhost:7001,https://localhost:7241,https://mclai-dbd7cvcfabdgayap.westeurope-01.azurewebsites.net"
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
