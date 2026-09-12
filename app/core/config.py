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
RAGIE_API_KEY = os.getenv("RAGIE_API_KEY", "")   # only the legacy /api/ragie/image proxy; retrieval is in-house
RAGIE_PARTITION = os.getenv("RAGIE_PARTITION", "mcl_spike")
KB_TOP_K = int(os.getenv("KB_TOP_K", os.getenv("RAGIE_TOP_K", "6")))
KB_INDEX_DIR = os.getenv("KB_INDEX_DIR", os.path.join(os.path.dirname(__file__), "..", "kb_index"))


def _resolve_public_url() -> str:
    base = os.getenv("API_PUBLIC_URL") or os.getenv("WEBSITE_HOSTNAME") or "http://127.0.0.1:8001"
    return base if base.startswith("http") else f"https://{base}"


API_PUBLIC_URL = _resolve_public_url()

# Unanswerable questions. A local MarieClaire sets GAP_SINK_URL to the central backend so the
# database credential stays in Azure; the central backend leaves it empty and writes directly.
GAP_SINK_URL = os.getenv("GAP_SINK_URL", "").strip()
GAP_INGEST_TOKEN = os.getenv("GAP_INGEST_TOKEN", "").strip()

# Shared secret MCL.Api presents when it forwards a user's turn. Unset = that surface is closed.
ASSISTANT_SERVICE_SECRET = os.getenv("ASSISTANT_SERVICE_SECRET", "").strip()


def _resolve_memories_dir() -> str:
    explicit = os.getenv("MEMORIES_DIR")
    if explicit:
        return explicit
    # On Azure App Service the app dir is wiped on each redeploy, but everything under
    # $HOME (Azure Files-backed) persists and is shared across instances. Default there
    # when we detect Azure; otherwise use a local dir for dev.
    if os.getenv("WEBSITE_HOSTNAME"):
        return os.path.join(os.getenv("HOME", "/home"), "data", "memories")
    return os.path.join(os.path.dirname(__file__), "..", "memories")


MEMORIES_DIR = _resolve_memories_dir()

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


def _to_sqlalchemy_url(conn: str):
    """Accept a real SQLAlchemy URL as-is; convert an Azure/ADO.NET keyword string
    (Server=...;Initial Catalog=...;User ID=...;Password=...) into an aioodbc URL.
    URL.create escapes the password itself, so characters like '+' survive."""
    if "://" in conn:
        return conn
    from sqlalchemy.engine import URL
    kv = dict(p.split("=", 1) for p in conn.split(";") if "=" in p)
    g = lambda *names: next((v for k, v in kv.items() for n in names if k.strip().lower() == n), "")
    server = g("server", "data source").removeprefix("tcp:")
    host, _, port = server.partition(",")
    return URL.create(
        "mssql+aioodbc",
        username=g("user id", "uid", "user"),
        password=g("password", "pwd"),
        host=host,
        port=int(port) if port else None,
        database=g("initial catalog", "database"),
        query={
            "driver": "ODBC Driver 18 for SQL Server",
            "Encrypt": "yes",
            "TrustServerCertificate": "yes" if g("trustservercertificate").lower() == "true" else "no",
        },
    )


engine = None
AsyncSessionLocal = None

if DATABASE_CONNECTION_STRING:
    try:
        engine = create_async_engine(
            _to_sqlalchemy_url(DATABASE_CONNECTION_STRING),
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
