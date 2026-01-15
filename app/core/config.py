import os
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import dspy

load_dotenv()

# Configure OpenAI Client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Configure DSPy Global LM
dspy_lm = dspy.LM(model="gpt-4o", max_tokens=1000, api_key=os.getenv("OPENAI_API_KEY"))
dspy.settings.configure(lm=dspy_lm)

# MCL Image Validation Configuration
# DISABLED by default - GPT-4o Vision cannot reliably identify proprietary apps like MCL
# Enable only if you have reference MCL screenshots for comparison
ENABLE_MCL_IMAGE_VALIDATION = os.getenv("ENABLE_MCL_IMAGE_VALIDATION", "false").lower() == "true"
MCL_VALIDATION_CONFIDENCE_THRESHOLD = float(os.getenv("MCL_VALIDATION_CONFIDENCE_THRESHOLD", "0.5"))

# Vector Store Configuration
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "mcl_vector_store")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

# Cohere Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")


# Database configuration for MCL feedback system
# Default to EMPTY string to prevent accidental connection attempts if drivers are missing
# Azure on Linux often creates APPSETTING_ prefix, try both.
DATABASE_CONNECTION_STRING = os.getenv("DATABASE_CONNECTION_STRING", "")
if not DATABASE_CONNECTION_STRING:
    DATABASE_CONNECTION_STRING = os.getenv("APPSETTING_DATABASE_CONNECTION_STRING", "")

# Create async engine (only if database connection string is available)
engine = None
AsyncSessionLocal = None

print("--- ENVIRONMENT DEBUG ---")
print(f"Available Environment Keys: {[k for k in os.environ.keys()]}")
print(f"WEAVIATE_URL value: {os.getenv('WEAVIATE_URL')}")
print(f"DATABASE_CONNECTION_STRING raw len: {len(os.getenv('DATABASE_CONNECTION_STRING', ''))}")
print(f"APPSETTING_DATABASE_CONNECTION_STRING raw len: {len(os.getenv('APPSETTING_DATABASE_CONNECTION_STRING', ''))}")
print(f"Final DATABASE_CONNECTION_STRING is set: {'YES' if DATABASE_CONNECTION_STRING else 'NO'}")
print("-------------------------")

if DATABASE_CONNECTION_STRING:
    try:
        print(f"Attempting Database Connection. String Length: {len(DATABASE_CONNECTION_STRING)}")
        engine = create_async_engine(
            DATABASE_CONNECTION_STRING,
            echo=False,
            future=True
        )

        # Create async session factory
        AsyncSessionLocal = sessionmaker(
            engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )
        print("Database connection initialized successfully")
    except Exception as e:
        print(f"Warning: Database connection failed - feedback system will be disabled: {e}")
        engine = None
        AsyncSessionLocal = None
else:
    print("WARNING: DATABASE_CONNECTION_STRING environment variable is not set. Database disabled.")

# Database dependency
async def get_db():
    if not AsyncSessionLocal:
        raise Exception("Database not available")
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
