import os
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Database configuration for MCL feedback system
DATABASE_CONNECTION_STRING = os.getenv(
    "DATABASE_CONNECTION_STRING",
    "mssql+aioodbc://adminMCLeu:%2BWorkappsadmin%21@mcleu-testdbserver.database.windows.net/MCLEU-SQLDB?driver=ODBC+Driver+17+for+SQL+Server"
)

# Create async engine (only if database connection string is available)
engine = None
AsyncSessionLocal = None

if DATABASE_CONNECTION_STRING:
    try:
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

# Database dependency
async def get_db():
    if not AsyncSessionLocal:
        raise Exception("Database not available")
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
