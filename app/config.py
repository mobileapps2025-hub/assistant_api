import os
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Database configuration
DATABASE_CONNECTION_STRING = "mssql+aioodbc://adminMCLeu:%2BWorkappsadmin%21@mcleu-testdbserver.database.windows.net/MCLEU-SQLDB?driver=ODBC+Driver+17+for+SQL+Server"

# Create async engine
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

# Database dependency
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
