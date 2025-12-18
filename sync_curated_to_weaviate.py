import asyncio
import sys
import os

# Add current directory to sys.path
sys.path.append(os.getcwd())

from sqlalchemy import select
from app.core.config import AsyncSessionLocal
from app.core.database import CuratedQA
from app.services.vector_store import VectorStoreService

async def sync_data():
    print("Initializing Vector Store Service...")
    vs_service = VectorStoreService()
    
    if not AsyncSessionLocal:
        print("Database not configured.")
        return

    print("Fetching Curated QA entries...")
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(CuratedQA).where(CuratedQA.active == True))
        qas = result.scalars().all()
        
        print(f"Found {len(qas)} active entries.")
        
        chunks_to_add = []
        for qa in qas:
            print(f"Preparing ID {qa.id}: {qa.question[:30]}...")
            chunk = {
                "text": f"Question: {qa.question}\nAnswer: {qa.answer}",
                "header_path": "Curated Knowledge",
                "source": "User Feedback",
                "chunk_index": 0
            }
            chunks_to_add.append(chunk)
            
        if chunks_to_add:
            print(f"Indexing {len(chunks_to_add)} chunks to Weaviate...")
            success = vs_service.add_documents(chunks_to_add)
            if success:
                print("Sync complete! Knowledge is now searchable.")
            else:
                print("Sync failed. Check logs.")
        else:
            print("Nothing to sync.")

if __name__ == "__main__":
    asyncio.run(sync_data())
