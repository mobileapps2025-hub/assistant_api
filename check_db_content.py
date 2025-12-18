import asyncio
import sys
import os

# Add current directory to sys.path to ensure app imports work
sys.path.append(os.getcwd())

from sqlalchemy import select
from app.core.config import AsyncSessionLocal
from app.core.database import Feedback, CuratedQA

async def check_data():
    if not AsyncSessionLocal:
        print("Database not configured in app.core.config.")
        return

    print("Connecting to database...")
    try:
        async with AsyncSessionLocal() as session:
            print("\n=== Recent Feedback Entries (Last 5) ===")
            result = await session.execute(select(Feedback).order_by(Feedback.created_at.desc()).limit(5))
            feedbacks = result.scalars().all()
            if not feedbacks:
                print("No feedback found.")
            for f in feedbacks:
                print(f"ID: {f.id} | Type: {f.feedback_type} | Comment: {f.user_comment} | Created: {f.created_at}")

            print("\n=== Recent Curated QA Entries (Last 5) ===")
            result = await session.execute(select(CuratedQA).order_by(CuratedQA.created_at.desc()).limit(5))
            qas = result.scalars().all()
            if not qas:
                print("No curated QA found.")
            for qa in qas:
                print(f"ID: {qa.id}")
                print(f"  Question: {qa.question}")
                print(f"  Answer:   {qa.answer}")
                print(f"  Active:   {qa.active}")
                print("-" * 30)
    except Exception as e:
        print(f"Error connecting or querying database: {e}")

if __name__ == "__main__":
    asyncio.run(check_data())
