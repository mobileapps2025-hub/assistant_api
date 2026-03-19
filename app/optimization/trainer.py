import asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.optimization.optimizer import DSPyOptimizer
from app.core.database import CuratedQA
from app.core.config import AsyncSessionLocal

# Mock "Golden" Dataset (Fallback)
GOLDEN_DATASET = [
    {
        "question": "How do I reset my password?",
        "context": ["To reset your password, go to Settings > Account > Security and click 'Reset Password'."],
        "answer": "You can reset your password by navigating to Settings > Account > Security and selecting the 'Reset Password' option."
    },
    {
        "question": "What is the MCL app?",
        "context": ["The MCL (Mobile Checklist) app is a tool for store managers to complete daily operational checks."],
        "answer": "The MCL app is a mobile tool designed for store managers to perform daily operational checklists."
    },
    {
        "question": "Can I use the app offline?",
        "context": ["Yes, the app supports offline mode. Data will sync when connection is restored."],
        "answer": "Yes, you can use the app offline. Your data will automatically sync once you are back online."
    }
]

async def fetch_curated_examples():
    """Fetch examples from the CuratedQA table."""
    if not AsyncSessionLocal:
        print("Database not configured. Using fallback dataset.")
        return []
        
    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(select(CuratedQA).where(CuratedQA.active == True))
            rows = result.scalars().all()
            
            examples = []
            for row in rows:
                examples.append({
                    "question": row.question,
                    "context": [row.answer], # Using answer as context for reinforcement
                    "answer": row.answer
                })
            
            print(f"Fetched {len(examples)} examples from CuratedQA.")
            return examples
        except Exception as e:
            print(f"Error fetching from DB: {e}")
            return []

async def run_training_pipeline():
    print("Starting DSPy Optimization Pipeline...")
    
    # Initialize Optimizer
    optimizer = DSPyOptimizer()
    
    # Load Data
    db_examples = await fetch_curated_examples()
    all_examples = GOLDEN_DATASET + db_examples
    
    train_examples = optimizer.load_examples(all_examples)
    
    # Compile
    # Run synchronous heavy compilation in a separate thread to avoid blocking the event loop
    # and potentially resolving DSPy async context issues.
    compiled_program = await asyncio.to_thread(optimizer.compile, train_examples)
    
    # Save
    output_path = "app/optimization/compiled_rag.json"
    optimizer.save(output_path)
    
    return {"status": "success", "examples_count": len(all_examples)}
