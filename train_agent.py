import asyncio
from dotenv import load_dotenv
from app.optimization.trainer import run_training_pipeline

# Load environment variables
load_dotenv()

def main():
    asyncio.run(run_training_pipeline())

if __name__ == "__main__":
    main()
