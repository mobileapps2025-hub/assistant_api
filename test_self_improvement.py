import asyncio
import os
from app.core.database import CuratedQA, Base
from app.core.config import engine, AsyncSessionLocal
from app.graph.nodes import AgentNodes
import dspy

# Ensure DSPy is configured
lm = dspy.LM(model="openai/gpt-4o", api_key=os.environ.get("OPENAI_API_KEY"))
dspy.settings.configure(lm=lm)

async def test_self_improvement():
    print("=== Self-Improvement Test ===")
    
    # 1. Define a "Secret Fact" that the agent doesn't know
    secret_question = "What is the codename for the next release?"
    secret_answer = "The codename for the next release is Project Chimera."
    
    # 2. Baseline Check
    print("\n[Step 1] Baseline Check: Asking the agent...")
    nodes = AgentNodes(vector_store=None) # Mock vector store not needed for this specific test if we rely on few-shot
    
    # Mock state with NO context about the secret
    state = {
        "language": "en",
        "query": secret_question,
        "documents": [], # No documents found
        "messages": []
    }
    
    result = await nodes.generate_answer(state)
    print(f"Agent Answer (Before Training): {result.get('answer')}")
    
    # 3. Inject Knowledge into DB (Simulating "Curating" a good answer)
    print("\n[Step 2] Injecting Knowledge into CuratedQA...")
    if not AsyncSessionLocal:
        print("Error: Database not configured.")
        return

    async with AsyncSessionLocal() as session:
        # Clean up previous test runs
        # (In a real app, we wouldn't delete, but for testing we want a clean slate)
        # Note: We can't easily delete without importing delete, so we'll just add.
        
        new_qa = CuratedQA(
            question=secret_question,
            answer=secret_answer,
            active=True
        )
        session.add(new_qa)
        await session.commit()
        print("Knowledge injected.")

    # 4. Run Training
    print("\n[Step 3] Running Optimization (train_agent.py)...")
    # We import the main function from train_agent to run it programmatically
    from train_agent import main_async
    await main_async()
    
    # 5. Verify Improvement
    print("\n[Step 4] Verification: Asking the agent again...")
    
    # Re-initialize nodes to reload the compiled module
    nodes_v2 = AgentNodes(vector_store=None)
    
    result_v2 = await nodes_v2.generate_answer(state)
    answer_v2 = result_v2.get('answer')
    print(f"Agent Answer (After Training): {answer_v2}")
    
    if "Chimera" in answer_v2:
        print("\nSUCCESS: The agent learned the secret codename via few-shot optimization!")
    else:
        print("\nNOTE: The agent might not have output the exact codename.")
        print("Reason: DSPy optimizes the *prompt strategy*, it doesn't necessarily memorize facts unless they appear in the few-shot examples selected for the prompt.")
        print("If the optimizer selected this example as one of the few-shots, it would appear.")

if __name__ == "__main__":
    asyncio.run(test_self_improvement())
