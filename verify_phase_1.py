import asyncio
import os
import logging
from app.services.vector_store import VectorStoreService
from app.services.ingestion_service import IngestionService
from app.services.chat_service import ChatService
from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService
from app.core.config import WEAVIATE_URL, COHERE_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Phase1Verifier")

async def main():
    print("="*50)
    print("PHASE 1 VERIFICATION SCRIPT")
    print("="*50)

    # 1. Check Environment
    print("\n[1] Checking Environment...")
    print(f"   - WEAVIATE_URL: {WEAVIATE_URL}")
    if not COHERE_API_KEY:
        print("   - ⚠️ COHERE_API_KEY not found. Re-ranking will be skipped.")
    else:
        print("   - ✅ COHERE_API_KEY found.")

    # 2. Initialize Services
    print("\n[2] Initializing Services...")
    try:
        vector_store = VectorStoreService()
        print("   - ✅ VectorStoreService initialized.")
        
        ingestion_service = IngestionService(vector_store)
        print("   - ✅ IngestionService initialized.")
        
        # Mock vision services for ChatService init
        vision_service = VisionService()
        image_validator = ImageValidatorService(vision_service)
        
        chat_service = ChatService(vector_store, vision_service, image_validator)
        print("   - ✅ ChatService initialized.")
        
    except Exception as e:
        print(f"   - ❌ Service initialization failed: {e}")
        return

    # 3. Test Ingestion
    print("\n[3] Testing Ingestion...")
    docs_path = "app/documents"
    if not os.path.exists(docs_path):
        print(f"   - ⚠️ Documents path '{docs_path}' not found. Creating dummy doc...")
        os.makedirs(docs_path, exist_ok=True)
        with open(os.path.join(docs_path, "test_doc.md"), "w") as f:
            f.write("# Test Document\n\n## Section 1\nThis is a test content for Weaviate ingestion.\n\n## Section 2\nAnother section with more details.")
    
    try:
        result = ingestion_service.ingest_all(docs_path)
        if result["success"]:
            print(f"   - ✅ Ingestion successful: {result['message']}")
        else:
            print(f"   - ❌ Ingestion failed: {result['message']}")
    except Exception as e:
        print(f"   - ❌ Ingestion error: {e}")

    # 4. Test Retrieval (Hybrid + Rerank)
    print("\n[4] Testing Retrieval...")
    query = "test content"
    try:
        # We use the vector_store directly to test retrieval
        results = vector_store.hybrid_search(query, alpha=0.5, limit=5)
        
        print(f"   - Query: '{query}'")
        print(f"   - Found {len(results)} results.")
        
        for i, res in enumerate(results):
            score = res.get('score', 'N/A')
            print(f"     {i+1}. [Score: {score}] {res.get('text')[:50]}...")
            
        if len(results) > 0:
            print("   - ✅ Retrieval successful.")
        else:
            print("   - ⚠️ No results found. (This might be expected if Weaviate is empty or query doesn't match)")
            
    except Exception as e:
        print(f"   - ❌ Retrieval error: {e}")

    print("\n" + "="*50)
    print("VERIFICATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
