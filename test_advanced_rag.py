"""
Test script for Advanced RAG implementation in MCL knowledge base.
Tests the 4 queries that were previously failing.
"""
import asyncio
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services import (
    initialize_semantic_search,
    process_mcl_documents_with_enhanced_chunking,
    create_embeddings_for_chunks,
    get_mcl_ai_response,
    find_relevant_chunks,
    expand_query_with_variants
)

async def test_query(query: str, description: str):
    """Test a single query and display results."""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"Query: '{query}'")
    print(f"{'='*80}")
    
    # Test query expansion
    print("\n[1] Query Expansion:")
    variants = expand_query_with_variants(query)
    for i, variant in enumerate(variants, 1):
        print(f"   Variant {i}: {variant}")
    
    # Test chunk retrieval
    print("\n[2] Chunk Retrieval:")
    chunks = find_relevant_chunks(query, max_chunks=15)
    print(f"   Retrieved {len(chunks)} chunks")
    
    if chunks:
        print("\n[3] Top Retrieved Documents:")
        doc_counts = {}
        for chunk in chunks[:10]:
            doc_name = chunk['document_name']
            doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
        
        for doc_name, count in sorted(doc_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {doc_name}: {count} chunk(s)")
        
        print("\n[4] Sample Content from Top Chunk:")
        top_chunk = chunks[0]
        content_preview = top_chunk['content'][:300].replace('\n', ' ')
        print(f"   Document: {top_chunk['document_name']}")
        print(f"   Content: {content_preview}...")
    else:
        print("   ⚠️ WARNING: No chunks retrieved!")
    
    # Test full AI response
    print("\n[5] AI Response:")
    try:
        messages = [{"role": "user", "content": query}]
        response = get_mcl_ai_response(messages)
        
        # Print first 500 characters of response
        response_preview = response[:500]
        print(f"   {response_preview}")
        if len(response) > 500:
            print(f"   ... (truncated, full response is {len(response)} characters)")
        
        # Check if response indicates "not found"
        not_found_indicators = [
            "nicht in den verfügbaren dokumenten",
            "information not found",
            "don't have that specific information",
            "nicht finde",
            "keine information"
        ]
        
        response_lower = response.lower()
        if any(indicator in response_lower for indicator in not_found_indicators):
            print("\n   ❌ FAIL: Response indicates information not found!")
        else:
            print("\n   ✅ PASS: Response provided information!")
            
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")

async def main():
    """Run all tests."""
    print("="*80)
    print("ADVANCED RAG TESTING")
    print("="*80)
    
    # Initialize knowledge base
    print("\n[SETUP] Initializing MCL knowledge base...")
    try:
        print("   1. Loading semantic search model...")
        initialize_semantic_search()
        
        print("   2. Processing MCL documents...")
        process_mcl_documents_with_enhanced_chunking()
        
        # Embeddings are created automatically during processing
        print("   3. Embeddings created automatically during processing")
        
        print("   ✅ Knowledge base ready!")
    except Exception as e:
        print(f"   ❌ Setup failed: {str(e)}")
        return
    
    # Test cases from user's examples
    test_cases = [
        ("How can I create a new task?", "Task Creation (English)"),
        ("What kind of questions there are in the app?", "Question Types (English)"),
        ("How can I get the general description of a task?", "Task Description (English)"),
        ("How to log into MCL?", "Login Procedure (English)"),
        ("Wie kann ich eine Checkliste erstellen?", "Checklist Creation (German)"),
        ("Welche Fragetypen gibt es in der App?", "Question Types (German)"),
    ]
    
    for query, description in test_cases:
        await test_query(query, description)
        await asyncio.sleep(0.5)  # Brief pause between tests
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
