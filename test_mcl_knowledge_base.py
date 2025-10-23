#!/usr/bin/env python3
"""
Test script for MCL Knowledge Base
"""

import sys
import asyncio
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent))

from app.services import (
    start_mcl_knowledge_base, 
    debug_mcl_knowledge_base,
    find_relevant_chunks,
    get_mcl_ai_response
)

async def test_mcl_knowledge_base():
    """Test the MCL knowledge base setup and search."""
    
    print("🚀 Starting MCL Knowledge Base Test")
    print("="*60)
    
    # Initialize the knowledge base
    print("1. Initializing MCL knowledge base...")
    vector_store_id = start_mcl_knowledge_base()
    
    if vector_store_id:
        print(f"✅ Knowledge base initialized: {vector_store_id}")
    else:
        print("❌ Failed to initialize knowledge base")
        return
    
    # Debug the knowledge base
    print("\n2. Debugging knowledge base contents...")
    debug_mcl_knowledge_base()
    
    # Test search functionality
    print("3. Testing search functionality...")
    
    test_queries = [
        "How to create a checklist",
        "Wie kann ich eine Checkliste erstellen",
        "Dashboard guide",
        "MCL tablet instructions",
        "Creating questions"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        chunks = find_relevant_chunks(query, max_chunks=3)
        
        if chunks:
            print(f"   ✅ Found {len(chunks)} relevant chunks:")
            for i, chunk in enumerate(chunks):
                print(f"      {i+1}. {chunk['document_name']} (Chunk {chunk['chunk_index']+1})")
        else:
            print(f"   ❌ No relevant chunks found")
    
    print("\n4. Testing AI response generation...")
    
    # Test AI response
    test_messages = [
        {"role": "user", "content": "How can I create a checklist?"},
        {"role": "user", "content": "Wie kann ich eine Checkliste anlegen?"}
    ]
    
    for messages in test_messages:
        print(f"\n🤖 Testing AI response for: '{messages['content']}'")
        try:
            response = get_mcl_ai_response([messages])
            response_text = response.choices[0].message.content
            print(f"   ✅ Response length: {len(response_text)} characters")
            print(f"   📝 Preview: {response_text[:200]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*60)
    print("🎉 MCL Knowledge Base Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_mcl_knowledge_base())