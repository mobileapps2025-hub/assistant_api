"""
Quick test of specific failing queries.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services import (
    initialize_semantic_search,
    process_mcl_documents_with_enhanced_chunking,
    get_mcl_ai_response,
)

# Initialize
print("[SETUP] Initializing...")
initialize_semantic_search()
process_mcl_documents_with_enhanced_chunking()
print("[SETUP] Ready!\n")

# Test 1: Question types (English)
print("="*80)
print("TEST: What kind of questions are there in the app?")
print("="*80)
messages = [{"role": "user", "content": "What kind of questions are there in the app?"}]
response = get_mcl_ai_response(messages)
print("\nRESPONSE:")
print(response)
print("\n")

# Test 2: Question types (German) 
print("="*80)
print("TEST: Welche Fragetypen gibt es in der App?")
print("="*80)
messages = [{"role": "user", "content": "Welche Fragetypen gibt es in der App?"}]
response = get_mcl_ai_response(messages)
print("\nRESPONSE:")
print(response)
