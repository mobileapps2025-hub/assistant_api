"""
Quick script to inspect MCL chunks for specific content.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services import (
    initialize_semantic_search,
    process_mcl_documents_with_enhanced_chunking,
    _mcl_document_chunks
)

# Initialize
print("Initializing knowledge base...")
initialize_semantic_search()
process_mcl_documents_with_enhanced_chunking()

print(f"\n{'='*80}")
print(f"Total chunks: {len(_mcl_document_chunks)}")
print(f"{'='*80}\n")

# Check for login info
print("SEARCHING FOR LOGIN INFORMATION:")
print("-" * 80)
login_terms = ["login", "log in", "sign in", "username", "password", "benutzername", "passwort", "anmeld"]
found_login = []

for chunk in _mcl_document_chunks:
    content_lower = chunk['content'].lower()
    for term in login_terms:
        if term in content_lower:
            found_login.append(chunk)
            print(f"\n✓ Found '{term}' in: {chunk['document_name']}")
            print(f"  Chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']}")
            # Show context around the term
            idx = content_lower.index(term)
            start = max(0, idx - 50)
            end = min(len(content_lower), idx + 150)
            print(f"  Context: ...{chunk['content'][start:end]}...")
            break  # Only show first match per chunk

print(f"\nTotal chunks with login info: {len(set(c['chunk_id'] for c in found_login))}")

# Check for question types
print(f"\n{'='*80}")
print("SEARCHING FOR QUESTION TYPES INFORMATION:")
print("-" * 80)
question_terms = ["question type", "types of question", "kind of question", "fragetyp", "art der frage"]
found_questions = []

for chunk in _mcl_document_chunks:
    content_lower = chunk['content'].lower()
    for term in question_terms:
        if term in content_lower:
            found_questions.append(chunk)
            print(f"\n✓ Found '{term}' in: {chunk['document_name']}")
            print(f"  Chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']}")
            idx = content_lower.index(term)
            start = max(0, idx - 50)
            end = min(len(content_lower), idx + 150)
            print(f"  Context: ...{chunk['content'][start:end]}...")
            break

print(f"\nTotal chunks with question type info: {len(set(c['chunk_id'] for c in found_questions))}")

# Also check "Creating Questions" document specifically
print(f"\n{'='*80}")
print("CONTENT FROM 'Creating Questions' DOCUMENT:")
print("-" * 80)
for chunk in _mcl_document_chunks:
    if "Creating Questions" in chunk['document_name']:
        print(f"\nChunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']}:")
        print(chunk['content'][:500])
        print("..." if len(chunk['content']) > 500 else "")
