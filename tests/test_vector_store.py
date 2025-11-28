import os
import shutil
import tempfile
import unittest
from pathlib import Path
from app.services.vector_store import VectorStoreService

class TestVectorStoreService(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.service = VectorStoreService(self.test_dir)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_initialization_creates_directory(self):
        """Test that the storage directory is created if it doesn't exist."""
        new_dir = os.path.join(self.test_dir, "subdir")
        VectorStoreService(new_dir)
        self.assertTrue(os.path.exists(new_dir))

    def test_index_exists_false(self):
        """Test index_exists returns False when files are missing."""
        self.assertFalse(self.service.index_exists())

    def test_index_exists_true(self):
        """Test index_exists returns True when files exist."""
        # Create dummy files
        with open(self.service.index_path, 'w') as f:
            f.write("dummy index")
        with open(self.service.metadata_path, 'w') as f:
            f.write("dummy metadata")
            
        self.assertTrue(self.service.index_exists())

    def test_index_exists_partial(self):
        """Test index_exists returns False when only one file exists."""
        with open(self.service.index_path, 'w') as f:
            f.write("dummy index")
        self.assertFalse(self.service.index_exists())

    def test_build_index(self):
        """Test building the index from documents."""
        # Create a dummy documents directory
        docs_dir = os.path.join(self.test_dir, "documents")
        os.makedirs(docs_dir, exist_ok=True)
        
        # Create a dummy MCL markdown file
        mcl_file = os.path.join(docs_dir, "mcl_guide.md")
        with open(mcl_file, "w", encoding="utf-8") as f:
            f.write("# MCL Guide\n\nThis is a test document for the MCL knowledge base.\nIt contains some sample text to be chunked and indexed.")
            
        # Create a dummy non-MCL file (should be ignored)
        spotplan_file = os.path.join(docs_dir, "spotplan_guide.md")
        with open(spotplan_file, "w", encoding="utf-8") as f:
            f.write("# Spotplan Guide\n\nThis should be ignored.")
            
        # Build the index
        result = self.service.build_index(docs_dir)
        
        # Check results
        if result.get("error") and "Missing dependencies" in result["error"]:
            print("Skipping build_index test due to missing dependencies")
            return

        self.assertTrue(result["success"])
        self.assertEqual(result["processed_files"], 1) # Only MCL file
        self.assertTrue(self.service.index_exists())
        
        # Verify metadata content
        with open(self.service.metadata_path, 'r', encoding='utf-8') as f:
            import json
            metadata = json.load(f)
            self.assertTrue(len(metadata) > 0)
            self.assertEqual(metadata[0]["document_name"], "mcl_guide.md")

    def test_load_index(self):
        """Test loading the index from disk."""
        # First build an index
        docs_dir = os.path.join(self.test_dir, "documents")
        os.makedirs(docs_dir, exist_ok=True)
        mcl_file = os.path.join(docs_dir, "mcl_guide.md")
        with open(mcl_file, "w", encoding="utf-8") as f:
            f.write("# MCL Guide\n\nTest content for loading.")
            
        build_result = self.service.build_index(docs_dir)
        if build_result.get("error") and "Missing dependencies" in build_result["error"]:
            print("Skipping load_index test due to missing dependencies")
            return

        # Create a new service instance to simulate a fresh start
        new_service = VectorStoreService(self.test_dir)
        
        # Load the index
        success = new_service.load_index()
        
        self.assertTrue(success)
        self.assertIsNotNone(new_service.index)
        self.assertEqual(len(new_service.chunks), 1)
        self.assertEqual(new_service.chunks[0]["document_name"], "mcl_guide.md")

    def test_load_index_missing(self):
        """Test loading when index files are missing."""
        success = self.service.load_index()
        self.assertFalse(success)

    def test_search(self):
        """Test semantic search."""
        # Build index with known content
        docs_dir = os.path.join(self.test_dir, "documents")
        os.makedirs(docs_dir, exist_ok=True)
        
        # Create two distinct documents
        doc1 = os.path.join(docs_dir, "mcl_login.md")
        with open(doc1, "w", encoding="utf-8") as f:
            f.write("# Login Guide\n\nTo login, enter your username and password on the main screen.")
            
        doc2 = os.path.join(docs_dir, "mcl_tasks.md")
        with open(doc2, "w", encoding="utf-8") as f:
            f.write("# Task Guide\n\nTo create a task, click the plus button in the bottom right corner.")
            
        build_result = self.service.build_index(docs_dir)
        if build_result.get("error") and "Missing dependencies" in build_result["error"]:
            print("Skipping search test due to missing dependencies")
            return
            
        # Search for login info
        results = self.service.search("how do I sign in?", limit=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document_name"], "mcl_login.md")
        self.assertTrue("similarity_score" in results[0])
        
        # Search for task info
        results = self.service.search("creating new tasks", limit=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document_name"], "mcl_tasks.md")

    def test_search_without_index(self):
        """Test search returns empty list when index is not loaded."""
        results = self.service.search("test")
        self.assertEqual(results, [])

if __name__ == '__main__':
    unittest.main()
