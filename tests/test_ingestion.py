import unittest
import os
from unittest.mock import MagicMock, patch, mock_open
from app.services.ingestion_service import IngestionService
from app.services.vector_store import VectorStoreService

class TestIngestionService(unittest.TestCase):
    def setUp(self):
        self.mock_vector_store = MagicMock(spec=VectorStoreService)
        self.service = IngestionService(self.mock_vector_store)

    def test_load_and_split_document(self):
        """Test splitting markdown with headers."""
        markdown_content = """
# Title
Intro text.

## Section 1
Section 1 text.

### Subsection A
Subsection A text.

## Section 2
Section 2 text.
"""
        with patch("builtins.open", mock_open(read_data=markdown_content)):
            chunks = self.service.load_and_split_document("dummy.md")

        self.assertTrue(len(chunks) > 0)
        
        # Check first chunk (Intro)
        # Note: MarkdownHeaderTextSplitter behavior depends on how it handles text before first header or under headers.
        # Usually:
        # Chunk 1: Title (Intro text)
        # Chunk 2: Title > Section 1 (Section 1 text)
        # Chunk 3: Title > Section 1 > Subsection A (Subsection A text)
        # Chunk 4: Title > Section 2 (Section 2 text)
        
        # Let's verify at least one chunk has correct header path
        found_subsection = False
        for chunk in chunks:
            if "Subsection A" in chunk["header_path"]:
                found_subsection = True
                self.assertIn("Subsection A text", chunk["text"])
                self.assertIn("Title", chunk["header_path"])
                self.assertIn("Section 1", chunk["header_path"])
        
        self.assertTrue(found_subsection)

    @patch("app.services.ingestion_service.glob.glob")
    @patch("app.services.ingestion_service.os.path.join")
    def test_ingest_all(self, mock_join, mock_glob):
        """Test full ingestion flow."""
        mock_glob.return_value = ["doc1.md", "doc2.md"]
        
        # Mock load_and_split to return dummy chunks
        with patch.object(self.service, 'load_and_split_document') as mock_split:
            mock_split.return_value = [{"text": "chunk", "header_path": "H1", "source": "doc", "chunk_index": 0}]
            
            self.mock_vector_store.add_documents.return_value = True
            
            result = self.service.ingest_all("dummy_dir")
            
            self.assertTrue(result["success"])
            self.assertEqual(result["processed_files"], 2)
            self.assertEqual(result["total_chunks"], 2)
            self.mock_vector_store.ensure_schema.assert_called_once()
            self.mock_vector_store.add_documents.assert_called_once()

    @patch("app.services.ingestion_service.glob.glob")
    def test_ingest_all_no_files(self, mock_glob):
        """Test ingestion with no files."""
        mock_glob.return_value = []
        
        result = self.service.ingest_all("dummy_dir")
        
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "No content found to ingest.")

if __name__ == '__main__':
    unittest.main()
