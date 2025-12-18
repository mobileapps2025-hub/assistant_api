import unittest
from unittest.mock import MagicMock, patch
from app.services.chat_service import ChatService
from app.services.vector_store import VectorStoreService
from app.services.vision_service import VisionService
from app.services.image_validator import ImageValidatorService

class TestChatService(unittest.TestCase):
    def setUp(self):
        self.mock_vector_store = MagicMock(spec=VectorStoreService)
        self.mock_vision_service = MagicMock(spec=VisionService)
        self.mock_image_validator = MagicMock(spec=ImageValidatorService)
        
        # Patch config to avoid real API keys
        with patch("app.services.chat_service.COHERE_API_KEY", "dummy-key"):
            with patch("app.services.chat_service.cohere.Client") as mock_cohere:
                self.mock_cohere_client = mock_cohere.return_value
                self.service = ChatService(
                    self.mock_vector_store,
                    self.mock_vision_service,
                    self.mock_image_validator
                )

    def test_find_relevant_chunks_with_reranking(self):
        """Test hybrid search followed by re-ranking."""
        # Mock initial hybrid search results
        initial_results = [
            {"text": "doc1", "source": "s1", "uuid": "1"},
            {"text": "doc2", "source": "s2", "uuid": "2"}
        ]
        self.mock_vector_store.hybrid_search.return_value = initial_results
        
        # Mock Cohere response
        mock_response = MagicMock()
        hit1 = MagicMock()
        hit1.index = 0
        hit1.relevance_score = 0.9
        
        hit2 = MagicMock()
        hit2.index = 1
        hit2.relevance_score = 0.5 # Should be filtered out
        
        mock_response.results = [hit1, hit2]
        self.mock_cohere_client.rerank.return_value = mock_response
        
        results = self.service._find_relevant_chunks("query", "en")
        
        # Verify hybrid search called with alpha=0.5 and limit=25
        self.mock_vector_store.hybrid_search.assert_called_with("query", limit=25, alpha=0.5)
        
        # Verify rerank called
        self.mock_cohere_client.rerank.assert_called()
        
        # Verify filtering (only score > 0.7)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "doc1")
        self.assertEqual(results[0]["rerank_score"], 0.9)

    def test_find_relevant_chunks_rerank_failure(self):
        """Test fallback when re-ranking fails."""
        initial_results = [{"text": "doc1"}]
        self.mock_vector_store.hybrid_search.return_value = initial_results
        
        # Simulate exception
        self.mock_cohere_client.rerank.side_effect = Exception("API Error")
        
        results = self.service._find_relevant_chunks("query", "en")
        
        # Should return initial results (fallback)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "doc1")

    def test_build_context_text(self):
        """Test context string construction with header paths."""
        chunks = [
            {"text": "content", "source": "doc.md", "header_path": "Title > Section"}
        ]
        
        context = self.service._build_context_text(chunks)
        
        self.assertIn("[Source: doc.md | Section: Title > Section]", context)
        self.assertIn("content", context)

if __name__ == '__main__':
    unittest.main()
