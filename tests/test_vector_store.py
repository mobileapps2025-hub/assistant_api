import unittest
from unittest.mock import MagicMock, patch
from app.services.vector_store import VectorStoreService

class TestVectorStoreService(unittest.TestCase):
    
    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    @patch("app.services.vector_store.WEAVIATE_API_KEY", "test-key")
    def test_initialization_local(self, mock_weaviate):
        """Test initialization with local URL."""
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        
        service = VectorStoreService()
        
        mock_weaviate.connect_to_local.assert_called_once()
        self.assertEqual(service.client, mock_client)
        # Ensure schema is called
        mock_client.collections.exists.assert_called_with("MCL_Document")

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "https://cluster.weaviate.cloud")
    @patch("app.services.vector_store.WEAVIATE_API_KEY", "test-key")
    def test_initialization_cloud(self, mock_weaviate):
        """Test initialization with cloud URL."""
        mock_client = MagicMock()
        mock_weaviate.connect_to_wcs.return_value = mock_client
        
        service = VectorStoreService()
        
        mock_weaviate.connect_to_wcs.assert_called_once()
        self.assertEqual(service.client, mock_client)

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_add_documents(self, mock_weaviate):
        """Test adding documents."""
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        
        # Mock batch context manager
        mock_batch = MagicMock()
        mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
        mock_collection.batch.failed_objects = []
        
        service = VectorStoreService()
        chunks = [
            {"text": "chunk1", "header_path": "H1", "source": "doc1", "chunk_index": 0},
            {"text": "chunk2", "header_path": "H1", "source": "doc1", "chunk_index": 1}
        ]
        
        success = service.add_documents(chunks)
        
        self.assertTrue(success)
        self.assertEqual(mock_batch.add_object.call_count, 2)

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_hybrid_search(self, mock_weaviate):
        """Test hybrid search."""
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        
        # Mock search response
        mock_obj = MagicMock()
        mock_obj.properties = {"text": "result", "header_path": "H1", "source": "doc1"}
        mock_obj.metadata.score = 0.9
        mock_obj.uuid = "123"
        
        mock_response = MagicMock()
        mock_response.objects = [mock_obj]
        mock_collection.query.hybrid.return_value = mock_response
        
        service = VectorStoreService()
        results = service.hybrid_search("query")
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "result")
        self.assertEqual(results[0]["score"], 0.9)
