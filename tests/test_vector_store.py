import pytest
from unittest.mock import MagicMock, patch
from app.services.vector_store import VectorStoreService


class TestVectorStoreServiceInit:

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    @patch("app.services.vector_store.WEAVIATE_API_KEY", "")
    def test_initialization_local(self, mock_weaviate):
        """Local URL → connect_to_local; client is stored."""
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client

        service = VectorStoreService()

        mock_weaviate.connect_to_local.assert_called_once()
        assert service.client is mock_client

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "https://cluster.weaviate.cloud")
    @patch("app.services.vector_store.WEAVIATE_API_KEY", "test-key")
    def test_initialization_cloud(self, mock_weaviate):
        """Cloud URL → connect_to_wcs."""
        mock_client = MagicMock()
        mock_weaviate.connect_to_wcs.return_value = mock_client

        service = VectorStoreService()

        mock_weaviate.connect_to_wcs.assert_called_once()
        assert service.client is mock_client

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_client_is_none_when_connection_fails(self, mock_weaviate):
        """Connection failure → client remains None; no exception raised."""
        mock_weaviate.connect_to_local.side_effect = Exception("refused")

        service = VectorStoreService()

        assert service.client is None


class TestVectorStoreAddDocuments:

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_add_documents_batches_correctly(self, mock_weaviate):
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        mock_batch = MagicMock()
        mock_collection.batch.dynamic.return_value.__enter__.return_value = mock_batch
        mock_collection.batch.failed_objects = []

        service = VectorStoreService()
        chunks = [
            {"text": "chunk1", "header_path": "H1", "source": "doc1", "chunk_index": 0},
            {"text": "chunk2", "header_path": "H1", "source": "doc1", "chunk_index": 1},
        ]

        success = service.add_documents(chunks)

        assert success is True
        assert mock_batch.add_object.call_count == 2

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_add_documents_returns_false_when_client_none(self, mock_weaviate):
        mock_weaviate.connect_to_local.side_effect = Exception("refused")

        service = VectorStoreService()
        result = service.add_documents([{"text": "x"}])

        assert result is False


class TestVectorStoreHybridSearch:

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_hybrid_search_returns_results(self, mock_weaviate):
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        mock_obj = MagicMock()
        mock_obj.properties = {
            "text": "result text",
            "header_path": "H1",
            "source": "doc.md",
        }
        mock_obj.metadata.score = 0.9
        mock_obj.uuid = "abc-123"

        mock_collection.query.hybrid.return_value = MagicMock(objects=[mock_obj])

        service = VectorStoreService()
        results = service.hybrid_search("create a task")

        assert len(results) == 1
        assert results[0]["text"] == "result text"
        assert results[0]["score"] == 0.9

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_hybrid_search_returns_empty_when_client_none(self, mock_weaviate):
        mock_weaviate.connect_to_local.side_effect = Exception("refused")

        service = VectorStoreService()
        results = service.hybrid_search("test")

        assert results == []
