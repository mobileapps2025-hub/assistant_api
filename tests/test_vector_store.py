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

    @patch("app.services.vector_store.EmbeddedOptions")
    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "embedded")
    def test_initialization_embedded(self, mock_weaviate, mock_embedded_options):
        """Embedded URL → managed Weaviate client is connected and stored."""
        mock_client = MagicMock()
        mock_weaviate.WeaviateClient.return_value = mock_client
        mock_embedded_options.return_value = "embedded-options"

        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            with patch.object(VectorStoreService, "_patch_hosts_for_embedded"):
                service = VectorStoreService()

        mock_embedded_options.assert_called_once()
        mock_weaviate.WeaviateClient.assert_called_once_with(
            embedded_options="embedded-options",
            additional_headers={},
        )
        mock_client.connect.assert_called_once()
        assert service.client is mock_client
        env_vars = mock_embedded_options.call_args.kwargs["additional_env_vars"]
        assert env_vars["CLUSTER_ADVERTISE_ADDR"] == "127.0.0.1"
        assert env_vars["RAFT_BOOTSTRAP_EXPECT"] == "1"
        # CLUSTER_HOSTNAME, CLUSTER_GOSSIP_BIND_PORT, CLUSTER_DATA_BIND_PORT must NOT be
        # overridden here — the weaviate-client library sets them consistently with RAFT_JOIN
        # (which we cannot override without knowing the random raft_port).  Overriding
        # CLUSTER_HOSTNAME alone was the root cause of the Raft bootstrap loop on Azure.
        assert "CLUSTER_HOSTNAME" not in env_vars
        assert "CLUSTER_GOSSIP_BIND_PORT" not in env_vars
        assert "CLUSTER_DATA_BIND_PORT" not in env_vars

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_client_is_none_when_connection_fails(self, mock_weaviate):
        """Connection failure → client remains None; no exception raised."""
        mock_weaviate.connect_to_local.side_effect = Exception("refused")

        service = VectorStoreService()

        assert service.client is None

    @patch("app.services.vector_store.EmbeddedOptions")
    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_azure_localhost_url_uses_embedded(self, mock_weaviate, mock_embedded_options):
        """Azure runtime should never try localhost Weaviate."""
        mock_client = MagicMock()
        mock_weaviate.WeaviateClient.return_value = mock_client
        mock_embedded_options.return_value = "embedded-options"

        with patch.dict("os.environ", {"WEBSITE_INSTANCE_ID": "abc", "OPENAI_API_KEY": ""}):
            with patch.object(VectorStoreService, "_patch_hosts_for_embedded"):
                service = VectorStoreService()

        mock_weaviate.connect_to_local.assert_not_called()
        mock_weaviate.WeaviateClient.assert_called_once_with(
            embedded_options="embedded-options",
            additional_headers={},
        )
        assert service.weaviate_url == "embedded"
        assert service.client is mock_client

    @patch("app.services.vector_store.EmbeddedOptions")
    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://weaviate.internal:8080")
    def test_azure_connection_failure_falls_back_to_embedded(self, mock_weaviate, mock_embedded_options):
        """Azure connection failures should retry with embedded before giving up."""
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.side_effect = Exception("refused")
        mock_weaviate.WeaviateClient.return_value = mock_client
        mock_embedded_options.return_value = "embedded-options"

        with patch.dict("os.environ", {"WEBSITE_INSTANCE_ID": "abc", "OPENAI_API_KEY": ""}):
            with patch.object(VectorStoreService, "_patch_hosts_for_embedded"):
                service = VectorStoreService()

        mock_weaviate.connect_to_local.assert_called_once()
        mock_weaviate.WeaviateClient.assert_called_once_with(
            embedded_options="embedded-options",
            additional_headers={},
        )
        assert service.weaviate_url == "embedded"
        assert service.client is mock_client

    @patch("app.services.vector_store.EmbeddedOptions")
    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "embedded")
    def test_embedded_already_listening_falls_back_to_connect_to_local(self, mock_weaviate, mock_embedded_options):
        """When embedded ports are already in use, fall back to connect_to_local()."""
        from weaviate.exceptions import WeaviateStartUpError
        local_client = MagicMock()
        mock_weaviate.WeaviateClient.return_value = MagicMock(
            connect=MagicMock(side_effect=WeaviateStartUpError(
                "Embedded DB did not start because processes are already listening on ports http:8079 and grpc:50060"
            ))
        )
        mock_weaviate.connect_to_local.return_value = local_client
        mock_embedded_options.return_value = "embedded-options"

        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            with patch.object(VectorStoreService, "_patch_hosts_for_embedded"):
                service = VectorStoreService()

        mock_weaviate.connect_to_local.assert_called_once_with(
            port=8079, grpc_port=50060, headers={}
        )
        assert service.client is local_client

    @patch("app.services.vector_store.WEAVIATE_URL", "embedded")
    def test_patch_hosts_writes_entry_when_absent(self):
        """_patch_hosts_for_embedded appends the loopback entry when missing."""
        service = VectorStoreService.__new__(VectorStoreService)
        import io
        fake_file_content = "127.0.0.1 localhost\n"
        written = []

        def fake_open(path, mode="r", **kw):
            if mode == "r":
                return io.StringIO(fake_file_content)
            buf = io.StringIO()
            buf.close = lambda: written.append(buf.getvalue())
            return buf

        with patch("builtins.open", side_effect=fake_open):
            service._patch_hosts_for_embedded(port=8079)

        assert any("Embedded_at_8079" in w for w in written)

    @patch("app.services.vector_store.WEAVIATE_URL", "embedded")
    def test_patch_hosts_skips_when_already_present(self):
        """_patch_hosts_for_embedded is a no-op when entry already exists."""
        service = VectorStoreService.__new__(VectorStoreService)
        import io
        fake_file_content = "127.0.0.1 localhost\n127.0.0.1 Embedded_at_8079\n"

        mock_open = MagicMock(return_value=io.StringIO(fake_file_content))
        with patch("builtins.open", mock_open):
            service._patch_hosts_for_embedded(port=8079)

        # open() called once (for reading), never for appending
        assert mock_open.call_count == 1


class TestVectorStoreAddDocuments:

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_ensure_schema_includes_source_title(self, mock_weaviate):
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        mock_client.collections.exists.return_value = False

        service = VectorStoreService()
        service.ensure_schema()

        properties = mock_client.collections.create.call_args.kwargs["properties"]
        property_names = [prop.name for prop in properties]
        assert "source_title" in property_names

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
            {"text": "chunk1", "header_path": "H1", "source": "doc1", "source_title": "Document One", "chunk_index": 0},
            {"text": "chunk2", "header_path": "H1", "source": "doc1", "chunk_index": 1},
        ]

        success = service.add_documents(chunks)

        assert success is True
        assert mock_batch.add_object.call_count == 2
        first_properties = mock_batch.add_object.call_args_list[0].kwargs["properties"]
        second_properties = mock_batch.add_object.call_args_list[1].kwargs["properties"]
        assert first_properties["source_title"] == "Document One"
        assert first_properties["doc_type"] == "faq"
        assert second_properties["source_title"] == "doc1"

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
            "source_title": "Guide",
            "doc_type": "faq",
            "chunk_index": 7,
        }
        mock_obj.metadata.score = 0.9
        mock_obj.uuid = "abc-123"

        mock_collection.query.hybrid.return_value = MagicMock(objects=[mock_obj])

        service = VectorStoreService()
        results = service.hybrid_search("create a task")

        assert len(results) == 1
        assert results[0]["text"] == "result text"
        assert results[0]["source_title"] == "Guide"
        assert results[0]["doc_type"] == "faq"
        assert results[0]["chunk_index"] == 7
        assert results[0]["score"] == 0.9

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_hybrid_search_falls_back_when_source_title_missing(self, mock_weaviate):
        mock_client = MagicMock()
        mock_weaviate.connect_to_local.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection

        mock_obj = MagicMock()
        mock_obj.properties = {
            "text": "result text",
            "header_path": "H1",
            "source": "legacy.md",
            "chunk_index": 3,
        }
        mock_obj.metadata.score = 0.8
        mock_obj.uuid = "legacy-123"

        mock_collection.query.hybrid.return_value = MagicMock(objects=[mock_obj])

        service = VectorStoreService()
        results = service.hybrid_search("legacy document")

        assert results[0]["source_title"] == "legacy.md"
        assert results[0]["chunk_index"] == 3

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_hybrid_search_returns_empty_when_client_none(self, mock_weaviate):
        mock_weaviate.connect_to_local.side_effect = Exception("refused")

        service = VectorStoreService()
        results = service.hybrid_search("test")

        assert results == []

    @patch("app.services.vector_store.weaviate")
    @patch("app.services.vector_store.WEAVIATE_URL", "http://localhost:8080")
    def test_hybrid_search_retries_when_client_none(self, mock_weaviate):
        first_client = MagicMock()
        second_client = MagicMock()
        mock_weaviate.connect_to_local.side_effect = [first_client, second_client]
        mock_collection = MagicMock()
        second_client.collections.get.return_value = mock_collection
        mock_collection.query.hybrid.return_value = MagicMock(objects=[])

        service = VectorStoreService()
        service.client = None
        results = service.hybrid_search("test")

        assert results == []
        assert mock_weaviate.connect_to_local.call_count == 2
        second_client.collections.get.assert_called_once_with(service.COLLECTION_NAME)
