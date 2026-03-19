import pytest
from unittest.mock import MagicMock, patch, mock_open
from app.services.ingestion_service import IngestionService
from app.services.vector_store import VectorStoreService


@pytest.fixture
def ingestion_service():
    mock_vs = MagicMock(spec=VectorStoreService)
    mock_vs.add_documents.return_value = True
    return IngestionService(mock_vs), mock_vs


class TestLoadAndSplitDocument:

    def test_splits_markdown_by_headers(self, ingestion_service):
        service, _ = ingestion_service
        markdown = (
            "# Title\nIntro text.\n\n"
            "## Section 1\nSection 1 text.\n\n"
            "### Subsection A\nSubsection A text.\n\n"
            "## Section 2\nSection 2 text.\n"
        )
        with patch("builtins.open", mock_open(read_data=markdown)):
            chunks = service.load_and_split_document("dummy.md")

        assert len(chunks) > 0
        subsection_chunks = [c for c in chunks if "Subsection A" in c["header_path"]]
        assert len(subsection_chunks) == 1
        assert "Subsection A text" in subsection_chunks[0]["text"]
        assert "Section 1" in subsection_chunks[0]["header_path"]

    def test_returns_empty_list_on_file_error(self, ingestion_service):
        service, _ = ingestion_service
        with patch("builtins.open", side_effect=OSError("not found")):
            chunks = service.load_and_split_document("missing.md")
        assert chunks == []


class TestIngestAll:

    @patch("app.services.ingestion_service.glob.glob")
    def test_processes_md_and_pdf_files(self, mock_glob, ingestion_service):
        """glob is called twice (md then pdf); both lists are merged."""
        service, mock_vs = ingestion_service
        # First call → md files, second call → pdf files
        mock_glob.side_effect = [["doc1.md", "doc2.md"], []]

        dummy_chunk = {"text": "chunk", "header_path": "H1", "source": "doc", "chunk_index": 0}
        with patch.object(service, "load_and_split_document", return_value=[dummy_chunk]):
            result = service.ingest_all("dummy_dir")

        assert result["success"] is True
        assert result["processed_files"] == 2
        assert result["total_chunks"] == 2
        mock_vs.ensure_schema.assert_called_once()
        mock_vs.add_documents.assert_called_once()

    @patch("app.services.ingestion_service.glob.glob")
    def test_returns_failure_when_no_files(self, mock_glob, ingestion_service):
        service, _ = ingestion_service
        mock_glob.side_effect = [[], []]

        result = service.ingest_all("dummy_dir")

        assert result["success"] is False
        assert result["message"] == "No content found to ingest."

    @patch("app.services.ingestion_service.glob.glob")
    def test_returns_failure_when_add_documents_fails(self, mock_glob, ingestion_service):
        service, mock_vs = ingestion_service
        mock_glob.side_effect = [["doc1.md"], []]
        mock_vs.add_documents.return_value = False

        dummy_chunk = {"text": "chunk", "header_path": "H1", "source": "doc", "chunk_index": 0}
        with patch.object(service, "load_and_split_document", return_value=[dummy_chunk]):
            result = service.ingest_all("dummy_dir")

        assert result["success"] is False
        assert "Failed to upload" in result["message"]
