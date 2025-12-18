import os
import glob
from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter
from app.services.vector_store import VectorStoreService
from app.core.logging import get_logger

logger = get_logger(__name__)

class IngestionService:
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store = vector_store_service

    def load_and_split_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a Markdown file and split it by headers.
        Returns a list of chunks with metadata.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]

            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            docs = markdown_splitter.split_text(text)

            chunks = []
            filename = os.path.basename(file_path)

            for i, doc in enumerate(docs):
                # Construct header path from metadata
                header_parts = []
                if "Header 1" in doc.metadata:
                    header_parts.append(doc.metadata["Header 1"])
                if "Header 2" in doc.metadata:
                    header_parts.append(doc.metadata["Header 2"])
                if "Header 3" in doc.metadata:
                    header_parts.append(doc.metadata["Header 3"])
                
                header_path = " > ".join(header_parts) if header_parts else "Root"

                chunk = {
                    "text": doc.page_content,
                    "header_path": header_path,
                    "source": filename,
                    "chunk_index": i
                }
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []

    def ingest_all(self, directory_path: str) -> Dict[str, Any]:
        """
        Ingest all Markdown files from the specified directory.
        """
        logger.info(f"Starting ingestion from {directory_path}...")
        
        # Ensure schema exists before ingestion
        self.vector_store.ensure_schema()

        all_chunks = []
        processed_files = 0
        failed_files = 0

        # Find all markdown files
        md_files = glob.glob(os.path.join(directory_path, "**/*.md"), recursive=True)
        
        for file_path in md_files:
            logger.info(f"Processing {file_path}...")
            chunks = self.load_and_split_document(file_path)
            if chunks:
                all_chunks.extend(chunks)
                processed_files += 1
            else:
                failed_files += 1

        if not all_chunks:
            logger.warning("No chunks generated from documents.")
            return {
                "success": False,
                "message": "No content found to ingest.",
                "processed_files": processed_files,
                "failed_files": failed_files
            }

        # Batch upload to Weaviate
        logger.info(f"Uploading {len(all_chunks)} chunks to Weaviate...")
        success = self.vector_store.add_documents(all_chunks)

        if success:
            logger.info("Ingestion completed successfully.")
            return {
                "success": True,
                "message": f"Successfully ingested {len(all_chunks)} chunks from {processed_files} files.",
                "total_chunks": len(all_chunks),
                "processed_files": processed_files,
                "failed_files": failed_files
            }
        else:
            logger.error("Ingestion failed during upload.")
            return {
                "success": False,
                "message": "Failed to upload chunks to vector store.",
                "processed_files": processed_files,
                "failed_files": failed_files
            }
