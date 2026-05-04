import os
import glob
from typing import List, Dict, Any
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from app.services.vector_store import VectorStoreService
from app.core.logging import get_logger
import pypdf

logger = get_logger(__name__)

class IngestionService:
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store = vector_store_service

    def _detect_doc_type(self, file_path: str) -> str:
        """Classify a file into a doc_type for dual-track retrieval."""
        # Normalise separators for consistent sub-path matching
        norm = file_path.replace("\\", "/")
        filename = os.path.basename(norm).lower()

        if "/visual/" in norm:
            return "visual_guide"
        if "identity" in filename:
            return "assistant_identity"
        if "platform" in filename:
            return "platform_note"
        return "faq"

    def load_pdf_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a PDF file and split it into chunks.
        Per-page metadata is extracted so that chunks carry a meaningful header_path
        instead of the generic "PDF Content" label, improving retrieval context.
        """
        try:
            reader = pypdf.PdfReader(file_path)
            filename = os.path.basename(file_path)
            doc_type = self._detect_doc_type(file_path)

            # Extract per-page text while tracking page number
            page_texts: List[tuple] = []  # (page_num, text)
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_texts.append((page_num, page_text))

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

            chunks = []
            chunk_index = 0

            for page_num, page_text in page_texts:
                docs = text_splitter.create_documents([page_text])
                for doc in docs:
                    content = doc.page_content.strip()
                    if not content:
                        continue

                    # Detect section header from the first meaningful line of the chunk.
                    # A header is: ALL CAPS line, a line starting with Q: or ending in ?,
                    # or a short line (≤80 chars) that looks like a title.
                    first_line = content.split("\n")[0].strip()
                    if (
                        first_line.isupper()
                        or first_line.startswith("Q:")
                        or first_line.endswith("?")
                        or (len(first_line) <= 80 and not first_line.endswith("."))
                    ):
                        section_header = first_line[:60]
                    else:
                        section_header = f"Page {page_num}"

                    header_path = f"Page {page_num} > {section_header}"

                    chunks.append({
                        "text": content,
                        "header_path": header_path,
                        "source": filename,
                        "chunk_index": chunk_index,
                        "doc_type": doc_type,
                    })
                    chunk_index += 1

            return chunks

        except Exception as e:
            logger.error(f"Error processing PDF file {file_path}: {e}")
            return []

    def load_and_split_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load a Markdown file and split it by headers.
        Returns a list of chunks with metadata.

        For `visual_guide` documents, splitting is bypassed entirely: each topical
        guide is a small, purpose-built unit (typically 600–1500 chars) where the
        prose and the inline `![…](images/…)` link MUST stay together so the
        retriever returns prose+image as a single context chunk. Splitting them
        causes the LLM to receive the prose without the image markdown and
        therefore omit the visual aid from the answer.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            doc_type = self._detect_doc_type(file_path)
            filename = os.path.basename(file_path)

            if doc_type == "visual_guide":
                # Try to extract the H1 (or topic from front-matter) for header_path
                header_path = "Root"
                for line in text.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("# "):
                        header_path = stripped[2:].strip()
                        break
                return [{
                    "text": text,
                    "header_path": header_path,
                    "source": filename,
                    "chunk_index": 0,
                    "doc_type": doc_type,
                }]

            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]

            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            docs = markdown_splitter.split_text(text)

            # Secondary split: long sections that exceed chunk_size get split further with overlap.
            # Image-aware separators ensure breaks happen before Markdown image links, not inside
            # the surrounding context.
            secondary_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n![", "\n![", "\n\n", "\n", " ", ""],
            )

            chunks = []
            chunk_index = 0

            for doc in docs:
                header_parts = []
                if "Header 1" in doc.metadata:
                    header_parts.append(doc.metadata["Header 1"])
                if "Header 2" in doc.metadata:
                    header_parts.append(doc.metadata["Header 2"])
                if "Header 3" in doc.metadata:
                    header_parts.append(doc.metadata["Header 3"])

                header_path = " > ".join(header_parts) if header_parts else "Root"

                # Split long sections so no single chunk exceeds the size limit
                if len(doc.page_content) > 800:
                    sub_docs = secondary_splitter.create_documents([doc.page_content])
                else:
                    sub_docs = [doc]

                for sub_doc in sub_docs:
                    chunks.append({
                        "text": sub_doc.page_content,
                        "header_path": header_path,
                        "source": filename,
                        "chunk_index": chunk_index,
                        "doc_type": doc_type,
                    })
                    chunk_index += 1

            return chunks

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []

    def ingest_all(self, directory_path: str, reset_collection: bool = False) -> Dict[str, Any]:
        """
        Ingest all Markdown files from the specified directory.
        """
        logger.info(f"Starting ingestion from {directory_path}...")
        
        # Ensure schema exists before ingestion
        if reset_collection:
            logger.info("Resetting vector collection before ingestion.")
            self.vector_store.delete_collection()
        self.vector_store.ensure_schema()

        all_chunks = []
        processed_files = 0
        failed_files = 0

        # Find all markdown and PDF files
        md_files = glob.glob(os.path.join(directory_path, "**/*.md"), recursive=True)
        pdf_files = glob.glob(os.path.join(directory_path, "**/*.pdf"), recursive=True)
        
        all_files = md_files + pdf_files
        
        for file_path in all_files:
            if os.path.basename(file_path).lower() == "topics.md":
                logger.info(f"Skipping internal registry file {file_path}.")
                continue

            logger.info(f"Processing {file_path}...")
            
            if file_path.lower().endswith('.pdf'):
                chunks = self.load_pdf_document(file_path)
            else:
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
