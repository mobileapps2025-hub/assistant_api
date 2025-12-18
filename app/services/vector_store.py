import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery
from typing import List, Dict, Any, Optional
import os
from app.core.config import WEAVIATE_URL, WEAVIATE_API_KEY, client as openai_client
from app.core.logging import get_logger

logger = get_logger(__name__)

class VectorStoreService:
    """
    Service for managing the Weaviate vector store.
    """
    
    COLLECTION_NAME = "MCL_Document"

    def __init__(self):
        """
        Initialize the VectorStoreService with Weaviate connection.
        """
        if not WEAVIATE_URL:
            logger.warning("WEAVIATE_URL not set. Vector store will not function.")
            self.client = None
            return

        try:
            # Connect to Weaviate Cloud or Local
            headers = {}
            if WEAVIATE_API_KEY:
                headers["X-OpenAI-Api-Key"] = os.getenv("OPENAI_API_KEY") # Pass OpenAI key for vectorization if using Weaviate's module
            
            auth_config = weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None

            # Determine connection type based on URL
            if "weaviate.cloud" in WEAVIATE_URL:
                 self.client = weaviate.connect_to_wcs(
                    cluster_url=WEAVIATE_URL,
                    auth_credentials=auth_config,
                    headers=headers
                )
            else:
                # Local connection
                host = WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0]
                self.client = weaviate.connect_to_local(
                    host=host,
                    headers=headers
                )
            
            logger.info(f"Connected to Weaviate at {WEAVIATE_URL}")
            self.ensure_schema()
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            logger.info("Tip: If you are not running Weaviate locally (e.g. Docker), consider using a free Weaviate Cloud Sandbox (https://console.weaviate.cloud).")
            self.client = None

    def ensure_schema(self):
        """
        Ensure the collection schema exists.
        """
        if not self.client:
            return

        try:
            if not self.client.collections.exists(self.COLLECTION_NAME):
                logger.info(f"Creating collection {self.COLLECTION_NAME}...")
                self.client.collections.create(
                    name=self.COLLECTION_NAME,
                    vectorizer_config=Configure.Vectorizer.text2vec_openai(), # Use OpenAI for embeddings
                    generative_config=Configure.Generative.openai(),
                    properties=[
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="header_path", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                    ]
                )
                logger.info(f"Collection {self.COLLECTION_NAME} created.")
            else:
                logger.info(f"Collection {self.COLLECTION_NAME} already exists.")
        except Exception as e:
            logger.error(f"Error ensuring schema: {e}")

    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Batch upload chunks to Weaviate.
        
        Args:
            chunks: List of dictionaries containing 'text', 'header_path', 'source', etc.
        """
        if not self.client:
            logger.error("Weaviate client not initialized.")
            return False

        collection = self.client.collections.get(self.COLLECTION_NAME)
        
        try:
            with collection.batch.dynamic() as batch:
                for chunk in chunks:
                    batch.add_object(
                        properties={
                            "text": chunk.get("text", ""),
                            "header_path": chunk.get("header_path", ""),
                            "source": chunk.get("source", ""),
                            "chunk_index": chunk.get("chunk_index", 0)
                        }
                        # Weaviate handles vectorization automatically via text2vec-openai
                    )
            
            if len(collection.batch.failed_objects) > 0:
                logger.error(f"Failed to import {len(collection.batch.failed_objects)} objects.")
                for failed in collection.batch.failed_objects:
                    logger.error(f"Failed object: {failed.message}")
                return False
                
            logger.info(f"Successfully imported {len(chunks)} chunks.")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def hybrid_search(self, query: str, alpha: float = 0.5, limit: int = 25) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (Vector + Keyword).
        
        Args:
            query: Search query.
            alpha: Weight for vector search (0 = pure keyword, 1 = pure vector).
            limit: Number of results.
        """
        if not self.client:
            return []

        collection = self.client.collections.get(self.COLLECTION_NAME)
        
        try:
            response = collection.query.hybrid(
                query=query,
                alpha=alpha,
                limit=limit,
                return_metadata=MetadataQuery(score=True)
            )
            
            results = []
            for obj in response.objects:
                results.append({
                    "text": obj.properties.get("text"),
                    "header_path": obj.properties.get("header_path"),
                    "source": obj.properties.get("source"),
                    "score": obj.metadata.score,
                    "uuid": str(obj.uuid)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            return []

    def delete_collection(self):
        """Delete the entire collection (for reset)."""
        if not self.client:
            return
        try:
            self.client.collections.delete(self.COLLECTION_NAME)
            logger.info(f"Deleted collection {self.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def close(self):
        """Close the client connection."""
        if self.client:
            self.client.close()
