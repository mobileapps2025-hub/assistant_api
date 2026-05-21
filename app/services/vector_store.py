import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery, HybridFusion, Filter
from weaviate.embedded import EmbeddedOptions
from typing import List, Dict, Any, Optional
import os
import urllib.parse
from app.core.config import WEAVIATE_URL, WEAVIATE_API_KEY, client as openai_client, SEARCH_LIMIT, SEARCH_ALPHA, MIN_SEARCH_SCORE
from app.core.logging import get_logger

logger = get_logger(__name__)

class VectorStoreService:
    """
    Service for managing the Weaviate vector store.
    """
    
    COLLECTION_NAME = "MCL_Document_v2"

    def __init__(self):
        """
        Initialize the VectorStoreService with Weaviate connection.
        """
        self.client = None
        self._source_title_supported: Optional[bool] = None
        self.weaviate_url = self._normalize_weaviate_url(WEAVIATE_URL)

        if not WEAVIATE_URL:
            logger.warning("WEAVIATE_URL not set. Vector store will not function.")
            return

        if self.weaviate_url != WEAVIATE_URL:
            logger.warning(
                f"[WEAVIATE] Using effective WEAVIATE_URL='{self.weaviate_url}' "
                f"instead of configured WEAVIATE_URL='{WEAVIATE_URL}'"
            )

        logger.info(f"[WEAVIATE] Initializing — WEAVIATE_URL='{self.weaviate_url}'")
        self._connect()

    def _build_headers(self) -> Dict[str, str]:
        """Build headers required by Weaviate vectorizer modules."""
        headers = {}
        if os.getenv("OPENAI_API_KEY"):
            headers["X-OpenAI-Api-Key"] = os.getenv("OPENAI_API_KEY")
        return headers

    def _is_azure_runtime(self) -> bool:
        return any(
            os.getenv(name)
            for name in (
                "WEBSITE_SITE_NAME",
                "WEBSITE_INSTANCE_ID",
                "WEBSITE_HOSTNAME",
                "APPSETTING_WEBSITE_SITE_NAME",
            )
        )

    def _is_local_url(self, url: str) -> bool:
        normalized = url.strip().lower()
        return normalized.startswith(("http://localhost", "http://127.0.0.1"))

    def _normalize_weaviate_url(self, url: str) -> str:
        if self._is_azure_runtime() and self._is_local_url(url):
            return "embedded"
        return url

    def _patch_hosts_for_embedded(self, port: int = 8079) -> None:
        """
        Add '127.0.0.1 Embedded_at_{port}' to /etc/hosts so that Weaviate's
        Raft bootstrap can resolve the embedded-node hostname.

        The weaviate-client library names the embedded node 'Embedded_at_{port}'
        and sets RAFT_JOIN=Embedded_at_{port}:{raft_port}.  On Azure App Service
        (no DNS for that name) Raft loops forever on "unable to resolve any node
        address to join".  Adding the loopback entry fixes it.
        """
        hostname = f"Embedded_at_{port}"
        entry = f"127.0.0.1 {hostname}\n"
        try:
            with open("/etc/hosts", "r") as f:
                if hostname in f.read():
                    logger.debug(f"[WEAVIATE] /etc/hosts already contains {hostname}")
                    return
            with open("/etc/hosts", "a") as f:
                f.write(entry)
            logger.info(f"[WEAVIATE] Patched /etc/hosts: {entry.strip()}")
        except Exception as e:
            logger.warning(f"[WEAVIATE] Could not patch /etc/hosts: {e}")

    def _connect_embedded(self, headers: Dict[str, str]):
        # Embedded mode: the weaviate-client downloads and manages the Weaviate binary.
        # Binary is cached at /home/.cache/weaviate-embedded (persistent on Azure App Service).
        # Data is stored at /home/weaviate_data (also persistent on Azure).
        # First startup downloads the binary once (~100 MB); subsequent starts reuse it.
        embedded_port = 8079
        embedded_grpc_port = 50060
        data_path = os.getenv("WEAVIATE_EMBEDDED_DATA_PATH", "/home/weaviate_data")
        binary_path = os.getenv("WEAVIATE_EMBEDDED_BINARY_PATH", "/home/.cache/weaviate-embedded")

        # Patch /etc/hosts BEFORE starting the binary.
        # The weaviate-client library sets CLUSTER_HOSTNAME=Embedded_at_{port} and
        # RAFT_JOIN=Embedded_at_{port}:{raft_port} by default (via setdefault).
        # Do NOT override CLUSTER_HOSTNAME or the gossip/data/raft ports here —
        # that would break the consistency between CLUSTER_HOSTNAME and RAFT_JOIN.
        # Instead we keep the library defaults and make the hostname resolvable.
        self._patch_hosts_for_embedded(port=embedded_port)

        logger.info(f"[WEAVIATE] Starting embedded mode — binary: {binary_path}, data: {data_path}")
        try:
            self.client = weaviate.WeaviateClient(
                embedded_options=EmbeddedOptions(
                    port=embedded_port,
                    grpc_port=embedded_grpc_port,
                    persistence_data_path=data_path,
                    binary_path=binary_path,
                    additional_env_vars={
                        "ENABLE_MODULES": "text2vec-openai,generative-openai",
                        "DEFAULT_VECTORIZER_MODULE": "none",
                        "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED": "true",
                        "QUERY_DEFAULTS_LIMIT": "25",
                        # Keep CLUSTER_ADVERTISE_ADDR explicit so Azure (no private IP) uses loopback.
                        # Do NOT set CLUSTER_HOSTNAME, CLUSTER_GOSSIP_BIND_PORT, CLUSTER_DATA_BIND_PORT —
                        # those are set consistently by the library and must not be split from RAFT_JOIN.
                        "CLUSTER_ADVERTISE_ADDR": os.getenv("WEAVIATE_CLUSTER_ADVERTISE_ADDR", "127.0.0.1"),
                        "RAFT_BOOTSTRAP_EXPECT": "1",
                    },
                ),
                additional_headers=headers,
            )
            self.client.connect()
            logger.info("[WEAVIATE] Embedded Weaviate started and connected successfully.")
        except Exception as e:
            if "already listening on ports" in str(e):
                # A previous Gunicorn worker already launched embedded Weaviate and it
                # is still running (orphaned process on port 8079/50060).
                # Connect directly instead of trying to start a second instance.
                logger.warning(
                    f"[WEAVIATE] Embedded Weaviate already running on ports "
                    f"{embedded_port}/{embedded_grpc_port}. Falling back to connect_to_local()."
                )
                self.client = weaviate.connect_to_local(
                    port=embedded_port,
                    grpc_port=embedded_grpc_port,
                    headers=headers,
                )
                logger.info("[WEAVIATE] Connected to already-running embedded Weaviate via connect_to_local().")
            else:
                raise

    def _connect_cloud(self, headers: Dict[str, str], auth_config):
        self.client = weaviate.connect_to_wcs(
            cluster_url=self.weaviate_url,
            auth_credentials=auth_config,
            headers=headers,
        )
        logger.info(f"[WEAVIATE] Connected to Weaviate Cloud at {self.weaviate_url}")

    def _connect_local(self, headers: Dict[str, str]):
        parsed = urllib.parse.urlparse(self.weaviate_url)
        host_only = parsed.hostname or "localhost"
        port_only = parsed.port or 8080
        self.client = weaviate.connect_to_local(
            host=host_only,
            port=port_only,
            headers=headers,
        )
        logger.info(f"[WEAVIATE] Connected to local Weaviate at {self.weaviate_url}")

    def _connect(self):
        try:
            headers = self._build_headers()
            auth_config = weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None

            if self.weaviate_url.strip().lower() == "embedded":
                self._connect_embedded(headers)
            elif "weaviate.cloud" in self.weaviate_url:
                self._connect_cloud(headers, auth_config)
            else:
                self._connect_local(headers)

        except Exception as e:
            logger.error(f"[WEAVIATE] Failed to connect: {e}", exc_info=True)
            self.client = None
            if self._is_azure_runtime() and self.weaviate_url.strip().lower() != "embedded":
                logger.warning("[WEAVIATE] Retrying with embedded Weaviate after Azure connection failure.")
                self.weaviate_url = "embedded"
                self._connect()

    def _ensure_client(self) -> bool:
        if self.client:
            return True
        logger.warning("[WEAVIATE] Client is not initialized. Attempting reconnect before continuing.")
        self._connect()
        return self.client is not None

    def _collection_property_names(self, collection) -> Optional[set]:
        try:
            config = collection.config.get()
            properties = getattr(config, "properties", None)
            if isinstance(properties, dict):
                return set(properties.keys())
            if isinstance(properties, (list, tuple, set)):
                return {
                    prop.name
                    for prop in properties
                    if getattr(prop, "name", None)
                }
        except Exception as e:
            logger.debug(f"[WEAVIATE] Could not inspect collection properties: {e}")
        return None

    def _ensure_optional_property(self, collection, property_name: str, data_type) -> bool:
        property_names = self._collection_property_names(collection)
        if property_names is not None and property_name in property_names:
            return True

        try:
            collection.config.add_property(
                Property(name=property_name, data_type=data_type)
            )
            logger.info(f"[WEAVIATE] Added missing property '{property_name}' to {self.COLLECTION_NAME}.")
            return True
        except Exception as e:
            logger.warning(
                f"[WEAVIATE] Could not add optional property '{property_name}' "
                f"to existing collection {self.COLLECTION_NAME}: {e}"
            )
            return False

    def ensure_schema(self):
        """
        Ensure the collection schema exists.
        """
        if not self._ensure_client():
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
                        Property(name="source_title", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                        Property(name="doc_type", data_type=DataType.TEXT),
                    ]
                )
                self._source_title_supported = True
                logger.info(f"Collection {self.COLLECTION_NAME} created.")
            else:
                logger.info(f"Collection {self.COLLECTION_NAME} already exists.")
                collection = self.client.collections.get(self.COLLECTION_NAME)
                self._source_title_supported = self._ensure_optional_property(
                    collection,
                    "source_title",
                    DataType.TEXT,
                )
        except Exception as e:
            logger.error(f"Error ensuring schema: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        """
        if not self._ensure_client():
            return {"status": "disconnected", "count": 0}

        try:
            if not self.client.collections.exists(self.COLLECTION_NAME):
                return {"status": "connected", "schema_exists": False, "count": 0}
            
            collection = self.client.collections.get(self.COLLECTION_NAME)
            # Aggregate count
            count_result = collection.aggregate.over_all(total_count=True)
            count = count_result.total_count
            
            return {
                "status": "connected", 
                "schema_exists": True, 
                "count": count
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e), "count": 0}

    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Batch upload chunks to Weaviate.
        
        Args:
            chunks: List of dictionaries containing 'text', 'header_path', 'source', etc.
        """
        if not self._ensure_client():
            logger.error("Weaviate client not initialized.")
            return False

        collection = self.client.collections.get(self.COLLECTION_NAME)
        
        try:
            with collection.batch.dynamic() as batch:
                for chunk in chunks:
                    properties = {
                        "text": chunk.get("text", ""),
                        "header_path": chunk.get("header_path", ""),
                        "source": chunk.get("source", ""),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "doc_type": chunk.get("doc_type", "faq"),
                    }
                    if self._source_title_supported is not False:
                        properties["source_title"] = (
                            chunk.get("source_title")
                            or chunk.get("source")
                            or ""
                        )

                    batch.add_object(
                        properties=properties
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

    def hybrid_search(self, query: str, alpha: float = SEARCH_ALPHA, limit: int = SEARCH_LIMIT, doc_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (Vector + Keyword).
        
        Args:
            query: Search query.
            alpha: Weight for vector search (0 = pure keyword, 1 = pure vector).
            limit: Number of results.
            doc_types: Optional list of doc_type values to filter by. When None, searches all types.
        """
        if not self._ensure_client():
            logger.warning("Weaviate client is not initialized. Skipping search.")
            return []

        collection = self.client.collections.get(self.COLLECTION_NAME)
        
        try:
            filters = Filter.by_property("doc_type").contains_any(doc_types) if doc_types else None
            logger.info(f"Executing hybrid search in Weaviate. Query: '{query}', Alpha: {alpha}, Limit: {limit}, doc_types={doc_types}")
            response = collection.query.hybrid(
                query=query,
                alpha=alpha,
                limit=limit,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                return_metadata=MetadataQuery(score=True),
                filters=filters,
            )
            
            results = []
            for obj in response.objects:
                score = obj.metadata.score or 0.0
                if score < MIN_SEARCH_SCORE:
                    continue
                properties = obj.properties or {}
                source = properties.get("source")
                results.append({
                    "text": properties.get("text"),
                    "header_path": properties.get("header_path"),
                    "source": source,
                    "source_title": properties.get("source_title") or source,
                    "doc_type": properties.get("doc_type"),
                    "chunk_index": properties.get("chunk_index", 0),
                    "score": score,
                    "uuid": str(obj.uuid)
                })

            logger.info(f"Weaviate returned {len(results)} results (min_score={MIN_SEARCH_SCORE}, doc_types={doc_types}).")
            if results:
                logger.debug(f"Top result score: {results[0]['score']}")

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
