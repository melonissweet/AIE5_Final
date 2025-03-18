import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http.models import VectorParams, Distance
from langchain_core.documents import Document
from config.settings import (
    QDRANT_URL, 
    QDRANT_API_KEY, 
    TICKETS_COLLECTION,
    TICKETS_FEEDBACK_COLLECTION,
    COLLECTION_NAME,
    OPENAI_API_KEY, 
    LLM_MODEL, 
    TEMPERATURE,
    MAX_TOKENS
)
from utils.redis_utils import rate_limit
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers import EnsembleRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vector_store")
# logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self._ensure_collections_exist()
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            cache=True  # Explicitly enable caching
        )

    def _ensure_collections_exist(self):
        """Make sure all required collections exist in Qdrant."""
        collections = [
            TICKETS_COLLECTION,
            TICKETS_FEEDBACK_COLLECTION,
            COLLECTION_NAME
        ]
        
        existing_collections = self.client.get_collections().collections
        existing_collection_names = [c.name for c in existing_collections]
        
        for collection in collections:
            if collection not in existing_collection_names:
                self.client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
                )
                logger.info(f"Created collection: {collection}")

    async def search(
        self, 
        collection_name: str, 
        query: str, 
        embedding_model: Any,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search the vector database for similar vectors."""
        logger.info(f"Searching collection {collection_name} with query length {len(query)}, limit {limit}")

        qdrant_vs = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embedding_model#cached_embedder
        )
        
        naive_retriever = qdrant_vs.as_retriever(search_kwargs={"k" : limit})
        multi_query_retriever = MultiQueryRetriever.from_llm(retriever=naive_retriever, llm=self.llm)

        compressor = CohereRerank(model="rerank-english-v3.0")
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=naive_retriever)

        retriever_list = [naive_retriever, multi_query_retriever, compression_retriever]
        equal_weighting = [1/len(retriever_list)] * len(retriever_list)
        ensemble_retriever = EnsembleRetriever(retrievers=retriever_list, weights=equal_weighting)

        search_results = await ensemble_retriever.ainvoke(query)

        results = []
        for doc in search_results:
            result = {
                "metadata": doc.metadata,
                "payload": doc.page_content
            }
            results.append(result)
            
        return results

    async def search_multiple_collections(
        self, 
        collection_names: List[str], 
        embedding_model: Any,
        query: str,
        limit: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search multiple collections and return results by collection."""
        logger.info(f"Searching multiple collections: {collection_names}")

        results = {}

        for collection in collection_names:
            logger.info(f"Searching collection: {collection} with query length {len(query)}, limit {limit}")
            
            qdrant_vs = QdrantVectorStore(
                client=self.client,
                collection_name=collection,
                embedding=embedding_model#cached_embedder
            )
            naive_retriever = qdrant_vs.as_retriever(search_kwargs={"k" : limit})
            multi_query_retriever = MultiQueryRetriever.from_llm(retriever=naive_retriever, llm=self.llm)

            compressor = CohereRerank(model="rerank-english-v3.0")
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=naive_retriever)

            retriever_list = [naive_retriever, multi_query_retriever, compression_retriever]
            equal_weighting = [1/len(retriever_list)] * len(retriever_list)
            ensemble_retriever = EnsembleRetriever(retrievers=retriever_list, weights=equal_weighting)
            results[collection] = await ensemble_retriever.ainvoke(query)

        return results
    
    @rate_limit(limit=100, period=60)
    async def store_ticket(
        self,
        ticket_id: str,
        text_to_embed: str,
        metadata: Dict[str, Any],
        embedding_model: Any
    ):
        """Store processed ticket data in the vector database."""
        logger.info(f"Storing ticket {ticket_id} in vector database collection {TICKETS_COLLECTION}")
        
        document = Document(
            page_content=text_to_embed,
            metadata=metadata,
        )
        qdrant_vs = QdrantVectorStore(
                client=self.client,
                collection_name=TICKETS_COLLECTION,
                embedding=embedding_model#cached_embedder
            )
        await qdrant_vs.aadd_documents(documents=[document], ids=[ticket_id])

    @rate_limit(limit=100, period=60)
    async def store_ticket_feedback(
        self,
        ticket_id: str,
        text_to_embed: str,
        metadata: Dict[str, Any],
        embedding_model: Any
    ):
        """Store processed ticket data with feedback in the vector database."""
        logger.info(f"Storing ticket feedback {ticket_id} in vector database collection {TICKETS_FEEDBACK_COLLECTION}")
        
        document = Document(
            page_content=text_to_embed,
            metadata=metadata,
        )
        qdrant_vs = QdrantVectorStore(
                client=self.client,
                collection_name=TICKETS_FEEDBACK_COLLECTION,
                embedding=embedding_model#cached_embedder
            )
        
        await qdrant_vs.aadd_documents(documents=[document], ids=[ticket_id])

    async def get_similar_tickets(
        self,
        query: str,
        embedding_model: Any,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Find similar previous tickets."""
        logger.info(f"Finding similar tickets with query length {len(query)}, limit {limit}")
        
        qdrant_vs = QdrantVectorStore(
                client=self.client,
                collection_name=TICKETS_COLLECTION,
                embedding=embedding_model#cached_embedder
            )
        return await qdrant_vs.asimilarity_search(query=query, k=limit)