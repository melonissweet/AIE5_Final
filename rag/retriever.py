from typing import List, Dict, Any
import logging
from database.vector_store import VectorStore
from llm.llm_client import LLMClient
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import (
    COLLECTION_NAME,
    RAG_TOP_K,
    EMBEDDING_MODEL
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("retriever")

class ContextRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm_client = LLMClient()
        # self.model_kwargs = {'trust_remote_code': True}
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
    async def retrieve_relevant_context(
        self, 
        ticket_content: str,
        company_specific: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context for the given ticket from multiple collections."""
        logger.info(
            f"Retrieving relevant context - content_length: {len(ticket_content)}, company_specific: {company_specific}"
        )
        
        # Define collections to search based on ticket content
        collections_to_search = [
            # COLLECTION_NAME
        ]
        
        # Add company content collection if company-specific tag is present
        if company_specific:
            collections_to_search.append(COLLECTION_NAME)
        
        # Search across all relevant collections
        all_results = await self.vector_store.search_multiple_collections(
            collection_names=collections_to_search,
            embedding_model=self.embedding_model,
            query=ticket_content,
            limit=RAG_TOP_K
        )
        # Flatten and format results
        formatted_results = []
        for collection, results in all_results.items():
            for doc in results:
                formatted_results.append(
                    {
                    "source": collection,
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                )
        
        return formatted_results
    
    async def retrieve_and_process_context(
        self, 
        ticket_content: str,
        company_specific: bool = True
    ) -> Dict[str, Any]:
        """Retrieve context and generate a summary and suggested response."""
        logger.info(
            f"Retrieving and processing context - content_length: {len(ticket_content)}, company_specific: {company_specific}"
        )
        
        # Get relevant contexts
        retrieved_contexts = await self.retrieve_relevant_context(
            ticket_content=ticket_content,
            company_specific=company_specific
        )
        
        if not retrieved_contexts:
            logger.info(
                f"No relevant context found - content_sample: {ticket_content[:100]}"
            )
            
            return {
                "context_summary": "No relevant context found.",
                "suggested_response": "I'm sorry, but I couldn't find specific information to assist with this inquiry.",
                "retrieved_contexts": []
            }
        
        # Generate context summary
        context_summary = await self.llm_client.generate_context_summary(
            ticket_content=ticket_content,
            retrieved_contexts=retrieved_contexts
        )
        
        # Generate suggested response
        suggested_response = await self.llm_client.generate_suggested_response(
            ticket_content=ticket_content,
            context_summary=context_summary
        )
        
        # Identify required actions
        actions = await self.llm_client.identify_required_actions(
            ticket_content=ticket_content,
            context_summary=context_summary
        )
        
        logger.info(
            f"Completed retrieving and processing context - content_sample: {ticket_content[:100]}, context_count: {len(retrieved_contexts)}, actions_count: {len(actions)}"
        )
        
        return {
            "context_summary": context_summary,
            "suggested_response": suggested_response,
            "actions": actions,
            "retrieved_contexts": retrieved_contexts
        }