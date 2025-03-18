import time
import asyncio
from schemas.models import Ticket, AgentResponse, HumanFeedback
from agent.workflows import CustomerSupportWorkflow
from agent.actions import ActionHandler
from database.vector_store import VectorStore
from config.settings import CONCURRENCY_LIMIT, EMBEDDING_MODEL
from dotenv import load_dotenv
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from utils.redis_utils import rate_limit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent")

load_dotenv()

class CustomerSupportAgent:
    def __init__(self):
        self.workflow = CustomerSupportWorkflow()
        self.action_handler = ActionHandler()
        self.vector_store = VectorStore()
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        self.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    @rate_limit(limit=50, period=60)  # Allow 50 ticket processings per minute
    async def process_ticket(self, ticket: Ticket) -> AgentResponse:
        """Process a customer support ticket asynchronously."""
        async with self.semaphore:
            start_time = time.time()
            
            logger.info(f"Started processing ticket {ticket.ticket_id} - customer_id: {ticket.customer_id}, subject: {ticket.subject}, tags: {ticket.tags}")
            
            # Add 'agent-processing' tag to indicate the ticket is being processed by the agent
            if "agent-processing" not in ticket.tags:
                ticket.tags.append("agent-processing")
            
            # Process ticket through the workflow
            agent_response = await self.workflow.process_ticket(ticket)
            
            # Store the processed ticket and agent response in the vector database
            await self._store_processed_ticket(ticket, agent_response)
            
            processing_time = time.time() - start_time
            logger.info(f"Completed processing ticket {ticket.ticket_id} - processing_time: {processing_time}")
            
            return agent_response
    
    @rate_limit(limit=100, period=60)  # Allow 100 feedback processings per minute
    async def process_human_feedback(
        self, 
        ticket: Ticket, 
        agent_response: AgentResponse, 
        feedback: HumanFeedback
    ) -> Ticket:
        """Process human feedback on the agent's response."""
        start_time = time.time()
        updated_ticket = ticket.model_copy()
        
        logger.info(f"Processing human feedback for ticket {ticket.ticket_id} - approved: {feedback.approved}, has_modified_actions: {feedback.modified_actions is not None}")
        
        if feedback.approved:
            # Add 'human-approved' tag
            if "human-approved" not in updated_ticket.tags:
                updated_ticket.tags.append("human-approved")
            
            # Apply the approved actions
            actions_to_apply = feedback.modified_actions or agent_response.actions
            updated_ticket = await self.action_handler.apply_approved_actions(
                updated_ticket, 
                actions_to_apply
            )
        else:
            # Add 'human-rejected' tag
            if "human-rejected" not in updated_ticket.tags:
                updated_ticket.tags.append("human-rejected")
        
        # Store the feedback in the vector database
        await self._store_human_feedback(updated_ticket, agent_response, feedback)
        
        processing_time = time.time() - start_time
        logger.info(f"Completed processing human feedback for ticket {ticket.ticket_id} - processing_time: {processing_time}, updated_tags: {updated_ticket.tags}")
        
        return updated_ticket
    
    async def _store_processed_ticket(
        self, 
        ticket: Ticket, 
        agent_response: AgentResponse
    ):
        """Store the processed ticket and agent response in the vector database."""
        start_time = time.time()
        
        # Combine ticket content and agent response for embedding
        combined_text = f"""
        Ticket: {ticket.subject}
        Content: {ticket.content}
        Context Summary: {agent_response.context_summary}
        Suggested Response: {agent_response.suggested_response}
        """
        
        # Prepare metadata
        metadata = {
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "subject": ticket.subject,
            "priority": ticket.priority.value,
            "status": ticket.status.value,
            "tags": ",".join(ticket.tags),
            "created_at": ticket.created_at.isoformat(),
            "actions": ",".join([action.value for action in agent_response.actions]),
            "processing_time": agent_response.processing_time
        }
        
        # Store in vector database
        await self.vector_store.store_ticket(
            ticket_id=ticket.ticket_id,
            text_to_embed=combined_text,
            metadata=metadata,
            embedding_model=self.embedding_model
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Stored processed ticket {ticket.ticket_id} in vector database - processing_time: {processing_time}")
    
    async def _store_human_feedback(
        self, 
        ticket: Ticket, 
        agent_response: AgentResponse, 
        feedback: HumanFeedback
    ):
        """Store human feedback in the vector database."""
        start_time = time.time()
        
        # Combine feedback information for embedding
        combined_text = f"""
        Ticket: {ticket.subject}
        Content: {ticket.content}
        Agent Suggested Response: {agent_response.suggested_response}
        Human Feedback: {feedback.feedback or ""}
        Modified Response: {feedback.modified_response or ""}
        Approved: {"Yes" if feedback.approved else "No"}
        """
        
        # Prepare metadata
        metadata = {
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "subject": ticket.subject,
            "priority": ticket.priority.value,
            "status": ticket.status.value,
            "tags": ",".join(ticket.tags),
            "created_at": ticket.created_at.isoformat(),
            "agent_response_id": feedback.agent_response_id,
            "actions": ",".join([action.value for action in agent_response.actions]),
            "approved": feedback.approved,
            "modified_actions": ",".join([action.value for action in (feedback.modified_actions or [])]),
            "processing_time": agent_response.processing_time,
            "timestamp": time.time()
        }
        
        await self.vector_store.store_ticket_feedback(
            ticket_id=ticket.ticket_id,
            text_to_embed=combined_text,
            metadata=metadata,
            embedding_model=self.embedding_model
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Stored human feedback for ticket {ticket.ticket_id} in vector database - processing_time: {processing_time}, approved: {feedback.approved}")