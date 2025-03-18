from typing import Dict, List, Any, Optional,TypedDict, AsyncGenerator
import time
import uuid
import logging
from langgraph.graph import StateGraph, END
from schemas.models import Ticket, AgentResponse, ActionType
from rag.retriever import ContextRetriever
from agent.actions import ActionHandler
from langgraph.checkpoint.memory import MemorySaver
# from langsmith import traceable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent_workflow")

class AgentState(TypedDict):
    """State for the agent workflow."""
    ticket: Ticket
    company_specific: bool
    retrieved_context: Optional[List[Dict[str, Any]]]
    context_summary: Optional[str]
    suggested_response: Optional[str]
    identified_actions: Optional[List[str]]
    processing_time: float
    agent_response: Optional[AgentResponse]

class CustomerSupportWorkflow:
    def __init__(self):
        self.context_retriever = ContextRetriever()
        self.action_handler = ActionHandler()
        self.memory = MemorySaver() # Create a memory saver for checkpointing
        self.graph = self._build_workflow()
        
    # @traceable(name="build_workflow")
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for the customer support agent."""
        workflow = StateGraph(AgentState)
        
        # Node: Check if ticket is company-specific
        workflow.add_node("check_company_specific", self._check_company_specific)
        
        # Node: Retrieve context from RAG system
        workflow.add_node("retrieve_context", self._retrieve_context)
        
        # Node: Process context and generate response
        workflow.add_node("generate_response", self._generate_response)
        
        # Node: Create final agent response
        workflow.add_node("create_agent_response", self._create_agent_response)
        
        # Define the workflow edges
        workflow.add_edge("check_company_specific", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "create_agent_response")
        workflow.add_edge("create_agent_response", END)
        
        # Set the entry point
        workflow.set_entry_point("check_company_specific")
        
        # Compile the graph with the checkpointer
        graph_with_checkpoint = workflow.compile(checkpointer=self.memory)
        
        return graph_with_checkpoint
    
    # @traceable(name="check_company_specific")
    async def _check_company_specific(self, state: AgentState) -> AgentState:
        """Check if the ticket has a company-specific tag."""
        ticket = state["ticket"]
        
        logger.info(f"Checking if ticket {ticket.ticket_id} is company-specific - tags: {ticket.tags}")
        
        # Check if "company-specific" tag exists
        company_specific = "company-specific" in ticket.tags
        
        return {**state, "company_specific": company_specific}
    
    # @traceable(name="retrieve_context")
    async def _retrieve_context(self, state: AgentState) -> AgentState:
        """Retrieve relevant context from the RAG system."""
        ticket = state["ticket"]
        company_specific = state["company_specific"]
        ticket_tags = ",".join(ticket.tags)

        logger.info(f"Retrieving context for ticket {ticket.ticket_id} - company_specific: {company_specific}")

        combined_ticket_text = f"""
        Ticket: {ticket.subject}
        Content: {ticket.content}
        Tags: {ticket_tags}
        """
        
        start_time = time.time()
        
        # Retrieve context
        context_results = await self.context_retriever.retrieve_relevant_context(
            ticket_content=combined_ticket_text,
            company_specific=company_specific
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"Retrieved context for ticket {ticket.ticket_id} - context_count: {len(context_results)}, processing_time: {processing_time}")
        
        return {
            **state,
            "retrieved_context": context_results,
            "processing_time": processing_time
        }
    
    # @traceable(name="generate_response")
    async def _generate_response(self, state: AgentState) -> AgentState:
        """Generate a response based on the retrieved context."""
        ticket = state["ticket"]
        retrieved_context = state["retrieved_context"]
        
        logger.info(f"Generating response for ticket {ticket.ticket_id} - context_count: {len(retrieved_context)}")
        
        start_time = time.time()
        
        # Generate context summary
        context_summary = await self.context_retriever.llm_client.generate_context_summary(
            ticket_content=ticket.content,
            retrieved_contexts=retrieved_context
        )
        
        # Generate suggested response and actions
        response_action = await self.context_retriever.llm_client.generate_suggested_response(
            ticket_content=ticket.content,
            context_summary=context_summary
        )

        # Extract the individual components from the response
        suggested_response = response_action["suggested_response"]
        actions = response_action["required_actions"]
        
        processing_time = time.time() - start_time + state["processing_time"]
        
        logger.info(f"Generated response for ticket {ticket.ticket_id} - processing_time: {processing_time - state['processing_time']}, actions_count: {len(actions)}")
        
        return {
            **state,
            "context_summary": context_summary,
            "suggested_response": suggested_response,
            "identified_actions": actions,
            "processing_time": processing_time
        }
    
    # @traceable(name="create_agent_response")
    async def _create_agent_response(self, state: AgentState) -> AgentState:
        """Create the final agent response object."""
        ticket = state["ticket"]
        
        logger.info(f"Creating agent response for ticket {ticket.ticket_id}")
        
        # Convert action strings to ActionType enum
        actions = []
        for action_str in state["identified_actions"]:
            try:
                action = ActionType(action_str.strip())
                actions.append(action)
            except ValueError:
                # If invalid action, default to NO_ACTION
                if action_str.strip() and action_str.strip() != "no_action":
                    logger.warning(f"Invalid action type '{action_str}' for ticket {ticket.ticket_id}")
                
                # Only add NO_ACTION if no other actions
                if not actions:
                    actions.append(ActionType.NO_ACTION)
        
        # Create agent response
        agent_response = AgentResponse(
            ticket_id=ticket.ticket_id,
            context_summary=state["context_summary"],
            suggested_response=state["suggested_response"],
            actions=actions,
            retrieved_contexts=state["retrieved_context"],
            processing_time=state["processing_time"]
        )
        
        logger.info(f"Created agent response for ticket {ticket.ticket_id} - actions: {[a.value for a in actions]}")
        
        return {**state, "agent_response": agent_response}
    
    # @traceable(name="process_ticket")
    async def process_ticket(self, ticket: Ticket) -> AgentResponse:
        """Process a ticket through the entire workflow."""
        thread_id = str(uuid.uuid4())
        
        logger.info(f"Starting workflow for ticket {ticket.ticket_id} - thread_id: {thread_id}")
        
        # Initialize state with the ticket
        initial_state = {
            "ticket": ticket,
            "company_specific": False,
            "retrieved_context": None,
            "context_summary": None,
            "suggested_response": None,
            "identified_actions": None,
            "processing_time": 0.0,
            "agent_response": None
        }
        
        # Run the workflow
        result = await self.graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})

        logger.info(f"Completed workflow for ticket {ticket.ticket_id} - thread_id: {thread_id}")

        # Return the agent response
        return result["agent_response"]
    
    async def process_ticket_stream(self, ticket: Ticket) -> AsyncGenerator[Dict[str, Any], None]:
        """Process a ticket and stream the results as they're generated."""
        thread_id = str(uuid.uuid4())
        
        logger.info(f"Starting streaming workflow for ticket {ticket.ticket_id} - thread_id: {thread_id}")
        
        # Initialize state with the ticket
        initial_state = {
            "ticket": ticket,
            "company_specific": False,
            "retrieved_context": None,
            "context_summary": None,
            "suggested_response": None,
            "identified_actions": None, 
            "processing_time": 0.0,
            "agent_response": None
        }
        
        # Stream through the workflow using astream
        async for chunk in self.graph.astream(
            initial_state, 
            config={"configurable": {"thread_id": thread_id}}
        ):
            for node, state in chunk.items():
                
                # Yield information based on which node just completed
                if node == "retrieve_context" and state["retrieved_context"]:
                    yield {
                        "node": "retrieve_context",
                        "message": f"Found {len(state['retrieved_context'])} relevant documents",
                        "processing_time": state["processing_time"]
                    }
                
                elif node == "generate_response":
                    yield {
                        "node": "generate_response",
                        "context_summary": state["context_summary"],
                        "suggested_response": state["suggested_response"],
                        "actions": state["identified_actions"],
                        "processing_time": state["processing_time"]
                    }
                
                elif node == "create_agent_response" and state["agent_response"]:
                    # Final response
                    yield {
                        "node": "create_agent_response",
                        "context_summary": state["agent_response"].context_summary,
                        "suggested_response": state["agent_response"].suggested_response,
                        "actions": [a.value for a in state["agent_response"].actions],
                        "complete": True,
                        "processing_time": state["processing_time"]
                    }
                    
        logger.info(f"Completed streaming workflow for ticket {ticket.ticket_id} - thread_id: {thread_id}")