import asyncio
import time
import uuid
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
from schemas.models import Ticket, AgentResponse, HumanFeedback, ActionType
from agent.agent import CustomerSupportAgent
from config.settings import API_HOST, API_PORT
from utils.redis_utils import redis_client_instance, RateLimitExceeded
import json
from agent.workflows import CustomerSupportWorkflow
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Initialize FastAPI app
app = FastAPI(
    title="LLM Customer Support Assistant",
    description="An LLM-powered agent to assist customer support representatives",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the customer support agent
agent = CustomerSupportAgent()

# In-memory storage for demo purposes (would be a database in production)
tickets_store = {}
responses_store = {}
feedback_store = {}

# Redis initialization on startup
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup."""
    logger.info("Initializing Redis connection...")
    redis_connected = await redis_client_instance.initialize()

    if redis_connected:
        try:
            set_llm_cache(InMemoryCache())
            logger.info("Semantic cache initialization: Success")
        except Exception as e:
            logger.warning("Semantic cache NOT initialized")
    else:
        logger.warning("Redis connection failed - semantic cache will not be available")
    logger.info("Application started", extra={"env": os.getenv("ENVIRONMENT", "development")})

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown."""
    await redis_client_instance.close()
    logger.info("Application shut down")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Generate a request ID
    request_id = str(uuid.uuid4())

    logger.info(
        f"Request started: {request.method} {request.url.path} - request_id: {request_id}, method: {request.method}, path: {request.url.path}, query_params: {str(request.query_params)}, client_host: {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        logger.info(
            f"Request completed: {request.method} {request.url.path} - request_id: {request_id}, method: {request.method}, path: {request.url.path}, status_code: {response.status_code}, process_time: {process_time}"
        )
        
        return response
    except Exception as e:
        logger.error(
            f"Request failed: {request.method} {request.url.path} - request_id: {request_id}, method: {request.method}, path: {request.url.path}, error: {str(e)}"
        )
        raise

async def rate_limit_dependency(request: Request):
    """Global rate limiting dependency."""
    client_host = request.client.host if request.client else "unknown"
    rate_key = f"global_ratelimit:{client_host}"
    
    # Allow 60 requests per minute per client IP
    current = await redis_client_instance.increment_counter(rate_key, 60)
    
    if current > 60:
        logger.warning(
            f"Global rate limit exceeded - client_host: {client_host}, count: {current}"
        )
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

@app.get("/", dependencies=[Depends(rate_limit_dependency)])
async def root():
    return {"message": "LLM Customer Support Assistant API is running"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

async def stream_agent_response(ticket: Ticket):
    """Stream the agent response in chunks as it's being generated."""
    # Start with initial response structure
    initial_response = {
        "ticket_id": ticket.ticket_id,
        "streaming": True,
        "message": "Starting ticket processing...",
        "processing_time": 0.0
    }
    
    # Send initial response
    yield f"data: {json.dumps(initial_response)}\n\n"
    
    start_time = time.time()
    logger.info(f"Start streaming agent response for ticket: {ticket.ticket_id}")
    
    try:
        # Create workflow instance to handle streaming
        workflow = CustomerSupportWorkflow()
        
        # Process ticket stream
        async for update in workflow.process_ticket_stream(ticket):
            # Include ticket_id and streaming flag
            update["ticket_id"] = ticket.ticket_id
            update["streaming"] = True
            
            # Add processing time if not included
            if "processing_time" not in update:
                update["processing_time"] = time.time() - start_time
                
            # Yield the update as a server-sent event
            yield f"data: {json.dumps(update)}\n\n"
            
            # If this is the final update, store it for future reference
            if update.get("complete", False):
                actions = []
                
                # Process action strings from the update
                action_list = update.get("actions", [])
                if action_list:
                    for action_str in action_list:
                        try:
                            # Handle both string and enum cases
                            if isinstance(action_str, str):
                                action = ActionType(action_str.strip())
                                actions.append(action)
                            else:
                                actions.append(action_str)
                        except ValueError:
                            if action_str and isinstance(action_str, str) and action_str.strip().lower() != "no_action":
                                logger.warning(f"Invalid action type '{action_str}' for ticket {ticket.ticket_id}")
                            
                            # Only add NO_ACTION if no other actions
                            if not actions:
                                actions.append(ActionType.NO_ACTION)
                else:
                    # Default to NO_ACTION if no actions were provided
                    actions.append(ActionType.NO_ACTION)
                
                # Create and store a proper AgentResponse object
                agent_response = AgentResponse(
                    ticket_id=ticket.ticket_id,
                    context_summary=update.get("context_summary", ""),
                    suggested_response=update.get("suggested_response", ""),
                    actions=actions,
                    retrieved_contexts=[],  # Simplified for streaming
                    processing_time=update.get("processing_time", 0.0)
                )
                
                # Store in the responses_store
                responses_store[ticket.ticket_id] = agent_response
                
                # Store the processed ticket and agent response in the vector database
                await agent._store_processed_ticket(ticket, agent_response)
        
    except Exception as e:
        logger.error(f"Error streaming response for ticket {ticket.ticket_id}: {str(e)}")
        error_response = {
            "ticket_id": ticket.ticket_id,
            "error": str(e),
            "complete": True,
            "streaming": True,
            "processing_time": time.time() - start_time
        }
        yield f"data: {json.dumps(error_response)}\n\n"

@app.post("/ticket/stream", dependencies=[Depends(rate_limit_dependency)])
async def process_ticket_stream(ticket: Ticket):
    """
    Process a customer support ticket and stream the response as it's generated.
    """
    # Generate ticket ID if not provided
    if not ticket.ticket_id:
        ticket.ticket_id = str(uuid.uuid4())
    
    logger.info(
        f"Processing new ticket with streaming: {ticket.ticket_id} - ticket_id: {ticket.ticket_id}, customer_id: {ticket.customer_id}, subject: {ticket.subject}"
    )
    
    # Store the ticket
    tickets_store[ticket.ticket_id] = ticket
    
    # Return a streaming response
    return StreamingResponse(
        stream_agent_response(ticket),
        media_type="text/event-stream"
    )

@app.post("/ticket", response_model=AgentResponse, dependencies=[Depends(rate_limit_dependency)])
async def process_ticket(ticket: Ticket, background_tasks: BackgroundTasks):
    """
    Process a customer support ticket and generate a suggested response.
    Legacy endpoint for non-streaming responses.
    """
    # Generate ticket ID if not provided
    if not ticket.ticket_id:
        ticket.ticket_id = str(uuid.uuid4())
    
    logger.info(
        f"Processing new ticket: {ticket.ticket_id} - ticket_id: {ticket.ticket_id}, customer_id: {ticket.customer_id}, subject: {ticket.subject}"
    )
    
    # Store the ticket
    tickets_store[ticket.ticket_id] = ticket
    
    # Process the ticket
    try:
        # Process ticket asynchronously
        agent_response = await agent.process_ticket(ticket)
        
        # Store the response
        responses_store[ticket.ticket_id] = agent_response
        
        logger.info(
            f"Completed processing ticket: {ticket.ticket_id} - ticket_id: {ticket.ticket_id}, processing_time: {agent_response.processing_time}"
        )
        
        return agent_response
    except RateLimitExceeded as e:
        logger.warning(
            f"Rate limit exceeded for ticket: {ticket.ticket_id} - ticket_id: {ticket.ticket_id}, error: {str(e)}"
        )
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(
            f"Error processing ticket: {ticket.ticket_id} - ticket_id: {ticket.ticket_id}, error: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"Error processing ticket: {str(e)}")
    
@app.post("/feedback/{ticket_id}", response_model=Ticket, dependencies=[Depends(rate_limit_dependency)])
# @traceable(name="submit_feedback_endpoint")
async def submit_feedback(ticket_id: str, feedback: HumanFeedback):
    """
    Submit human feedback for an agent-processed ticket.
    """
    # Check if ticket exists
    if ticket_id not in tickets_store:
        logger.warning(
            f"Ticket not found: {ticket_id} - ticket_id: {ticket_id}"
        )
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    # Check if agent response exists
    if ticket_id not in responses_store:
        logger.warning(
            f"Agent response not found: {ticket_id}  - ticket_id: {ticket_id}"
        )
        raise HTTPException(status_code=404, detail="Agent response not found")
    
    logger.info(
        f"Processing feedback for ticket: {ticket_id} - ticket_id: {ticket_id}, approved: {feedback.approved}"
    )
    
    ticket = tickets_store[ticket_id]
    agent_response = responses_store[ticket_id]
    
    try:
        # Process the human feedback
        updated_ticket = await agent.process_human_feedback(
            ticket=ticket,
            agent_response=agent_response,
            feedback=feedback
        )
        
        # Update the ticket in the store
        tickets_store[ticket_id] = updated_ticket
        
        # Store the feedback
        feedback_store[ticket_id] = feedback
        
        logger.info(
            f"Completed processing feedback for ticket: {ticket_id} - ticket_id: {ticket_id}, updated_tags: {updated_ticket.tags}"
        )
        
        return updated_ticket
    except RateLimitExceeded as e:
        logger.warning(
            f"Rate limit exceeded for feedback: {ticket_id} - ticket_id: {ticket_id}, error: {str(e)}"
        )
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(
            f"Error processing feedback: {ticket_id} - ticket_id: {ticket_id}, error: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@app.get("/ticket/{ticket_id}", response_model=Dict[str, Any], dependencies=[Depends(rate_limit_dependency)])
# @traceable(name="get_ticket_details_endpoint")
async def get_ticket_details(ticket_id: str):
    """
    Get details for a specific ticket, including agent response and feedback if available.
    """
    # Check if ticket exists
    if ticket_id not in tickets_store:
        logger.warning(
            f"Ticket not found: {ticket_id} - ticket_id: {ticket_id}"
        )
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    logger.info(
        f"Retrieved ticket details: {ticket_id} - ticket_id: {ticket_id}"
    )
    
    result = {
        "ticket": tickets_store[ticket_id],
        "agent_response": responses_store.get(ticket_id),
        "feedback": feedback_store.get(ticket_id)
    }
    
    return result

@app.get("/tickets", response_model=List[Ticket], dependencies=[Depends(rate_limit_dependency)])
# @traceable(name="list_tickets_endpoint")
async def list_tickets():
    """
    List all processed tickets.
    """
    logger.info(
        f"Listed all tickets - ticket_count: {len(tickets_store)}"
    )
    
    return list(tickets_store.values())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=True)