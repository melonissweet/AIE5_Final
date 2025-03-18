from typing import Dict, List, Any
from schemas.models import ActionType, Ticket, AgentResponse
import time
import logging
from utils.redis_utils import rate_limit
# from langsmith import traceable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent_actions")

class ActionHandler:
    """Handles different actions that can be taken on tickets."""
    
    # @traceable(name="process_action")
    @rate_limit(limit=100, period=60)  # Allow 100 actions per minute
    async def process_actions(
        self, 
        ticket: Ticket,
        agent_response: AgentResponse
    ) -> Dict[str, Any]:
        """Process the actions identified by the agent."""
        start_time = time.time()
        action_results = {}
        
        logger.info(f"Processing actions for ticket {ticket.ticket_id} - actions: {[a.value for a in agent_response.actions]}")
        
        for action in agent_response.actions:
            if action == ActionType.NO_ACTION:
                continue
                
            # Call the appropriate action handler based on action type
            if action == ActionType.FORWARD_TO_TEAM_A:
                result = await self.forward_to_team_a(ticket)
                action_results[action] = result
                
            elif action == ActionType.FORWARD_TO_TEAM_B:
                result = await self.forward_to_team_b(ticket)
                action_results[action] = result
                
            elif action == ActionType.ESCALATE:
                result = await self.escalate_ticket(ticket)
                action_results[action] = result
                
            elif action == ActionType.FOLLOW_UP:
                result = await self.schedule_follow_up(ticket)
                action_results[action] = result
        
        processing_time = time.time() - start_time
        logger.info(f"Completed processing actions for ticket {ticket.ticket_id} - processing_time: {processing_time}, actions_count: {len(action_results)}")
        
        return action_results
    
    # @traceable(name="apply_approved_actions")
    @rate_limit(limit=200, period=60)
    async def apply_approved_actions(
        self, 
        ticket: Ticket,
        approved_actions: List[ActionType]
    ) -> Ticket:
        """Apply approved actions to the ticket by updating tags."""
        start_time = time.time()
        updated_ticket = ticket.model_copy()
        
        logger.info(f"Applying approved actions for ticket {ticket.ticket_id} - actions: {[a.value for a in approved_actions]}")
        
        # Add action tags based on approved actions
        for action in approved_actions:
            if action == ActionType.NO_ACTION:
                continue
                
            action_tag = f"action:{action.value}"
            if action_tag not in updated_ticket.tags:
                updated_ticket.tags.append(action_tag)
        
        # Add agent-assisted tag
        if "agent-assisted" not in updated_ticket.tags:
            updated_ticket.tags.append("agent-assisted")
        
        processing_time = time.time() - start_time
        logger.info(f"Completed applying approved actions for ticket {ticket.ticket_id} - processing_time: {processing_time}, updated_tags: {updated_ticket.tags}")
        
        return updated_ticket
    
    # @traceable(name="forward_to_team_a")
    async def forward_to_team_a(self, ticket: Ticket) -> Dict[str, Any]:
        """Forward the ticket to Team A (technical team)."""
        logger.info(f"Forwarding ticket {ticket.ticket_id} to Team A")
        
        # In a real implementation, this would integrate with ticket system API
        # For now, we just return a success message
        return {
            "status": "success",
            "message": f"Ticket {ticket.ticket_id} has been forwarded to Team A",
            "timestamp": time.time()
        }
    
    # @traceable(name="forward_to_team_b")
    async def forward_to_team_b(self, ticket: Ticket) -> Dict[str, Any]:
        """Forward the ticket to Team B (billing team)."""
        logger.info(f"Forwarding ticket {ticket.ticket_id} to Team B")
        
        # In a real implementation, this would integrate with ticket system API
        return {
            "status": "success",
            "message": f"Ticket {ticket.ticket_id} has been forwarded to Team B",
            "timestamp": time.time()
        }
    
    # @traceable(name="escalate_ticket")
    async def escalate_ticket(self, ticket: Ticket) -> Dict[str, Any]:
        """Escalate the ticket to a supervisor or manager."""
        logger.info(f"Escalating ticket {ticket.ticket_id}")
        
        # In a real implementation, this would integrate with ticket system API
        return {
            "status": "success",
            "message": f"Ticket {ticket.ticket_id} has been escalated to management",
            "timestamp": time.time()
        }
    
    # @traceable(name="schedule_follow_up")
    async def schedule_follow_up(self, ticket: Ticket) -> Dict[str, Any]:
        """Schedule a follow-up for the ticket."""
        logger.info(f"Scheduling follow-up for ticket {ticket.ticket_id}")
        
        # In a real implementation, this would create a calendar reminder or task
        return {
            "status": "success",
            "message": f"Follow-up scheduled for ticket {ticket.ticket_id}",
            "timestamp": time.time(),
            "follow_up_date": "24 hours from now"  # Placeholder
        }