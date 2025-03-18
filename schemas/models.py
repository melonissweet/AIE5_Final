from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TicketStatus(str, Enum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"

class ActionType(str, Enum):
    NEWS_TEAM = "forward_to_news_team"
    DIGITAL_DEPARTMENT = "forward_to_digital_department"
    COMMUNICATION_TEAM="forward_to_communication_team"
    ESCALATE = "escalate"
    FOLLOW_UP = "follow_up"
    NO_ACTION = "no_action"

class Ticket(BaseModel):
    ticket_id: str
    customer_id: str
    subject: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    priority: TicketPriority = TicketPriority.MEDIUM
    status: TicketStatus = TicketStatus.NEW
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class AgentResponse(BaseModel):
    ticket_id: str
    context_summary: str
    suggested_response: str
    actions: List[ActionType] = [ActionType.NO_ACTION]
    retrieved_contexts: List[Dict[str, Any]] = []
    processing_time: float

class HumanFeedback(BaseModel):
    ticket_id: str
    agent_response_id: str
    approved: bool
    feedback: Optional[str] = None
    modified_response: Optional[str] = None
    modified_actions: Optional[List[ActionType]] = None
