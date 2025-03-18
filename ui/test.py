import streamlit as st
import requests
import json
import time
from datetime import datetime
import uuid

# API endpoint
API_URL = "http://0.0.0.0:8000"

# Page configuration
st.set_page_config(
    page_title="Customer Support Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for conversation threading
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_ticket_id" not in st.session_state:
    st.session_state.current_ticket_id = None
if "tickets" not in st.session_state:
    st.session_state.tickets = []
if "error" not in st.session_state:
    st.session_state.error = None

# Function to create a new conversation
def create_new_conversation():
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.current_ticket_id = None
    if "last_response" in st.session_state:
        del st.session_state.last_response
    if "last_ticket" in st.session_state:
        del st.session_state.last_ticket

# Function to create a new ticket with streaming support
def create_ticket(customer_id, subject, content, tags):
    # Generate a unique ticket_id for this conversation
    ticket_id = str(uuid.uuid4())
    
    ticket_data = {
        "ticket_id": ticket_id,
        "customer_id": customer_id,
        "subject": subject,
        "content": content,
        "created_at": datetime.now().isoformat(),
        "tags": tags,
        "priority": "medium",
        "status": "new",
        "metadata": {
            "conversation_id": st.session_state.conversation_id,
            "session_id": st.session_state.conversation_id
        }
    }
    
    try:
        # Create a placeholder for streaming updates
        response_placeholder = st.empty()
        context_placeholder = st.empty()
        action_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        # Add to conversation history
        if st.session_state.conversation_id not in st.session_state.conversations:
            st.session_state.conversations[st.session_state.conversation_id] = []
        
        # Add user message to conversation
        st.session_state.conversations[st.session_state.conversation_id].append({
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Store the ticket ID in the session state
        st.session_state.current_ticket_id = ticket_id
        
        # Use streaming endpoint for real-time updates
        with requests.post(f"{API_URL}/ticket/stream", json=ticket_data, stream=True, timeout=300) as response:
            if response.status_code != 200:
                st.session_state.error = f"Error creating ticket: {response.text}"
                return None, ticket_data
            
            # Initialize response data
            context_summary = ""
            suggested_response = ""
            actions = []
            
            # Initialize progress
            progress = 0
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    # Parse SSE format
                    line_text = line.decode('utf-8')
                    if line_text.startswith('data: '):
                        try:
                            data = json.loads(line_text[6:])
                            
                            # Update progress
                            progress = min(0.9, progress + 0.1)  # Gradually increase up to 90%
                            
                            # Update context summary if available
                            if "context_summary" in data and data["context_summary"]:
                                context_summary = data["context_summary"]
                                with context_placeholder.container():
                                    st.text("Building context...")
                            
                            # Update suggested response if available
                            if "suggested_response" in data and data["suggested_response"]:
                                suggested_response = data["suggested_response"]
                                with response_placeholder.container():
                                    st.write("Generating response...")
                                    st.write(suggested_response)
                            
                            # Update actions if available
                            if "actions" in data and data["actions"]:
                                actions = data["actions"]
                                with action_placeholder.container():
                                    st.write("Identifying actions...")
                                    for action in actions:
                                        st.write(f"- {action}")
                            
                            # Update progress bar
                            progress_placeholder.progress(progress, text=f"Processing: {int(progress * 100)}%")
                            
                            # If complete, set progress to 100%
                            if data.get("complete", False):
                                progress = 1.0
                                progress_placeholder.progress(progress, text=f"Complete: 100%")
                                
                                # Add assistant response to conversation history
                                st.session_state.conversations[st.session_state.conversation_id].append({
                                    "role": "assistant",
                                    "content": suggested_response,
                                    "timestamp": datetime.now().isoformat()
                                })
                                
                                # Store final response
                                final_response = {
                                    "ticket_id": ticket_id,
                                    "context_summary": context_summary,
                                    "suggested_response": suggested_response,
                                    "actions": actions,
                                    "processing_time": data.get("processing_time", 0.0)
                                }
                                
                                st.session_state.last_ticket = ticket_data
                                st.session_state.last_response = final_response
                                
                                return final_response, ticket_data
                            
                        except json.JSONDecodeError as e:
                            st.warning(f"Error parsing streaming response: {e}")
        
        return None, ticket_data
        
    except requests.exceptions.RequestException as e:
        st.session_state.error = f"Request error: {str(e)}"
        return None, ticket_data
    
# Function to submit feedback
def submit_feedback(ticket_id, agent_response_id, approved, feedback_text, modified_response, modified_actions):
    feedback_data = {
        "ticket_id": ticket_id,
        "agent_response_id": agent_response_id,
        "approved": approved,
        "feedback": feedback_text,
        "modified_response": modified_response,
        "modified_actions": modified_actions
    }
    
    try:
        response = requests.post(f"{API_URL}/feedback/{ticket_id}", json=feedback_data, timeout=60)
        
        if response.status_code == 200:
            # Add to conversation
            if st.session_state.conversation_id in st.session_state.conversations:
                st.session_state.conversations[st.session_state.conversation_id].append({
                    "role": "feedback",
                    "content": feedback_text,
                    "approved": approved,
                    "timestamp": datetime.now().isoformat()
                })
            
            return response.json()
        else:
            st.session_state.error = f"Error submitting feedback: {response.text}"
            return None
    except requests.exceptions.RequestException as e:
        st.session_state.error = f"Request error: {str(e)}"
        return None

# Display errors if they exist
if st.session_state.error:
    st.error(st.session_state.error)
    if st.button("Clear Error"):
        st.session_state.error = None
    st.markdown("---")

# App title
st.title("🤖 Customer Support Assistant")

# Sidebar for navigation and conversation management
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Create Ticket", "View Tickets"])

# New conversation button
if st.sidebar.button("New Conversation"):
    create_new_conversation()
    st.sidebar.success("Started new conversation!")

# Display current conversation ID
st.sidebar.write(f"Current Conversation: {st.session_state.conversation_id[:8]}...")

# Display conversation history
if st.session_state.conversation_id in st.session_state.conversations:
    st.sidebar.write("Conversation History:")
    for idx, message in enumerate(st.session_state.conversations[st.session_state.conversation_id]):
        role = message["role"]
        content_preview = message["content"][:30] + "..." if len(message["content"]) > 30 else message["content"]
        
        if role == "user":
            st.sidebar.info(f"You: {content_preview}")
        elif role == "assistant":
            st.sidebar.success(f"Assistant: {content_preview}")
        elif role == "feedback":
            if message.get("approved", False):
                st.sidebar.success(f"Feedback (Approved): {content_preview}")
            else:
                st.sidebar.warning(f"Feedback (Rejected): {content_preview}")

if page == "Create Ticket":
    st.header("Create New Support Ticket")
    
    # Ticket form
    with st.form(key=f"ticket_form_{st.session_state.conversation_id}"):
        customer_id = st.text_input("Customer ID", value=f"cust_{uuid.uuid4().hex[:6]}")
        subject = st.text_input("Subject")
        content = st.text_area("Ticket Content", height=200)
        
        # Tag options
        st.write("Tags")
        col1, col2, col3 = st.columns(3)
        with col1:
            company_specific = st.checkbox("company-specific")
        with col2:
            urgent = st.checkbox("urgent")
        with col3:
            billing = st.checkbox("billing")
        
        # Compile tags
        tags = []
        if company_specific:
            tags.append("company-specific")
        if urgent:
            tags.append("urgent")
        if billing:
            tags.append("billing")
        
        submitted = st.form_submit_button("Submit Ticket")
        
        if submitted:
            if not subject or not content:
                st.error("Please fill out all required fields.")
            else:
                with st.spinner("Processing ticket..."):
                    agent_response, ticket = create_ticket(customer_id, subject, content, tags)
                    
                    if agent_response:
                        st.session_state.last_ticket = ticket
                        st.session_state.last_response = agent_response
                        
                        # Add to conversation
                        if st.session_state.conversation_id in st.session_state.conversations:
                            st.session_state.conversations[st.session_state.conversation_id].append({
                                "role": "assistant",
                                "content": agent_response["suggested_response"],
                                "timestamp": datetime.now().isoformat()
                            })
                        
                        st.success("Ticket created and processed successfully!")
    
    # Show response if available
    if hasattr(st.session_state, 'last_response') and st.session_state.last_response:
        st.header("Agent Response")
        
        response = st.session_state.last_response
        ticket = st.session_state.last_ticket
        
        # Display ticket info
        st.subheader("Ticket Information")
        st.write(f"**Ticket ID:** {ticket['ticket_id']}")
        st.write(f"**Subject:** {ticket['subject']}")
        st.write(f"**Tags:** {', '.join(ticket['tags'])}")
        
        # Display context summary
        with st.expander("Context Summary", expanded=False):
            st.write(response["context_summary"])
        
        # Display suggested response
        st.subheader("Suggested Response")
        st.write(response["suggested_response"])
        
        # Display actions
        st.subheader("Suggested Actions")
        if response["actions"]:
            for action in response["actions"]:
                st.write(f"- {action}")
        else:
            st.write("No actions suggested.")
        
        # Display confidence
        # st.progress(response["confidence_score"], text=f"Confidence: {response['confidence_score']:.2f}")
        
        # Human feedback form
        st.header("Human Agent Feedback")
        
        with st.form(key=f"feedback_form_{st.session_state.conversation_id}"):
            approved = st.radio("Approve this response?", ["Yes", "No"]) == "Yes"
            
            feedback_text = st.text_area("Feedback (optional)")
            
            modified_response = st.text_area("Modified Response (leave empty to use the suggested response)", 
                                           value=response["suggested_response"] if not approved else "")
            
            # Action modification options
            st.write("Modify Actions")
            action_options = ["no_action", "forward_to_team_a", "forward_to_team_b", "escalate", "follow_up"]
            selected_actions = []
            
            for action in action_options:
                is_selected = action in [a.lower() for a in response["actions"]]
                if st.checkbox(action, value=is_selected, key=f"action_{action}_{st.session_state.conversation_id}"):
                    selected_actions.append(action)
            
            feedback_submitted = st.form_submit_button("Submit Feedback")
            
            if feedback_submitted:
                with st.spinner("Processing feedback..."):
                    result = submit_feedback(
                        ticket_id=ticket["ticket_id"],
                        agent_response_id=response["ticket_id"],
                        approved=approved,
                        feedback_text=feedback_text,
                        modified_response=modified_response if modified_response else None,
                        modified_actions=selected_actions if selected_actions else None
                    )
                    
                    if result:
                        st.success("Feedback submitted successfully!")
                        st.write("Updated ticket tags:", ", ".join(result["tags"]))
                        
                        # Create a new conversation for the next ticket
                        if st.button("Start New Conversation"):
                            create_new_conversation()

elif page == "View Tickets":
    st.header("View Processed Tickets")
    
    if st.button("Refresh Tickets"):
        with st.spinner("Loading tickets..."):
            try:
                response = requests.get(f"{API_URL}/tickets", timeout=30)
                
                if response.status_code == 200:
                    tickets = response.json()
                    st.session_state.tickets = tickets
                else:
                    st.error(f"Error loading tickets: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request error: {str(e)}")
    
    if hasattr(st.session_state, 'tickets') and st.session_state.tickets:
        # Display tickets in a table
        ticket_data = []
        for ticket in st.session_state.tickets:
            conversation_id = ticket.get("metadata", {}).get("conversation_id", "Unknown")
            ticket_data.append({
                "Ticket ID": ticket["ticket_id"],
                "Subject": ticket["subject"],
                "Status": ticket["status"],
                "Tags": ", ".join(ticket["tags"]),
                "Created": ticket["created_at"],
                "Conversation": conversation_id[:8] + "..." if len(conversation_id) > 8 else conversation_id
            })
        
        st.dataframe(ticket_data)
        
        # Ticket details
        st.subheader("Ticket Details")
        ticket_id = st.selectbox("Select a ticket to view details", 
                               options=[t["ticket_id"] for t in st.session_state.tickets])
        
        if ticket_id:
            try:
                response = requests.get(f"{API_URL}/ticket/{ticket_id}", timeout=30)
                
                if response.status_code == 200:
                    ticket_details = response.json()
                    
                    # Create tabs for different sections
                    ticket_tab, response_tab, feedback_tab = st.tabs(["Ticket Info", "Agent Response", "Human Feedback"])
                    
                    with ticket_tab:
                        st.write(f"**Subject:** {ticket_details['ticket']['subject']}")
                        st.write(f"**Content:** {ticket_details['ticket']['content']}")
                        st.write(f"**Tags:** {', '.join(ticket_details['ticket']['tags'])}")
                        st.write(f"**Created:** {ticket_details['ticket']['created_at']}")
                        
                        # Add conversation ID if available
                        conversation_id = ticket_details['ticket'].get("metadata", {}).get("conversation_id", "Unknown")
                        st.write(f"**Conversation ID:** {conversation_id}")
                        
                        # Button to switch to this conversation
                        if conversation_id != "Unknown":
                            if st.button("Continue this conversation"):
                                st.session_state.conversation_id = conversation_id
                                st.session_state.current_ticket_id = ticket_id
                                st.experimental_rerun()
                    
                    # Display agent response if available
                    with response_tab:
                        if ticket_details.get("agent_response"):
                            st.write(f"**Context Summary:** {ticket_details['agent_response']['context_summary']}")
                            st.write(f"**Suggested Response:** {ticket_details['agent_response']['suggested_response']}")
                            st.write(f"**Actions:** {', '.join(ticket_details['agent_response']['actions'])}")
                            # st.write(f"**Confidence Score:** {ticket_details['agent_response']['confidence_score']}")
                            # st.progress(ticket_details['agent_response']['confidence_score'])
                            st.write(f"**Processing Time:** {ticket_details['agent_response']['processing_time']:.2f} seconds")
                        else:
                            st.info("No agent response available for this ticket.")
                    
                    # Display feedback if available
                    with feedback_tab:
                        if ticket_details.get("feedback"):
                            st.write(f"**Approved:** {'Yes' if ticket_details['feedback']['approved'] else 'No'}")
                            
                            if ticket_details['feedback'].get('feedback'):
                                st.write(f"**Feedback:** {ticket_details['feedback']['feedback']}")
                            
                            if ticket_details['feedback'].get('modified_response'):
                                st.write(f"**Modified Response:** {ticket_details['feedback']['modified_response']}")
                            
                            if ticket_details['feedback'].get('modified_actions'):
                                st.write(f"**Modified Actions:** {', '.join(ticket_details['feedback']['modified_actions'])}")
                        else:
                            st.info("No human feedback available for this ticket.")
                else:
                    st.error(f"Error loading ticket details: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request error: {str(e)}")
    else:
        st.info("No tickets found. Click 'Refresh Tickets' to load tickets.")

# Footer
st.markdown("---")
st.caption("LLM Customer Support Assistant | Powered by LangChain & LangGraph")
