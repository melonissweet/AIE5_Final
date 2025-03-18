from typing import List, Dict, Any
import logging
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import PromptTemplate
from config.settings import (
    OPENAI_API_KEY, 
    LLM_MODEL, 
    TEMPERATURE,
    MAX_TOKENS
)
import hashlib
from langchain.callbacks import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm_client")

class LLMClient:
    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=LLM_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            cache=True # Explicitly enable caching
        )
        
        logger.info("LLM client initialized") 

    def _generate_cache_key(self, prefix: str, text: str) -> str:
        """Generate a safe cache key from text."""
        # Create a hash of the text to ensure it's a valid key
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:10]
        return f"{prefix}:{text_hash}"

    async def generate_context_summary(
        self, 
        ticket_content: str, 
        retrieved_contexts: List[Dict[str, Any]]
    ) -> str:
        """Generate a summary of the relevant contexts for the ticket."""
        logger.info("Start context summary")
        
        context_text = "\n\n".join([
            f"Source: {ctx['source']}\nContent: {ctx['content']}"
            for ctx in retrieved_contexts
        ])
        cache_key = self._generate_cache_key("summary", f"{ticket_content}::{context_text}")

        logger.info(
            f"Generating context summary [{cache_key}] - content_length: {len(ticket_content)}, contexts_count: {len(retrieved_contexts)}"
        )

        prompt = PromptTemplate.from_template(
            """[TASK: GENERATE_CONTEXT_SUMMARY]
            You are an expert customer support assistant helping to analyze retrieved context for a support ticket.
            
            Given the following ticket content and retrieved context from company knowledge base, provide a detailed and comprehensive summary of the most relevant information from the retrieved context that relates to the customer's issue.
            The summary should only be based on the provided ticket content and retrieved context.

            Ticket Content: 
            {ticket}
            
            Retrieved Context:
            {context}
            
            Summary:"""
        )
        
        chain = prompt | self.llm | StrOutputParser()
        with get_openai_callback() as cb:
            result = await chain.ainvoke({
                "ticket": ticket_content,
                "context": context_text
            })
        logger.info(f"Summary request tokens: {cb.total_tokens}")
        return result
    
    async def generate_suggested_response(
        self, 
        ticket_content: str, 
        context_summary: str
    ) -> str:
        """Generate a suggested response for the ticket based on context summary."""
        cache_key = self._generate_cache_key("suggested_response", f"{ticket_content}::{context_summary}")
        logger.info(
            f"Generating suggested response [{cache_key}] - content_length: {len(ticket_content)}, summary_length: {len(context_summary)}"
        )
        
        prompt = PromptTemplate.from_template(
            """[TASK: GENERATE_SUGGESTED_RESPONSE_AND_IDENTIFY_REQUIRED_ACTIONS]
            You are an expert customer support agent. Your goal is to generate email response to customer support ticket and identify action(s) need to be taken by the human customer support agent. Complete the following 2 tasks.

            Tasks:
            1. SUGGESTED RESPONSE: Create a professional, empathetic, and concise email response based on the context. Follow the Reponse Instructions and Language guidance to generate response.
            2. REQUIRED ACTIONS: Identify which actions should be taken for this ticket from the provided List of Action Types.
            
            Return your response in the following JSON format:
            {{
            "suggested_response": "Your suggested email response here",
            "required_actions": ["action1", "action2"]
            }}

            List of Action Types:
            - News Team: The ticket should be forwarded to News Team (journalistic/content issues)
            - Digital Department: The ticket should be forwarded to Digital Department (technical issues)
            - Communication Team: The ticket should be forwarded to Communication Team (media/communication issue)
            - escalate: The ticket should be escalated to a supervisor or manager
            - follow_up: The ticket requires a follow-up after the initial response
            - no_action: No additional actions are required

            Response Instructions:
            1. It is most important to able to answer and address the question or comments from customers in the ticket content. You will only use the context summary to help answer and address the question or comments from the ticket. Carefully review the ticket content and review the retrieved context to use it as knowledge base to help respond to the ticket.
            2. Maintain a professional and courteous tone throughout the response.
            3. The response should be only based on the provided context summary to ensure your response is accurate and aligned with company's knowledge base.
            4. Keep the email response short, brief, concise, and easy for the customer to understand. Avoid using bullet points or numbered list as response unless really needed.
            5. If applicable, provide clear and actionable steps for the customer to resolve their issue.
            6. Follow the given language guidance for generating response
            7. Write the response as the content of an email reply. Write in complete sentence. Do not include a subject line, greeting, or closing. Only provide the body of the email.

            
            Language guidance:
            Quality and Precision
            The company is a language model for its audiences. Good usage and accuracy are essential to high quality journalism. Our language should be simple, clear and concrete.
            Journalistic style is accurate, concise and accessible. Our purpose is to make complex subjects understandable. When specialized or technical vocabulary needs to be used, it is explained and put in a context that makes it easy to understand.
            The description of facts, however concise, must provide the nuances necessary to ensure that the account is faithful and easy to understand.
            Clarity is also essential when numbers and statistics are involved. It is essential to avoid confusion and to take care to properly grasp the numbers used.
            The use of certain highly charged words can undermine credibility and merits special consideration. Language is constantly evolving. We will be attentive to shifts in the meaning of words. We consult language resources and editorial management as needed to grasp the impact of expressions that are open to multiple interpretations and capable of offending some audience members.
            Language Level and Good Taste
            We use the language of accessible, articulate everyday speech.
            We respect and reflect the generally accepted values of society. We are aware that the audiences we address do not all have the same definition of good taste. We choose a tone that will not gratuitously offend audience sensitivities. In particular we avoid swearing and coarse, vulgar, offensive or violent language except where its omission would alter the nature and meaning of the information reported.
            Respect and Absence of Prejudice
            Our vocabulary choices are consistent with equal rights.
            Our language reflects equality of the sexes and we prefer inclusive forms where they are not prohibitively cumbersome.
            We are aware of our influence on how minorities or vulnerable groups are perceived. We do not mention national or ethnic origin, colour, religious affiliation, physical characteristics or disabilities, mental illness, sexual orientation or age except when important to an understanding of the subject or when a person is the object of a search and such personal characteristics will facilitate identification.
            We avoid generalizations, stereotypes and any degrading or offensive words or images that could feed prejudice or expose people to hatred or contempt. Criminal matters require special care and precision.
            When a minority group is referred to, the vocabulary is chosen with care and with consideration for changes in the language.
            </Language guidance>

            Ticket Content: 
            {ticket}
            
            Context Summary: 
            {context_summary}
            
            Suggested Response:"""
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        with get_openai_callback() as cb:
            result = await chain.ainvoke({
                "ticket": ticket_content,
                "context_summary": context_summary
            })
        logger.info(f"Response request tokens: {cb.total_tokens}")
        return result