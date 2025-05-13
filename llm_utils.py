from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

llm = None

def get_llm_instance():
    """
    Returns a singleton instance of the Gemini LLM
    """
    global llm
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            stream=True,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    return llm

def get_response(user_query, conversation_history):
    """
    Uses the LLM to get a response based on user query and conversation history.
    """
    prompt_template = """
    You are an AI assistant. Answer the following question considering the history of the conversation:
    
    Chat history: {conversation_history}
    
    User question: {user_query}
    """
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm = get_llm_instance()
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "conversation_history": conversation_history,
        "user_query": user_query
    })
