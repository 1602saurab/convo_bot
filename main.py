import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from llm_utils import get_response

st.set_page_config(page_title="Conversational Bot")
st.title("Conversational Chatbot 💬")

# Initialize session state to store chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello, I am a bot. How can I help you?")]

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message("assistant" if isinstance(message, AIMessage) else "user"):
        st.write(message.content)

# Accept user input
prompt = st.chat_input("Say Something")

if prompt:
    # Add user's message to the session
    user_message = HumanMessage(content=prompt)
    st.session_state.messages.append(user_message)
    
    with st.chat_message("user"):
        st.write(prompt)

    # Generate response using the LLM chain
    with st.chat_message("assistant"):
        response = st.write_stream(get_response(prompt, st.session_state.messages))
    
    # Save bot's response to session
    st.session_state.messages.append(AIMessage(content=response))
