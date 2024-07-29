from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

st.header("EEHC Chatbot")

# Initialize session state for chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Initialize session state for prompt input if it doesn't exist
if "prompt_input" not in st.session_state:
    st.session_state["prompt_input"] = ""

# Input prompt for user query
prompt = st.text_input("Prompt", placeholder="Enter Your Question", key="input")

# Function to handle the response and update chat history
def add_to_chat_history(user_input, bot_response):
    st.session_state["chat_history"].append({"role": "user", "content": user_input})
    st.session_state["chat_history"].append({"role": "bot", "content": bot_response})

# Process the user input when the Send button is clicked
if st.button("Send"):
    if prompt:
        # Run the LLM with the user's query
        response = run_llm(query=prompt)
        # Save the query and response in the chat history
        add_to_chat_history(prompt, response['result'])
        # Clear the input box by resetting the session state
        st.session_state["prompt_input"] = ""

# Display the chat history
if "chat_history" in st.session_state:
    for chat in st.session_state["chat_history"]:
        if chat["role"] == "user":
            message(chat["content"], is_user=True)
        else:
            message(chat["content"])
else:
    st.write("No chat history available.")
