import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq

# Load environment variables from the .env file
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Construction AI",
    page_icon="🦙",
    layout="centered"
)

# Get the GROQ API key from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ensure the API key is set, or raise an error
if GROQ_API_KEY is None:
    st.error("GROQ API key not found. Please set it in the .env file.")
    st.stop()

# Save the API key to the environment variable (optional if needed for compatibility)
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()

# Initialize chat history in Streamlit session state if not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit page title
st.title("🏗️ Construction AI")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask construction AI...")

if user_prompt:

    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Send user's message to the LLM and get a response
    messages = [
        {"role": "system", "content": "You are a helpful assistant for a construction company. Provide accurate and relevant information about construction rates, materials, techniques, and regulations. Follow Indian Standard Codes for suggestions. Say 'Please ask me construction-related questions' if they ask irrelevant questions."},
        *st.session_state.chat_history
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display the LLM's response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # Display the 'Need more details?' button as an HTML link
    st.markdown(
        '<a href="https://housedesigninnepal.com" target="_blank"><button>Need more details?</button></a>',
        unsafe_allow_html=True
    )
