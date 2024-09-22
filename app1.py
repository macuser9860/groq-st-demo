import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.callbacks import StreamlitCallbackHandler

# Load environment variables
load_dotenv()

# Get the GROQ and Tavily API keys from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Streamlit page configuration
st.set_page_config(
    page_title="Construction AI",
    page_icon="🏗️",
    layout="centered"
)

# Ensure the API keys are set, or raise an error
if GROQ_API_KEY is None:
    st.error("GROQ API key not found. Please set it in the .env file.")
    st.stop()

if TAVILY_API_KEY is None:
    st.error("Tavily API key not found. Please set it in the .env file.")
    st.stop()

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize chat history in Streamlit session state if not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Streamlit page title
st.title("🏗️ Construction AI")

# Function to execute the construction-focused agent
def execute_construction_agent(query):
    """
    Execute the construction-focused agent using Groq and Tavily tools.
    """
    # Define Groq LLM Model
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="mixtral-8x7b-32768"
    )

    # Web Search Tool with Tavily
    tools = [TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)]

    # Pull prompt from LangChain Hub and modify it for a construction focus
    react_prompt = hub.pull("hwchase17/react")
    construction_prompt = react_prompt.partial(
        system_message="You are a helpful assistant for a construction company. Provide accurate and relevant information about construction rates, materials, techniques, and regulations."
    )

    # Construct the ReAct agent with construction focus
    agent = create_react_agent(llm, tools, construction_prompt)

    # Create an agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Streamlit callback handler for interactive feedback
    st_callback = StreamlitCallbackHandler(st.container())
    
    return agent_executor.invoke({"input": query}, {"callbacks": [st_callback]})

# Function to check if text is construction-related
def is_construction_related(text):
    """Check if the given text is related to construction using OpenAI."""
    response = client.moderations.create(input=text)
    return response.results[0].flagged

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask construction AI...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Execute the construction agent
    with st.spinner("Generating response..."):
        try:
            results = execute_construction_agent(user_prompt)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

    # Display the LLM's response
    assistant_response = results['output']
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        st.markdown(assistant_response)

    # Display the 'Need more details?' button as an HTML link
    st.markdown(
        '<a href="https://housedesigninnepal.com" target="_blank"><button>Need more details?</button></a>',
        unsafe_allow_html=True
    )
