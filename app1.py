import os
from dotenv import load_dotenv
import streamlit as st
from langchain.adapters.openai import convert_openai_messages
from langchain_community.chat_models import ChatOpenAI
from tavily import TavilyClient
from requests.exceptions import HTTPError
from groq import Groq

# Load environment variables from the .env file
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="Construction AI Research",
    page_icon="🤖",
    layout="centered"
)

# Get the Tavily API key from the environment
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure the API keys are set, or raise an error
if not TAVILY_API_KEY or not OPENAI_API_KEY:
    st.error("API keys not found. Please set TAVILY_API_KEY and OPENAI_API_KEY in the .env file.")
    st.stop()

# Initialize the Tavily client
client = TavilyClient(api_key=TAVILY_API_KEY)

# Initialize chat history in Streamlit session state if not present already
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Streamlit page title
st.title("🤖 Construction AI Research Assistant")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Input field for user's message
user_prompt = st.chat_input("Ask your research question...") 

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    with st.spinner("Searching for information..."):
        # Step 2. Executing the search query and getting the results
        try:
            # content = client.search(user_prompt, search_depth="advanced")["results"]
            content = client.get_search_context(query=user_prompt)
            # client = Groq()


            # Step 3. Setting up the OpenAI prompts
            prompt = [{
                "role": "system",
                "content": 'You are an AI critical thinker research assistant on niche residential construction in nepal. Indian Standard Codes is in practice in nepal. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text.'
            }, {
                "role": "user",
                "content": f'Information: """"""\n\nUsing the above information, answer the following query: "{user_prompt}" in a detailed report -- Please use markdown syntax.'
            }]


            # messages = [{
            #     "role": "system",
            #     "content": 'You are an AI critical thinker research assistant on niche residential construction in nepal. Indian Standard Codes is in practice in nepal. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text.'
            # }, {
            #     "role": "user",
            #     "content": f'Information: """"""\n\nUsing the above information, answer the following query: "{user_prompt}" in a detailed report -- Please use markdown syntax.'
            # }, *st.session_state.chat_history
            # ]

            # response = client.chat.completions.create(
            #     model="llama3-groq-70b-8192-tool-use-preview",
            #     messages=messages
            # )

            # report = response.choices[0].message.content

            # Step 4. Running OpenAI through Langchain
            lc_messages = convert_openai_messages(prompt)
            report = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY).invoke(lc_messages).content

            st.session_state.chat_history.append({"role": "assistant", "content": report})

            # Display the LLM's response
            with st.chat_message("assistant"):
                st.markdown(report)

            # Optional links
            st.markdown('[Call Us](tel:9860115463)')
            st.markdown(
                '<a href="https://housedesigninnepal.com" target="_blank"><button>Need more details?</button></a>',
                unsafe_allow_html=True
            )

        except HTTPError as e:
            st.error(f"Error: {e.response.text}")
            content = []

    
