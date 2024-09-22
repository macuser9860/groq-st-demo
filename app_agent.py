import time
import json
from datetime import datetime

import os
from dotenv import load_dotenv
import gspread
import streamlit as st
from google.oauth2 import service_account
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.callbacks import StreamlitCallbackHandler
from openai import OpenAI

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def execute_construction_agent(query):
    """
    Execute the construction-focused agent
    """
    # Define Groq LLM Model
    llm = ChatGroq(temperature=0,
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama-3.1-8b-instant")

    # Web Search Tool
    tools = [TavilySearchResults(max_results=3)]

    # Pull prompt from LangChain Hub and modify for construction focus
    react_prompt = hub.pull("hwchase17/react")
    construction_prompt = react_prompt.partial(
        system_message="You are a helpful assistant for a construction company. Provide accurate and relevant information about construction rates, materials, techniques, and regulations."
    )

    # Construct the ReAct agent with construction focus
    agent = create_react_agent(llm, tools, construction_prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor.invoke({"input": query}, 
                                 {"callbacks": [st_callback]})

def check_text(text):
    """
    Check if the text is appropriate and construction-related
    """
    response = client.moderations.create(input=text)
    return response.results[0].flagged

def is_construction_related(text):
    """Check if the given text is related to construction using GPT-3.5."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "Determine if the given text is related to construction. If yes, return `1`, else return `0`"},
            {"role": "user", "content": text}
        ],
        max_tokens=1,
        temperature=0,
        seed=0,
        logit_bias={"15": 100, "16": 100}
    )

    result = int(response.choices[0].message.content)
    return result == 1

def append_to_sheet(prompt, generated, answer):
    """
    Add to GSheet (commented out for now, uncomment and configure as needed)
    """
    # Uncomment and configure Google Sheets integration as needed
    pass

st.title("Construction AI Assistant")
st.subheader("Get Instant Answers to Your Construction Questions")
st.write("Powered by AI and construction industry expertise.")
query = st.text_input("Construction Query", "how much does it cost to build house in nepal?")
button = st.empty()

if button.button("Get Answer"):
    button.empty()
    with st.spinner("Analyzing your query..."):
        is_inappropriate = check_text(query)
        is_construction = is_construction_related(query)
    
    if is_inappropriate or not is_construction:
        st.warning("Your query was flagged as inappropriate or not related to construction. Please try again with a construction-specific question.", icon="🚫")
        append_to_sheet(query, False, "NIL")
        st.stop()
    
    start_time = time.time()
    st_callback = StreamlitCallbackHandler(st.container())
    try:
        results = execute_construction_agent(query)
    except ValueError:
        st.error("An error occurred while processing your request. Please try again.")
        st.stop()
    
    execution_time = round(time.time() - start_time, 2)
    st.success(f"Answer generated in {execution_time} seconds.")
    
    st.markdown(f"""### Question: {results['input']}

**Answer:** {results['output']}""")
    
    # Add construction-specific tips or resources
    st.info("Remember to always consult with licensed professionals and adhere to local building codes and regulations.")
    
    # Provide options for further information
    if st.button("Need more details?"):
        st.write("Here are some additional resources you might find helpful:")
    
    append_to_sheet(results['input'], True, results['output'])
    st.info("Refresh the page to ask another construction-related question.")
