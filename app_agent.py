import time
import os
from dotenv import load_dotenv
import streamlit as st
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.callbacks import StreamlitCallbackHandler
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def execute_search_agent(query):
    """
    Execute the search agent with a focus on construction-related queries
    """
    llm = ChatGroq(temperature=0,
                   groq_api_key=os.getenv("GROQ_API_KEY"),
                   model_name="llama3-8b-8192")
    
    tools = [TavilySearchResults(max_results=3)]
    
    construction_prompt = hub.pull("hwchase17/react")
    construction_prompt = construction_prompt.partial(
        system_message="""You are an AI assistant specialized in construction and architecture. 
        Focus on providing accurate and relevant information about building costs, materials, techniques, and regulations.
        When using tools, make sure to use the exact tool name as provided. For example, use 'tavily_search_results_json' instead of 'Search the tavily_search_results_json'."""
    )
    
    agent = create_react_agent(llm, tools, construction_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor.invoke({"input": query}, 
                                 {"callbacks": [StreamlitCallbackHandler(st.container())]})

def check_text(text):
    """
    Check if the text is appropriate and construction-related
    """
    response = client.moderations.create(input=text)
    return response.results[0].flagged

def is_construction_question(text):
    """Check if the given text is a valid construction-related question."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "Determine if the given text is a valid construction-related question. Return `1` if it is, else return `0`."},
            {"role": "user", "content": text}
        ],
        max_tokens=1,
        temperature=0,
        seed=0,
        logit_bias={"15": 100, "16": 100}
    )
    return int(response.choices[0].message.content)

st.title("Construction AI")
st.subheader("Get Instant Answers to Your Construction Questions")
st.write("Powered by AI and construction industry expertise.")

query = st.text_input("Ask a construction-related question", "What is the average cost per square foot for steel frame construction?")

button = st.empty()
if button.button("Search"):
    button.empty()
    
    with st.spinner("Validating your question..."):
        is_inappropriate = check_text(query)
        is_construction_related = is_construction_question(query)
    
    if is_inappropriate or not is_construction_related:
        st.warning("Please ask a valid construction-related question. Refresh the page to try again.", icon="🚫")
        st.stop()
    
    start_time = time.time()
    
    try:
        results = execute_search_agent(query)
        execution_time = round(time.time() - start_time, 2)
        st.success(f"Answer generated in {execution_time} seconds.")
        st.info(f"""### Question: {results['input']}
**Answer:** {results['output']}""")
    except Exception as e:
        st.error(f"An error occurred while processing your request: {str(e)}. Please try again.")
    
    st.info("Refresh the page to ask another construction-related question.")
