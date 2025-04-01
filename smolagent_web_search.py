import streamlit as st
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
import requests
from PIL import Image
import io
import base64

# Main page setup
st.title("AI Research Assistant")
st.subheader("Research Query Input")

# Sidebar for instructions (no API key input needed)
with st.sidebar:
    st.header("Instructions")
    st.write("""
    1. Input your research query in the text area below
    2. Click 'Run Query' to generate a detailed research report
    3. The report will include search phrases, search results, website analyses, and sources
    4. View the report in the output area
    """)

# Load API key from Streamlit secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("Gemini API key not configured. Please contact the app administrator.")
    st.stop()

# Model setup
model = LiteLLMModel(model_id="gemini/gemini-2.0-flash-lite", api_key=api_key)

# Agent setup
query_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    additional_authorized_imports=[
        'pandas', 'statsmodels', 'sklearn', 'numpy', 'json', 're', 'requests', 'bs4', 'datetime',
    ],
    model=model,
    add_base_tools=True,
)

# Query input area
query = st.text_area("Enter your research query here",
                     "What's the latest development in AI agents?")
output_container = st.empty()

task = """
Based on the query: '{query}', perform the following steps and compile a report:

1. Perform a search with the query and collect the top search results.

2. Select all websites from the search results.

3. For each selected website, extract information from the website.

4. Compile all the extracted information into a detailed, readable report, including:

   - Generate answer to the user's query
   - List the key points, weird points, and conclusion
   - The list of search phrases
   - The search results put a check mark if used as a source

Please format the report in markdown for better readability.
"""

# Run query button
if st.button("Run Query"):
    with st.spinner("Generating research report..."):
        try:
            # Construct the full task with the user's query
            full_task = task.format(query=query)
            # Run the agent with the full task
            result = query_agent.run(full_task)
            # Display the result as markdown
            output_container.markdown(result)
        except Exception as e:
            output_container.write(f"Error occurred: {str(e)}")