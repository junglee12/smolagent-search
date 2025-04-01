import streamlit as st
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
import requests
from PIL import Image
import io
import base64

# Main page setup
st.title("AI Research Assistant")
st.subheader("Research Query Input")

# Sidebar for API key, model selection, and instructions
with st.sidebar:
    st.header("Configuration")
    # Use session state to persist the API key during the session
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    api_key = st.text_input("Enter your Gemini API Key", type="password", value=st.session_state.api_key)
    if api_key:
        st.session_state.api_key = api_key

    # Model selection
    model_options = ['gemini/gemini-2.0-flash-lite', 'gemini/gemini-2.5-pro-exp-03-25']
    selected_model = st.selectbox("Select Model", model_options, index=0)  # Default to first option

    st.header("Instructions")
    st.write("""
    1. Enter your personal Gemini API key above (it’s masked for your privacy)
    2. Choose a model from the dropdown
    3. Input your research query in the text area below
    4. Click 'Run Query' to generate a detailed research report
    5. The report will include search phrases, search results, website analyses, and sources
    **Note**: Your API key stays in your session and is not stored or logged by this app.
    """)

# Model and agent setup (only initialize if API key is provided)
model = None
query_agent = None
if api_key:
    try:
        model = LiteLLMModel(model_id=selected_model, api_key=api_key)
        query_agent = CodeAgent(
            tools=[DuckDuckGoSearchTool()],
            additional_authorized_imports=[
                'pandas', 'statsmodels', 'sklearn', 'numpy', 'json', 're', 'requests', 'bs4', 'datetime',
            ],
            model=model,
            add_base_tools=True,
        )
    except Exception as e:
        st.error(f"Invalid API key or model initialization error: {str(e)}")

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

Please format the report in markdown (avoid 'katex-html') for better readability.
"""

# Run query button
if st.button("Run Query"):
    if not api_key:
        output_container.write("Please enter your Gemini API key in the sidebar to proceed.")
    elif not query_agent:
        output_container.write("API key is invalid or model failed to initialize. Please check your key.")
    else:
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