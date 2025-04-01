import streamlit as st
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel
# Removed unused imports: requests, PIL, io, base64 (unless used elsewhere)

# Main page setup
st.title("AI Research Assistant")
st.subheader("Research Query Input")

# --- Configuration Dictionary ---
# Maps user-friendly names to LiteLLM model IDs and necessary parameters
# Find standard LiteLLM model IDs here: https://docs.litellm.ai/docs/providers
MODEL_CONFIG = {
    "Gemini 2.0 Flash Lite (User Specified)": {
        "id": "gemini/gemini-2.0-flash-lite", # User specified ID
        "provider": "gemini",
        "key_name": "gemini_api_key"
    },
    "Gemini 2.5-Pro-Exp (User Specified)": {
        "id": "gemini/gemini-2.5-pro-exp-03-25", # User specified ID
        "provider": "gemini",
        "key_name": "gemini_api_key"
    },
    "OpenAI GPT-4o Mini": {
        "id": "openai/gpt-4o-mini", # Standard LiteLLM format
        "provider": "openai",
        "key_name": "openai_api_key"
    },
    "XAI Grok-2": {
        "id": "xai/grok-2-latest",
        "provider": "xai",
        "key_name": "xai_api_key"
    }
    # Add other models as needed
}

# Sidebar for API keys, model selection, and instructions
with st.sidebar:
    st.header("Configuration")

    # --- API Key Management ---
    # Initialize keys in session state if they don't exist
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    if "xai_api_key" not in st.session_state:
        st.session_state.xai_api_key = ""

    # Use session state to persist API keys during the session
    st.session_state.gemini_api_key = st.text_input(
        "Enter your Google AI / Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key
    )
    st.session_state.openai_api_key = st.text_input(
        "Enter your OpenAI API Key",
        type="password",
        value=st.session_state.openai_api_key
    )
    st.session_state.xai_api_key = st.text_input(
        "Enter your XAI (Grok) API Key",
        type="password",
        value=st.session_state.xai_api_key
    )

    # --- Model Selection ---
    model_options = list(MODEL_CONFIG.keys())
    selected_model_name = st.selectbox(
        "Select Model",
        model_options,
        index=0 # Default to the first model in the list
    )

    st.header("Instructions")
    st.write(f"""
    1. Enter the API key(s) required for the model you select.
       - **Gemini Models**: Requires Google AI / Gemini Key
       - **OpenAI Models**: Requires OpenAI Key
       - **XAI Grok Models**: Requires XAI Key
    2. Choose a model from the dropdown.
    3. Input your research query in the text area.
    4. Click 'Run Query' to generate a detailed research report.
    5. The report will include search phrases, search results, website analyses, and sources.
    **Note**: Your API keys stay in your session and are not stored or logged by this app. Keys are masked for privacy.
    Currently selected model requires: **{MODEL_CONFIG[selected_model_name]['provider'].upper()} API Key**
    """)

# --- Model and Agent Setup ---
model = None
query_agent = None
error_message = None

selected_config = MODEL_CONFIG[selected_model_name]
required_key_name = selected_config["key_name"]
api_key = st.session_state.get(required_key_name, "")

if api_key:
    try:
        # Prepare parameters for LiteLLMModel
        model_params = {
            "api_key": api_key
        }
        
        # Instantiate LiteLLMModel
        model = LiteLLMModel(
            model_id=selected_config["id"],
            **model_params # Pass api_key and optional base_url etc.
        )

        # Instantiate CodeAgent
        query_agent = CodeAgent(
            tools=[DuckDuckGoSearchTool()],
            additional_authorized_imports=[
                'pandas', 'statsmodels', 'sklearn', 'numpy', 'json', 're', 'requests', 'bs4', 'datetime',
            ],
            model=model,
            add_base_tools=True,
        )
    except Exception as e:
        error_message = f"Error initializing model '{selected_model_name}': {str(e)}. Check if the correct API key is provided and valid."
        st.error(error_message) # Display error immediately if initialization fails
else:
    # Only set error message if the button hasn't been clicked yet
    # If button is clicked later, a different message is shown there.
    if 'run_query_clicked' not in st.session_state:
         error_message = f"Please enter the {selected_config['provider'].upper()} API Key in the sidebar to use the '{selected_model_name}' model."

# Query input area
query = st.text_area("Enter your research query here",
                     "What's the latest development in AI agents?")
output_container = st.empty()

# Display placeholder or initial error message
if error_message and 'run_query_clicked' not in st.session_state:
     output_container.warning(error_message)

task = """
Based on the query: '{query}', perform the following steps and compile a report:

1. Perform a search with the query and collect the top search results.
2. Select all relevant websites from the search results.
3. For each selected website, extract key information related to the query.
4. Compile all the extracted information into a detailed, readable report, including:
   - A comprehensive answer to the user's query based on the findings.
   - A list of key points discovered.
   - Any surprising or unusual points ('weird points').
   - A concluding summary.
   - The list of search phrases used during the process.
   - The search results (URLs), indicating which ones were successfully used as sources (e.g., with a check mark ✔️ or similar).

Please format the report in markdown (avoid complex formatting like 'katex-html' if possible) for better readability.
"""

# Run query button
if st.button("Run Query"):
    st.session_state['run_query_clicked'] = True # Track button click
    if not api_key:
        output_container.error(f"Please enter the {selected_config['provider'].upper()} API Key in the sidebar to use the '{selected_model_name}' model.")
    elif not query_agent:
        # Display the initialization error if agent setup failed
        output_container.error(error_message or "Model or Agent failed to initialize. Please check your API key and configuration.")
    else:
        with st.spinner(f"Generating research report using {selected_model_name}..."):
            try:
                # Construct the full task with the user's query
                full_task = task.format(query=query)
                # Run the agent with the full task
                result = query_agent.run(full_task)
                # Display the result as markdown
                output_container.markdown(result)
            except Exception as e:
                output_container.error(f"Error occurred during report generation: {str(e)}")
    # Clean up the button click state if needed, or keep it for debugging
    # del st.session_state['run_query_clicked']