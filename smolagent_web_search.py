# Add 're' to your imports at the top
import re
import streamlit as st
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, tool
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import requests
import json
import os

os.system("playwright install")

# --- Your Tool Definitions (playwright_web_fetcher, brave_searcher) remain the same ---
@tool
def playwright_web_fetcher(url: str) -> str:
    """Fetches the full HTML content of the provided URL using a headless browser (Playwright).

    Use this tool for accessing websites, especially those that rely heavily on JavaScript rendering.
    It returns the full HTML content as a string if successful, or an error message string
    (starting with 'Error:') if fetching fails (e.g., timeout, invalid URL, other exceptions).

    Args:
        url (str): The complete URL (must start with http:// or https://) of the web page to fetch.
    """
    # Function body remains the same...
    if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
        return "Error: Invalid input. URL must be a string starting with http:// or https://"
    page_content = None
    error_message = None
    st.info(f"[Tool: playwright_web_fetcher] Attempting to fetch: {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_default_timeout(30000)
            page.goto(url, wait_until='domcontentloaded')
            page_content = page.content()
            browser.close()
            st.info(f"[Tool: playwright_web_fetcher] Successfully fetched content (Length: {len(page_content)})")
    except PlaywrightTimeoutError:
        error_message = f"Error: Timeout occurred while loading URL: {url}"
        st.error(error_message)
        if 'browser' in locals() and browser.is_connected(): browser.close()
    except Exception as e:
        error_message = f"Error: Failed to fetch URL '{url}' using Playwright: {type(e).__name__} - {str(e)}"
        st.error(error_message)
        if 'browser' in locals() and browser.is_connected(): browser.close()
    if page_content:
        return page_content
    else:
        return error_message or f"Error: Unknown error fetching URL: {url}"

@tool
def brave_searcher(query: str) -> str:
    """Performs a web search using the Brave Search API and returns formatted results.

    Use this tool to search the web for information based on a query string.
    It returns a formatted string containing the top search results (title, URL, description),
    or an error message string if the search fails or the API key is missing.

    Args:
        query (str): The search query string.
    """
    # --- Get API Key from Streamlit Session State ---
    api_key = st.session_state.get("brave_api_key")
    if not api_key:
        return "Error: Brave Search API Key not found in session state. Please configure it in the sidebar."
    # Function body remains the same...
    search_url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json","Accept-Encoding": "gzip","X-Subscription-Token": api_key}
    params = {"q": query,"count": 5}
    st.info(f"[Tool: brave_searcher] Searching Brave for: {query}")
    try:
        response = requests.get(search_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get("web", {}).get("results", [])
        if not results:
            return f"No results found from Brave Search for query: {query}"
        formatted_results = []
        for i, result in enumerate(results):
            title = result.get("title", "No Title")
            url = result.get("url", "No URL")
            description = result.get("description", "No Description")
            formatted_results.append(f"Result {i+1}:\nTitle: {title}\nURL: {url}\nDescription: {description}\n---")
        st.info(f"[Tool: brave_searcher] Found {len(results)} results for: {query}")
        return "\n".join(formatted_results)
    except requests.exceptions.Timeout:
        st.error(f"[Tool: brave_searcher] Request timed out for query: {query}")
        return f"Error: Brave Search API request timed out for query: {query}"
    except requests.exceptions.HTTPError as http_err:
        st.error(f"[Tool: brave_searcher] HTTP error occurred: {http_err} - Response: {response.text}")
        return f"Error: Brave Search API returned HTTP error {response.status_code} for query: {query}"
    except requests.exceptions.RequestException as req_err:
        st.error(f"[Tool: brave_searcher] Request exception occurred: {req_err}")
        return f"Error: Failed to connect to Brave Search API for query: {query}. Details: {req_err}"
    except json.JSONDecodeError:
        st.error(f"[Tool: brave_searcher] Failed to decode JSON response from Brave API.")
        return f"Error: Could not parse Brave Search API response for query: {query}"
    except KeyError as key_err:
        st.error(f"[Tool: brave_searcher] Unexpected response format from Brave API: Missing key {key_err}")
        return f"Error: Unexpected response format from Brave Search API for query: {query}"
    except Exception as e:
        st.error(f"[Tool: brave_searcher] An unexpected error occurred: {e}")
        return f"Error: An unexpected error occurred in brave_searcher for query: {query}. Details: {e}"


# --- The rest of your Streamlit app setup (title, config, sidebar) remains the same ---
# Main page setup
st.title("AI Research Assistant")
st.subheader("Research Query Input")

# --- Configuration Dictionary (Keep as is) ---
MODEL_CONFIG = {
    "Gemini 2.0 Flash Lite (User Specified)": {"id": "gemini/gemini-2.0-flash-lite", "provider": "gemini", "key_name": "gemini_api_key"},
    "Gemini 2.5-Pro-Exp (User Specified)": {"id": "gemini/gemini-2.5-pro-exp-03-25", "provider": "gemini", "key_name": "gemini_api_key"},
    "OpenAI GPT-4o Mini": {"id": "openai/gpt-4o-mini", "provider": "openai", "key_name": "openai_api_key"},
    "XAI Grok-2": {"id": "xai/grok-2-latest", "provider": "xai", "key_name": "xai_api_key"}
}

# Sidebar
with st.sidebar:
    st.header("Configuration")
    # --- API Key Management ---
    if "gemini_api_key" not in st.session_state: st.session_state.gemini_api_key = ""
    if "openai_api_key" not in st.session_state: st.session_state.openai_api_key = ""
    if "xai_api_key" not in st.session_state: st.session_state.xai_api_key = ""
    if "brave_api_key" not in st.session_state: st.session_state.brave_api_key = ""
    st.session_state.gemini_api_key = st.text_input("Enter your Google AI / Gemini API Key", type="password", value=st.session_state.gemini_api_key)
    st.session_state.openai_api_key = st.text_input("Enter your OpenAI API Key", type="password", value=st.session_state.openai_api_key)
    st.session_state.xai_api_key = st.text_input("Enter your XAI (Grok) API Key", type="password", value=st.session_state.xai_api_key)
    st.session_state.brave_api_key = st.text_input("Enter your Brave Search API Key", type="password", value=st.session_state.brave_api_key)
    # --- Model Selection ---
    model_options = list(MODEL_CONFIG.keys())
    selected_model_name = st.selectbox("Select Model", model_options, index=0)
    st.header("Instructions")
    st.warning("Requires Playwright setup: Run `pip install playwright` and then **`playwright install`** in your terminal.")
    st.write(f"""
    1. Enter API keys required for the model and tools.
    2. Choose a model.
    3. Input query.
    4. Click 'Run Query'. Agent uses search tools and Playwright.
    Currently selected model requires: **{MODEL_CONFIG[selected_model_name]['provider'].upper()} API Key**
    """)

# --- Model and Agent Setup (remains the same) ---
model = None
query_agent = None
error_message = None
selected_config = MODEL_CONFIG[selected_model_name]
required_key_name = selected_config["key_name"]
api_key = st.session_state.get(required_key_name, "")
if api_key:
    try:
        model_params = {"api_key": api_key}
        model = LiteLLMModel(model_id=selected_config["id"], **model_params)
        query_agent = CodeAgent(
            tools=[DuckDuckGoSearchTool(), playwright_web_fetcher, brave_searcher],
            additional_authorized_imports=['pandas', 'statsmodels', 'sklearn', 'numpy', 'json', 're','requests', 'bs4', 'datetime','playwright'],
            model=model, add_base_tools=True,
        )
    except Exception as e:
        error_message = f"Error initializing model or agent '{selected_model_name}': {str(e)}."
        st.error(error_message)
else:
    if 'run_query_clicked' not in st.session_state:
        provider_name = selected_config.get('provider', 'selected provider').upper()
        error_message = f"Please enter the {provider_name} API Key in the sidebar."

# Query input area
query = st.text_area("Enter your research query here", "What's the latest development in AI agents?")
output_container = st.empty()

if error_message and not api_key and 'run_query_clicked' not in st.session_state:
     output_container.warning(error_message)


# --- TASK PROMPT (Consider adding stricter formatting rules) ---
task = """
Based on the query: '{query}', perform the following steps and compile a report:

1.  **Perform a web search** using `DuckDuckGoSearchTool` or `brave_searcher`. Choose one. Collect top results (URLs, descriptions). If `brave_searcher` fails due to key, use DuckDuckGo.

2.  Select all website URLs from the search results.

3.  For each selected URL:
    a. Use 'playwright_web_fetcher' tool (provide URL as 'url' arg) to fetch HTML.
    b. If error occurs, record URL/error, skip.
    c. If successful, parse HTML (e.g., using BeautifulSoup from bs4) for key info relevant to '{query}'. Extract meaningful text.

4.  Compile findings into a detailed, readable report:
    *   Comprehensive answer.
    *   Key points.
    *   Weird points.
    *   Conclusion.
    *   Search phrases used.
    *   Search tool used (`DuckDuckGoSearchTool` or `brave_searcher`).
    *   Website URLs attempted (indicate ✔️ success or ❌ failure type).

**IMPORTANT Formatting Rule:** Format the report in **standard Markdown ONLY**. Avoid non-standard syntax, directives (like `::` or `:::` at the start of lines), admonitions, or complex block elements. Stick to headings, lists, bold, italics, links, and standard code blocks (```).
"""
# --- END TASK PROMPT ---


# Run query button
if st.button("Run Query"):
    st.session_state['run_query_clicked'] = True
    if not api_key:
        provider_name = selected_config.get('provider', 'selected provider').upper()
        output_container.error(f"Please enter the {provider_name} API Key.")
    elif not query_agent:
        output_container.error(error_message or f"Agent for '{selected_model_name}' failed to initialize.")
    else:
        output_container.empty()
        with st.spinner(f"Generating research report using {selected_model_name}..."):
            st.info("Starting report generation process...")
            try:
                full_task = task.format(query=query)
                result = query_agent.run(full_task)

                # --- SANITIZATION STEP ---
                sanitized_result = result
                # Remove lines starting with :: or ::: (common directive triggers)
                # Use re.MULTILINE to check start of each line
                sanitized_result = re.sub(r'^\s*[:]{2,}.*$', '', sanitized_result, flags=re.MULTILINE)
                # Optional: Add more rules here if needed, e.g., escaping specific characters

                if result != sanitized_result:
                    st.warning("Applied sanitization to agent output to prevent rendering errors.")
                # --- END SANITIZATION ---

                # Display the sanitized result
                output_container.markdown(sanitized_result)
                st.success("Report generation complete!")

            except Exception as e:
                st.error(f"An error occurred during report generation: {str(e)}")
                # st.exception(e) # Uncomment for full traceback if needed
                # --- Add this to see the raw output that caused the error ---
                st.error("Raw agent output that potentially caused the error:")
                st.text(result if 'result' in locals() else "Result not generated before error.")