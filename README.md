# 🚀 Influencer Marketing Campaign Agent

This project implements an AI-powered Influencer Marketing Campaign Agent using Streamlit for the UI and LangGraph for orchestrating the agent's workflow. The agent automates several key stages of an influencer marketing campaign, from initialization and influencer discovery to AI-driven negotiation and performance reporting.

The primary goal is to showcase how LangGraph can be used to build complex, stateful AI agents that can interact with tools and make decisions based on an evolving state.

## ✨ Features

*   **Campaign Initialization**: Extracts campaign details (brand, goal, audience, budget, timeline, offers) from user input using an LLM.
*   **Influencer Discovery**: Searches a local JSON database (`influencer_data_stimuler.json`) for suitable influencers based on platform, keywords, and target audience. Includes fallback mechanisms if no influencers are found.
*   **Influencer Vetting & Selection**: Simulates a vetting process and selects the top influencers based on simulated scores (content quality, authenticity, engagement).
*   **AI-driven Negotiation**: Simulates a negotiation process between a "brand agent" LLM and an "influencer" LLM (persona-driven) for each selected influencer.
*   **Campaign Performance Monitoring**: Simulates campaign performance (impressions, engagement, clicks, conversions, ROI) if negotiations are successful.
*   **Report Generation**: Generates a summary report of the simulated campaign performance.
*   **Synthetic Data Generation**: Includes a script (`influencer_data_generator.py`) to create realistic, LLM-generated influencer profiles, including personas and video summaries tailored to a specific brand (Stimuler).
*   **Streamlit UI**: Provides an interactive web interface to input campaign details and observe the agent's progress and logs.

## 🛠️ Tech Stack

*   **Python 3.9+**
*   **Streamlit**: For the web application UI.
*   **LangGraph**: For building the stateful, multi-actor agent.
*   **Langchain**: Core library for LLM interactions, prompts, tools, and Pydantic models.
    *   `langchain-core`
    *   `langchain-google-genai`: For integrating with Google's Gemini models.
*   **Google Generative AI**: Utilizes Gemini models (e.g., `gemini-2.0-flash`, `gemini-1.5-flash`) for LLM tasks.
*   **Pydantic (v1)**: For data validation and structuring.
*   **Dotenv**: For managing environment variables.

## 📋 Prerequisites

1.  **Python**: Ensure you have Python 3.9 or newer installed.
2.  **Google API Key**: You'll need a Google API key with access to the Generative Language API (for Gemini models).
    *   Go to [Google AI Studio](https://aistudio.google.com/app/apikey) to create an API key.

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    A `requirements.txt` file should be created with the following content:
    ```txt
    streamlit
    langchain
    langchain-core
    langgraph
    langchain-google-genai
    pydantic==1.10.13 # Important for Langchain's Pydantic v1 usage
    python-dotenv
    google-generativeai
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```
    *Note on Pydantic*: The codebase uses `pydantic.v1` and `langchain_core.pydantic_v1`. Installing `pydantic==1.10.13` ensures compatibility. If you have Pydantic v2 installed globally, using a virtual environment is crucial.

4.  **Set up environment variables:**
    Create a `.env` file in the root directory of the project and add your Google API key:
    ```env
    GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    ```

5.  **Generate Influencer Data (First-time setup or to refresh data):**
    The application relies on `influencer_data_stimuler.json`. If this file doesn't exist or you want to generate fresh data:
    ```bash
    python influencer_data_generator.py
    ```
    This will create/overwrite `influencer_data_stimuler.json` with synthetic influencer profiles. This script also uses the `GOOGLE_API_KEY`.

## 🚀 Running the Application

Once the setup is complete and `influencer_data_stimuler.json` exists, run the Streamlit application:

```bash
streamlit run app.py
```

This will open the application in your default web browser. Enter the campaign details in the text area and click "Start Campaign" to initiate the agent.

## 📁 File Structure

```
.
├── app.py                      # Main Streamlit application and LangGraph agent logic
├── influencer_data_generator.py # Script to generate synthetic influencer data
├── influencer_data_stimuler.json # Generated influencer data (used by app.py)
├── .env                        # Stores API keys (not committed to Git)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🧠 How It Works

The agent operates through a series of states managed by LangGraph:

1.  **`initialize_campaign`**:
    *   Takes user input.
    *   Uses an LLM to parse the input and extract structured `CampaignDetails` (brand name, goal, audience, budget, timeline, initial/max offers).
    *   Sets default values if information is missing.

2.  **`find_influencers`**:
    *   Uses the `campaign_details` (especially `target_audience` and `campaign_goal`) to search the `influencer_data_stimuler.json` file.
    *   Keywords are extracted from the target audience's interests.
    *   Matching logic considers platform, niche, and keywords in video summaries.
    *   Includes a fallback to select random influencers or generate test influencers if no matches are found.

3.  **`vet_and_select`**:
    *   Takes the `influencers_found`.
    *   Simulates a vetting process (currently, it just passes the influencer through).
    *   Calculates a total score based on content quality, authenticity, and engagement rate.
    *   Selects the top 3 influencers based on this score.

4.  **`negotiate` (manage_negotiations)**:
    *   For each `selected_influencer`:
        *   Initiates an AI-driven negotiation using `negotiate_contract_with_ai`.
        *   Two LLMs are involved: one acting as the "brand agent" and another acting as the "influencer" (with a persona derived from `influencer.persona`).
        *   The negotiation proceeds in turns, with offers and counter-offers, aiming to reach an agreement within the campaign's budget constraints (`initial_offer`, `max_offer`).
        *   A separate, lightweight LLM (`verification_llm`) is used to determine if an agreement has been reached based on the influencer's messages.
        *   Negotiation logs are displayed in the Streamlit UI.

5.  **`monitor_and_report`**:
    *   If negotiations were successful (all selected influencers agreed):
        *   Simulates campaign performance using `monitor_campaign_performance_simulation` (generates random metrics like impressions, engagement, ROI).
        *   Generates a campaign report using `generate_report`.
    *   If not all negotiations were successful, it proceeds with only the successful ones or ends if none were successful.

6.  **Routing**:
    *   After `negotiate`, a conditional edge (`route_based_on_negotiation`) determines whether to proceed to `monitor` (if all agreements reached) or `END`.

The Streamlit UI provides input fields and displays the progress of these steps, including logs and negotiation dialogues.

## 🔧 Key Components

### `app.py`

*   **Streamlit UI**: Sets up the page, input fields, and containers for displaying agent output.
*   **Data Structures**: `CampaignDetails`, `CampaignState`, `NegotiateContractInput` (Pydantic models) define the data flow.
*   **LLM Setup**: Initializes `ChatGoogleGenerativeAI` for the main agent and negotiation.
*   **Influencer Data Loading**: `load_influencer_data` loads and validates data from `influencer_data_stimuler.json`.
*   **Tool Definitions**:
    *   `search_influencers_from_data`: Custom logic for searching local JSON.
    *   `vet_influencer_from_data`: Simulated vetting.
    *   `negotiate_contract_with_ai`: Core negotiation logic between two AI agents.
    *   `monitor_campaign_performance_simulation`: Simulates campaign metrics.
    *   `generate_report`: Creates a text-based report.
*   **Agent Nodes**: Functions corresponding to each step in the LangGraph workflow (`initialize_campaign`, `find_influencers`, etc.).
*   **LangGraph Workflow**: Defines the graph, nodes, edges, and conditional routing.
*   **Async Execution**: Uses `asyncio` and `graph.astream_log` to run the agent and stream updates to the UI.

### `influencer_data_generator.py`

*   **Purpose**: Creates the `influencer_data_stimuler.json` file.
*   **`Influencer` Model**: Pydantic model defining the structure of an influencer's data.
*   **LLM Integration**: Uses `ChatGoogleGenerativeAI` (Gemini) to:
    *   Generate realistic video titles and summaries tailored to a brand ("Stimuler").
    *   Generate detailed influencer personas (personality, communication style, values, quirks, etc.).
*   **Data Synthesis**: Combines LLM-generated content with randomly generated data (followers, engagement, platform, niche) to create comprehensive influencer profiles.

## 💡 Potential Improvements

*   Integrate with actual social media APIs for real influencer data.
*   Implement more sophisticated vetting logic (e.g., sentiment analysis of comments, audience overlap analysis).
*   Allow human-in-the-loop for negotiation approval or guidance.
*   Store campaign state and results in a database.
*   More detailed and visually appealing report generation.
*   Error handling and resilience for API calls and data processing.
