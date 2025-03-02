import streamlit as st
import asyncio
import os
import json
import random
from typing import Dict, List, TypedDict, Sequence, Callable, Any
from dotenv import load_dotenv

from pydantic.v1 import BaseModel as BaseModelV1, Field as FieldV1
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.tools import Tool
from langgraph.prebuilt import ToolInvocation
from langchain_core.tools import StructuredTool

# Import the Influencer class
from influencer_data_generator import Influencer

load_dotenv()

# --- Streamlit App ---
st.set_page_config(page_title="Influencer Marketing Agent", page_icon="🚀")

# --- Global Variable Initialization ---
global influencer_data
influencer_data = []

# --- Data Structures ---
class CampaignDetails(BaseModel):
    brand_name: str = Field(description="The name of the brand.")
    campaign_goal: str = Field(description="The primary goal of the campaign.")
    target_audience: Dict[str, str] = Field(description="Target audience demographics.")
    budget: float = Field(description="The total budget for the campaign.")
    timeline: str = Field(description="The campaign timeline.")
    content_guidelines: str = Field(description="Content guidelines.")
    legal_requirements: str = Field(description="Legal requirements.")
    initial_offer: str = Field(description="The initial offer.")
    max_offer: str = Field(description="The maximum offer.")

class CampaignState(TypedDict):
    campaign_details: CampaignDetails
    influencers_found: List[Influencer]
    selected_influencers: List[Influencer]
    negotiation_status: Dict[str, str]
    campaign_performance: Dict
    messages: Sequence[BaseMessage]
    negotiation_messages: Dict[str, List[BaseMessage]]

class NegotiateContractInput(BaseModelV1):
    influencer: Influencer = FieldV1(description="The influencer to negotiate with")
    campaign_details: CampaignDetails = FieldV1(description="Details of the campaign")
    agent_llm: ChatGoogleGenerativeAI = FieldV1(description="The LLM agent for negotiation")
    state: Dict = FieldV1(description="The current campaign state as a dictionary")

    class Config:
        arbitrary_types_allowed = True  # Allow custom types like Influencer, ChatGoogleGenerativeAI, etc.

# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), convert_system_message_to_human=True)

# --- Load Influencer Data ---
@st.cache_resource
def load_influencer_data(filepath: str = "influencer_data_stimuler.json") -> List[Influencer]:
    """Loads influencer data from a JSON file with improved error handling."""
    global influencer_data  # Reference the global variable

    try:
        if not os.path.exists(filepath):
            st.error(f"❌ ERROR: Influencer data file not found: {filepath}")
            # Look for alternative files
            json_files = [f for f in os.listdir() if f.endswith('.json')]
            if json_files:
                alt_file = json_files[0]
                st.warning(f"Found alternative file: {alt_file}")
                filepath = alt_file
            else:
                st.warning("No JSON files found in directory")
                return []
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # st.success(f"Loaded {len(data)} influencers from {filepath}")
        
        # Validate the data structure
        valid_influencers = []
        for item in data:
            try:
                # Ensure persona field exists
                if 'persona' not in item or not isinstance(item['persona'], dict):
                    item['persona'] = {
                        "personality": "engaging and motivational",
                        "communication_style": "clear and direct",
                        "values": "education and self-improvement",
                        "quirks": "uses catchphrases",
                        "speaking_style": "articulate",
                        "tone": "informative"
                    }
                
                # Create the Influencer object
                influencer = Influencer(**item)
                valid_influencers.append(influencer)
            except Exception as e:
                st.error(f"Error parsing influencer: {e}")
        
        # st.success(f"Validated {len(valid_influencers)} influencers")
        influencer_data = valid_influencers  # Update the global variable
        return valid_influencers

    except Exception as e:
        st.error(f"❌ ERROR loading influencer data: {e}")
        return []

# --- Tool Definitions ---
def search_influencers_from_data(query_data: List[Influencer], platform: str, keywords: List[str], target_audience: Dict[str, str]) -> List[Influencer]:
    """Searches for influencers with improved matching logic."""
    # Convert keywords to lowercase for case-insensitive matching
    keywords_lower = [k.lower() for k in keywords]
    
    # Get age range and interests from target audience
    target_age = target_audience.get("age", "")
    target_interests = target_audience.get("interests", "").lower().split(",")
    target_interests = [interest.strip() for interest in target_interests]
    
    results = []
    
    # Expanded keywords for language learning
    expanded_keywords = [
        "language", "learning", "english", "ielts", "toefl", "esl", "education",
        "study", "tutor", "teacher", "student", "speaking", "linguistics"
    ]
    for kw in keywords_lower:
        if kw not in expanded_keywords:
            expanded_keywords.append(kw)
    
    for influencer in influencer_data:
        # Flexible platform matching
        platform_match = (
            platform.lower() == "any" or
            platform.lower() == "all" or
            influencer.platform.lower() == platform.lower() or
            (platform.lower() == "youtube" and "tube" in influencer.platform.lower()) or
            (platform.lower() == "instagram" and "insta" in influencer.platform.lower()) or
            (platform.lower() == "tiktok" and "tik" in influencer.platform.lower())
        )
        
        if not platform_match:
            continue
        
        # Keyword matching in niche or video summaries
        niche_lower = influencer.niche.lower()
        keyword_match = False
        
        for kw in expanded_keywords:
            if kw in niche_lower:
                keyword_match = True
                break
        
        if not keyword_match and hasattr(influencer, 'video_summaries'):
            for video in influencer.video_summaries:
                video_title = video.get('title', '').lower()
                video_summary = video.get('summary', '').lower()
                for kw in expanded_keywords:
                    if kw in video_title or kw in video_summary:
                        keyword_match = True
                        break
                if keyword_match:
                    break
        
        if not keyword_match:
            continue
        
        results.append(influencer)
    
    # Fallback to random influencers if no matches
    if not results and influencer_data:
        import random
        fallback_count = min(3, len(influencer_data))
        results = random.sample(influencer_data, fallback_count)
    
    return results

def vet_influencer_from_data(influencer: Influencer) -> Influencer:
    """Vets an influencer (simulated)."""
    return influencer

def create_influencer_llm(influencer: Influencer):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"""You are {influencer.name} (@{influencer.handle}), an influencer on {influencer.platform}.
    Niche: {influencer.niche}. Persona:
    Personality: {influencer.persona.get('personality', 'N/A')}
    Communication Style: {influencer.persona.get('communication_style', 'N/A')}
    Values: {influencer.persona.get('values', 'N/A')}
    Quirks: {influencer.persona.get('quirks', 'N/A')}
    Speaking Style: {influencer.persona.get('speaking_style', 'N/A')}
    Tone: {influencer.persona.get('tone', 'N/A')}
    Negotiate realistically for a fair price. Consider your follower count, engagement, and campaign details.
    """
    ),
    MessagesPlaceholder(variable_name="messages")
    ])
    st.write(f"DEBUG: Influencer LLM Prompt: {prompt_template}")  # Add debug
    return prompt_template | llm

async def negotiate_contract_with_ai(influencer: Influencer, campaign_details: CampaignDetails, agent_llm, state: CampaignState, chat_container) -> str:
    import time
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    import re
    import streamlit as st

    # Initialize chat display
    chat_container.markdown(f"\n\n{'='*80}\n\n💬 NEGOTIATION SESSION WITH @{influencer.handle}\n\n")
    chat_container.markdown(f"Influencer: {influencer.name} (@{influencer.handle})\n\nPlatform: {influencer.platform}\n\nNiche: {influencer.niche}\n\nFollowers: {influencer.followers}")
    chat_container.markdown(f"-"*80 + "\n\n Influencer Persona:")
    for key, value in influencer.persona.items():
        chat_container.markdown(f"  {key.replace('_', ' ').title()}: {value}")
    chat_container.markdown(f"-"*80)
    chat_container.markdown(f"Campaign: {campaign_details.brand_name} - {campaign_details.campaign_goal}")
    chat_container.markdown(f"Initial Offer: ${float(campaign_details.initial_offer)} | Max Offer: ${float(campaign_details.max_offer)}")
    chat_container.markdown(f"{'='*80}\n\n")

    # Verification LLM
    from langchain_google_genai import ChatGoogleGenerativeAI
    verification_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        convert_system_message_to_human=True
    )

    # Initialize negotiation state
    if "negotiation_messages" not in state:
        state["negotiation_messages"] = {}
    if influencer.handle not in state["negotiation_messages"]:
        state["negotiation_messages"][influencer.handle] = []
    negotiation_messages = state["negotiation_messages"][influencer.handle]

    # Calculate rates
    min_fair_rate = max(300, int(influencer.followers / 100))
    ideal_rate = max(500, int(influencer.followers / 50))
    max_rate = max(1000, int(influencer.followers / 25))

    # Initial message
    if not negotiation_messages:
        initial_message = (f"Hi {influencer.name}, I'm representing {campaign_details.brand_name}. We'd like to collaborate on "
                           f"a campaign for {campaign_details.campaign_goal}. We're offering ${campaign_details.initial_offer} "
                           f"for a sponsored post highlighting our language learning platform. Are you interested?")
        negotiation_messages.append(HumanMessage(content=initial_message))
        chat_container.markdown(f"🤖 BRAND AGENT: {initial_message}")

    max_turns = 5
    current_offer = float(campaign_details.initial_offer)
    status = "Negotiation in progress"
    influencer_counter = 0

    for turn in range(max_turns):
        chat_container.markdown(f"\n\n[Turn {turn+1}/{max_turns}]")
        time.sleep(1)

        # Influencer response
        try:
            if turn == 0:
                influencer_strategy = f"""
                Express interest in the collaboration but professionally state your typical rate (${ideal_rate}).
                Ask about the specific deliverables expected for this campaign.
                Be friendly but firm about your value.
                """
            elif turn == 1:
                influencer_strategy = f"""
                Acknowledge their counter-offer but remain firm on your value.
                If their offer is still far below your rate, suggest a middle ground of ${min(max_rate, int((ideal_rate + current_offer)/2))}.
                Ask about specific deliverables to determine if the scope can be adjusted.
                """
            elif turn == 2:
                influencer_strategy = f"""
                Make your final position clear.
                If their offer is at least ${min_fair_rate}, consider accepting.
                If still too low, politely suggest that your rates start at ${min_fair_rate} for minimal content.
                """
            else:
                influencer_strategy = f"""
                Make a final decision based on their latest offer.
                Accept if it's at least ${min_fair_rate}.
                Decline if it's below that, but leave the door open for future collaborations.
                Keep it brief and professional.
                """

            influencer_instruction = f"""You are {influencer.name} (@{influencer.handle}), an influencer on {influencer.platform} with {influencer.followers} followers.
Your niche is: {influencer.niche}
Persona traits: {', '.join(f"{k}: {v}" for k, v in influencer.persona.items())}
NEGOTIATION CONTEXT:
- The brand "{campaign_details.brand_name}" wants you to promote their language learning platform
- Their current offer is ${current_offer}
- Your ideal rate would be ${ideal_rate} based on your audience size
- The minimum you'd accept is ${min_fair_rate}
YOUR STRATEGY:
{influencer_strategy}
IMPORTANT:
- Respond as a professional influencer negotiating a business deal
- Be articulate but conversational
- If this negotiation has dragged on too long, be direct in your final answer
- Maximum 3-4 paragraphs in your response
"""
            messages = [
                SystemMessage(content=influencer_instruction),
                HumanMessage(content=f"Brand's latest message: {negotiation_messages[-1].content}\n\nYour response:")
            ]
            
            inf_response = await agent_llm.ainvoke(messages)
            time.sleep(1.5)
            
            if not inf_response.content.strip():
                inf_response_text = f"Thanks for your offer of ${current_offer}. " + (
                    f"That works for me!" if current_offer >= min_fair_rate else
                    f"My standard rate is ${ideal_rate}. I'd need at least ${min_fair_rate}."
                )
                inf_response = AIMessage(content=inf_response_text)
                chat_container.markdown(f"⚠️ WARNING: Empty response. Using fallback.")
            
            counter_matches = re.findall(r'\$([0-9,]+(?:\.[0-9]+)?)', inf_response.content)
            for match in counter_matches:
                match = match.replace(',', '')
                try:
                    potential_counter = float(match)
                    if potential_counter > current_offer * 1.05 and potential_counter > min_fair_rate:
                        influencer_counter = potential_counter
                        chat_container.markdown(f"💰 Influencer counter-offer: ${influencer_counter}")
                        break
                except ValueError:
                    continue
            
            negotiation_messages.append(inf_response)
            chat_container.markdown(f"👤 @{influencer.handle}: {inf_response.content}")
            time.sleep(1)
            
            verification_result = await verify_negotiation_status(verification_llm, inf_response.content, current_offer)
            if verification_result == "accepted":
                status = "Agreement reached"
                chat_container.markdown(f"✅ AGREEMENT REACHED at ${current_offer}!")
                break
            elif verification_result == "rejected":
                status = "Agreement failed"
                chat_container.markdown(f"❌ AGREEMENT FAILED")
                break
            time.sleep(1)
            
        except Exception as e:
            chat_container.markdown(f"⚠️ ERROR in influencer response: {e}")
            inf_response = AIMessage(content=f"My rates start at ${min_fair_rate}. Let me know if that works.")
            negotiation_messages.append(inf_response)
            chat_container.markdown(f"👤 @{influencer.handle}: {inf_response.content}")
        
        if influencer_counter > 0 and influencer_counter > float(campaign_details.max_offer) * 1.5:
            chat_container.markdown(f"⚠️ Large gap: ${influencer_counter} vs max ${campaign_details.max_offer}")
            status = "Agreement unlikely - too far apart"
            if current_offer < float(campaign_details.max_offer) * 0.95:
                final_offer = float(campaign_details.max_offer) * 0.95
                final_message = f"Our max is ${final_offer:.0f}. Would that work for a single post?"
                negotiation_messages.append(HumanMessage(content=final_message))
                chat_container.markdown(f"🤖 BRAND AGENT: {final_message}")
                current_offer = final_offer

                time.sleep(1.5)
                final_response = await agent_llm.ainvoke([
                    SystemMessage(content=f"Final offer: ${final_offer:.0f}. Min acceptable: ${min_fair_rate}. Decide."),
                    HumanMessage(content=f"Final offer: {final_message}\n\nYour response:")
                ])
                time.sleep(1.5)

                if not final_response.content.strip():
                    final_response = AIMessage(content=f"${final_offer:.0f} {'works' if final_offer >= min_fair_rate else 'is below my minimum of $' + str(min_fair_rate)}.")
                negotiation_messages.append(final_response)
                chat_container.markdown(f"👤 @{influencer.handle}: {final_response.content}")

                time.sleep(1)
                status = "Agreement reached" if await verify_negotiation_status(verification_llm, final_response.content, final_offer) == "accepted" else "Agreement failed"
            break

        time.sleep(1)

        # Brand response
        try:
            next_offer = min(
                (current_offer + influencer_counter) / 2 if influencer_counter > 0 else current_offer * (1.15 if turn == 0 else 1.2),
                float(campaign_details.max_offer) * (0.85 if turn < 2 else 0.95)
            )
            next_offer = round(next_offer / 25) * 25
            next_offer = max(next_offer, current_offer)

            agent_strategy = (
                f"Acknowledge their rate, offer ${next_offer:.0f}, outline deliverables." if turn == 0 else
                f"Address concerns, offer ${next_offer:.0f}, near limit." if turn == 1 else
                f"Final offer ${next_offer:.0f}, emphasize partnership value."
            )
            agent_instruction = f"""You are a brand rep for {campaign_details.brand_name}.
Negotiating with {influencer.name} (@{influencer.handle}).
Current: ${current_offer}, Next: ${next_offer}, Max: ${campaign_details.max_offer}.
Strategy: {agent_strategy}
Respond professionally, max 3 paragraphs."""

            agent_response = await agent_llm.ainvoke([
                SystemMessage(content=agent_instruction),
                HumanMessage(content=f"Influencer: {inf_response.content}\n\nYour response:")
            ])
            time.sleep(1.5)

            if not agent_response.content.strip():
                agent_response = HumanMessage(content=f"We can offer ${next_offer:.0f} for a post. Does that work?")
                chat_container.markdown(f"⚠️ Empty agent response, using fallback: ${next_offer:.0f}")

            negotiation_messages.append(agent_response)
            chat_container.markdown(f"🤖 BRAND AGENT: {agent_response.content}")
            current_offer = next_offer
            chat_container.markdown(f"💰 Current offer: ${current_offer}")

        except Exception as e:
            chat_container.markdown(f"⚠️ ERROR in agent response: {e}")
            next_offer = min(current_offer * 1.1, float(campaign_details.max_offer))
            agent_response = HumanMessage(content=f"We can offer ${next_offer:.0f}. Does that work?")
            negotiation_messages.append(agent_response)
            chat_container.markdown(f"🤖 BRAND AGENT: {agent_response.content}")
            current_offer = next_offer

        if current_offer >= float(campaign_details.max_offer) * 0.95:
            chat_container.markdown(f"⚠️ Near max offer: ${campaign_details.max_offer}")

    if status == "Negotiation in progress":
        time.sleep(1)
        final_message = f"Our final offer is ${current_offer:.0f}. Does that work?"
        negotiation_messages.append(HumanMessage(content=final_message))
        chat_container.markdown(f"🤖 BRAND AGENT: {final_message}")

        final_response = await agent_llm.ainvoke([
            SystemMessage(content=f"Final offer: ${current_offer:.0f}. Min: ${min_fair_rate}. Decide."),
            HumanMessage(content=f"Final offer: {final_message}\n\nYour response:")
        ])
        time.sleep(1.5)

        if not final_response.content.strip():
            final_response = AIMessage(content=f"${current_offer:.0f} {'works' if current_offer >= min_fair_rate else 'is below my $' + str(min_fair_rate)}.")
        negotiation_messages.append(final_response)
        chat_container.markdown(f"👤 @{influencer.handle}: {final_response.content}")

        time.sleep(1)
        status = "Agreement reached" if await verify_negotiation_status(verification_llm, final_response.content, current_offer) == "accepted" else "Agreement failed"

    chat_container.markdown(f"\n\nNEGOTIATION RESULT: {status}")
    state["negotiation_messages"][influencer.handle] = negotiation_messages
    return status

async def verify_negotiation_status(verification_llm, message_content, current_offer):
    """
    Uses a lightweight LLM to verify if an influencer has accepted or rejected an offer.
    Returns: "accepted", "rejected", or "continuing"
    """
    try:
        verification_prompt = f"""
        TASK: Determine if the influencer has ACCEPTED or REJECTED the brand's offer of ${current_offer}, or if the negotiation is CONTINUING.

        Influencer's message:
        "{message_content}"

        Analyze ONLY THIS MESSAGE and determine if the influencer has:
        1. ACCEPTED the offer
        2. REJECTED the offer
        3. Neither clearly accepted nor rejected (CONTINUING)

        Output EXACTLY ONE of these words: "accepted", "rejected", or "continuing"
        """
        
        # Clean up the prompt by removing extra whitespace
        verification_prompt = "\n\n".join(line.strip() for line in verification_prompt.split("\n\n"))
        
        # Use direct LLM call for verification
        response = await verification_llm.ainvoke([HumanMessage(content=verification_prompt)])
        
        result = response.content.lower().strip()
        
        # Normalize the response
        if "accept" in result:
            return "accepted"
        elif "reject" in result:
            return "rejected"
        else:
            return "continuing"
    except Exception as e:
        st.error(f"⚠️ ERROR in verification: {e}")
        # If verification fails, base the decision on keywords as a fallback
        message_lower = message_content.lower()
        
        if ("accept" in message_lower and "offer" in message_lower and "not" not in message_lower[:20]) or \
           ("agree" in message_lower and "work" in message_lower and "not" not in message_lower[:20]) or \
           ("look forward" in message_lower and "collaborating" in message_lower):
            return "accepted"
        elif ("decline" in message_lower) or \
             ("reject" in message_lower) or \
             ("cannot accept" in message_lower) or \
             ("unfortunately" in message_lower and "below" in message_lower):
            return "rejected"
        else:
            return "continuing"

def monitor_campaign_performance_simulation(campaign_details: CampaignDetails, selected_influencers: List[Influencer]) -> Dict:
    """Simulates campaign performance."""
    st.write(f"*** Simulating campaign performance for {campaign_details.brand_name} ***")
    num_influencers = len(selected_influencers)
    impressions = random.randint(10000 * num_influencers, 100000 * num_influencers)
    engagement = int(impressions * random.uniform(0.01, 0.05))
    clicks = int(engagement * random.uniform(0.05, 0.2))
    conversions = int(clicks * random.uniform(0.01, 0.1))
    roi = round((conversions * 50) / campaign_details.budget, 2) if campaign_details.budget > 0 else 0
    return {"impressions": impressions, "engagement": engagement, "clicks": clicks, "conversions": conversions, "ROI": roi}

def generate_report(campaign_performance: Dict) -> str:
    """Generates a campaign report."""
    st.write("*** Generating campaign report ***")
    report = f"""
    Campaign Performance Report:

    Impressions: {campaign_performance.get('impressions', 0)}
    Engagement: {campaign_performance.get('engagement', 0)}
    Clicks: {campaign_performance.get('clicks', 0)}
    Conversions: {campaign_performance.get('conversions', 0)}
    ROI: {campaign_performance.get('ROI', 0)}

    Overall Assessment: ... (Further analysis would be added here)
    """
    return report

# --- Tool Creation ---
influencer_data = load_influencer_data()  # Load data before defining tools

tools = [
    Tool(
        name="search_influencers",
        func=lambda platform, keywords, target_audience: search_influencers_from_data(
            influencer_data, platform, keywords, target_audience),
        description="Searches for influencers.",
    ),
    Tool(
        name="vet_influencer",
        func=vet_influencer_from_data,
        description="Vets an influencer.",
    ),
    StructuredTool(
        name="negotiate_contract",
        func=negotiate_contract_with_ai,
        description="Negotiates with an influencer.",
        args_schema=NegotiateContractInput
    ),
    Tool(
        name="monitor_campaign_performance",
        func=monitor_campaign_performance_simulation,
        description="Monitors campaign performance.",
    ),
    Tool(
        name="generate_report",
        func=generate_report,
        description="Generates a report of campaign performance"
    )
]

tool_executor = ToolExecutor(tools)

# --- Agent Nodes ---
def initialize_campaign(state: CampaignState) -> CampaignState:
    messages = state["messages"]
    log_messages = []  # Collect log messages
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert influencer marketing manager. From the input, extract campaign details:
    - campaign_goal (infer if not explicit, e.g., 'promote language learning' for language learners)
    - target_audience (age, interests, etc.)
    - budget (in dollars)
    - timeline (duration)
    - content_guidelines (default to 'follow brand guidelines' if unspecified)
    - legal_requirements (default to 'standard disclosure' if unspecified)
    - initial_offer (starting offer in dollars, numeric only)
    - max_offer (maximum offer in dollars, numeric only)
    Campaign is for Stimuler. Ask for clarification if needed."""),
            MessagesPlaceholder(variable_name="messages")
        ])

        initial_response = llm.invoke(prompt.invoke({"messages": messages}))
        log_messages.append(f"DEBUG: Initial LLM response: {initial_response.content}")

        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """Extract the following fields into valid JSON:
                campaign_goal: str (infer if not explicit)
                target_audience: dict with age, interests, etc.
                budget: float
                timeline: str
                content_guidelines: str (default to "" if missing)
                legal_requirements: str (default to "" if missing)
                initial_offer: str (numeric value only, no '$')
                max_offer: str (numeric value only, no '$')
                Use empty strings "" for missing text fields unless specified otherwise."""),
            ("user", f"Extract structured data from this text: {initial_response.content}")
        ])
        
        json_response = llm.invoke(extraction_prompt.invoke({}))
        log_messages.append(f"DEBUG: Extracted JSON: {json_response.content}")
        
        content = json_response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        campaign_data = json.loads(content)
        
        # Clean initial_offer and max_offer to remove '$' if present
        initial_offer = str(campaign_data.get("initial_offer") or "0").replace("$", "")
        max_offer = str(campaign_data.get("max_offer") or "0").replace("$", "")
        
        details = CampaignDetails(
            brand_name="Stimuler",
            campaign_goal=campaign_data.get("campaign_goal") or "promote language learning",
            target_audience=campaign_data.get("target_audience") or {"age": "18-35", "interests": "language learners"},
            budget=float(campaign_data.get("budget") or 0),
            timeline=campaign_data.get("timeline") or "",
            content_guidelines=campaign_data.get("content_guidelines") or "follow brand guidelines",
            legal_requirements=campaign_data.get("legal_requirements") or "standard disclosure",
            initial_offer=initial_offer,
            max_offer=max_offer
        )
        
        return {
            "campaign_details": details,
            "influencers_found": [],
            "selected_influencers": [],
            "negotiation_status": {},
            "campaign_performance": {},
            "messages": [*state["messages"], AIMessage(content=f"I've captured the campaign details for {details.brand_name}.")],
            "negotiation_messages": {},
            "log_messages": log_messages  # Add log messages to the state
        }
    except Exception as e:
        log_messages.append(f"Initialization error: {e}")
        default_details = CampaignDetails(
            brand_name="Stimuler",
            campaign_goal="promote language learning",
            target_audience={"age": "18-35", "interests": "language learners"},
            budget=3000.0,
            timeline="3 weeks",
            content_guidelines="follow brand guidelines",
            legal_requirements="standard disclosure",
            initial_offer="500",  # Numeric string
            max_offer="800"       # Numeric string
        )
        return {
            "campaign_details": default_details,
            "influencers_found": [],
            "selected_influencers": [],
            "negotiation_status": {},
            "campaign_performance": {},
            "messages": [*state["messages"], AIMessage(content="Error occurred, using default campaign details.")],
            "negotiation_messages": {},
            "log_messages": log_messages  # Add log messages to the state
        }

def find_influencers(state: CampaignState) -> CampaignState:
    """Finds potential influencers with improved error handling and search flexibility."""
    global influencer_data
    campaign_details = state["campaign_details"]
    log_messages = []  # Collect log messages

    log_messages.append("\n\n" + "="*80)
    log_messages.append(f"🔍 FINDING INFLUENCERS FOR CAMPAIGN: {campaign_details.brand_name}")
    log_messages.append(f"Campaign Goal: {campaign_details.campaign_goal}")
    log_messages.append(f"Target Audience: {campaign_details.target_audience}")
    log_messages.append("="*80)

    # Check if influencer data is loaded
    if not influencer_data:
        log_messages.append("❌ ERROR: No influencer data available. Loading failed or empty file.")
        # Try to reload data
        try:
            log_messages.append("Attempting to reload influencer data...")
            reloaded_data = load_influencer_data()
            if reloaded_data:
                log_messages.append(f"Successfully reloaded {len(reloaded_data)} influencers")
                influencer_data = reloaded_data
            else:
                log_messages.append("Reload attempt failed - no data")
        except Exception as e:
            log_messages.append(f"Error reloading data: {e}")
    else:
        log_messages.append(f"ℹ️ {len(influencer_data)} total influencers available in database")

    # Check for valid campaign details
    if not campaign_details or not hasattr(campaign_details, "target_audience") or not campaign_details.target_audience:
        log_messages.append("❌ ERROR: Cannot find influencers without valid campaign details.")
        # Create dummy target audience as fallback
        target_audience = {"interests": "language learning, education, study"}
        log_messages.append(f"Using fallback target audience: {target_audience}")
    else:
        target_audience = campaign_details.target_audience

    # Get interests (safely)
    interests = target_audience.get("interests", "")
    keywords = interests.split(", ") if interests else ["language", "education", "learning"]

    log_messages.append(f"🔑 Search Keywords: {keywords}")

    # Try searching on multiple platforms to increase chances of matches
    all_influencers = []
    platforms = ["YouTube", "Instagram", "TikTok"]

    for platform in platforms:
        log_messages.append(f"Searching on platform: {platform}")
        try:
            # Call search_influencers function directly
            platform_influencers = search_influencers_from_data(
                query_data=influencer_data,
                platform=platform,
                keywords=keywords,
                target_audience=target_audience
            )
            all_influencers.extend(platform_influencers)
            log_messages.append(f"Found {len(platform_influencers)} influencers on {platform}")
        except Exception as e:
            log_messages.append(f"❌ ERROR finding influencers on {platform}: {e}")

    # Remove duplicates (based on handle)
    unique_handles = set()
    unique_influencers = []
    for inf in all_influencers:
        if inf.handle not in unique_handles:
            unique_handles.add(inf.handle)
            unique_influencers.append(inf)

    log_messages.append(f"✅ Found {len(unique_influencers)} unique potential influencers:")
    for i, inf in enumerate(unique_influencers, 1):
        log_messages.append(f"  {i}. @{inf.handle} ({inf.name}) - {inf.followers} followers, {inf.niche} niche")

    # Fallback: If no influencers found, create some random ones for testing
    if not unique_influencers:
        log_messages.append("⚠️ No influencers found in data. Creating test influencers...")
        from datetime import datetime
        random_seed = int(datetime.now().timestamp())
        import random
        random.seed(random_seed)
        
        # Create 3 test influencers
        for i in range(3):
            # Create a dummy influencer
            from influencer_data_generator import Influencer
            test_inf = Influencer(
                platform=random.choice(["YouTube", "Instagram", "TikTok"]),
                handle=f"test_influencer_{i}",
                name=f"Test Influencer {i}",
                followers=random.randint(10000, 500000),
                engagement_rate=random.uniform(2.0, 10.0),
                audience_demographics={
                    "age_18-24": 0.4,
                    "age_25-34": 0.3,
                    "gender_female": 0.5,
                    "gender_male": 0.5
                },
                content_quality_score=random.uniform(7.0, 9.5),
                authenticity_score=random.uniform(7.0, 9.5),
                past_collaborations=["TestBrand"],
                contact_info="test@example.com",
                video_summaries=[
                    {"title": "How to Learn Languages Fast", "summary": "Tips for language learning"}
                ],
                niche="language learning",
                persona={
                    "personality": "engaging and motivational",
                    "communication_style": "clear and direct",
                    "values": "education and self-improvement",
                    "quirks": "always starts videos with a greeting in different languages",
                    "speaking_style": "articulate with occasional humor",
                    "tone": "encouraging and informative"
                }
            )
            unique_influencers.append(test_inf)
            log_messages.append(f"  Created test influencer: @{test_inf.handle}")

    return {
        "campaign_details": campaign_details,
        "influencers_found": unique_influencers,
        "selected_influencers": state.get("selected_influencers", []),
        "negotiation_status": state.get("negotiation_status", {}),
        "campaign_performance": state.get("campaign_performance", {}),
        "messages": [*state["messages"], 
                    AIMessage(content=f"Found {len(unique_influencers)} influencers.")],
        "negotiation_messages": state.get("negotiation_messages", {}),
        "log_messages": log_messages
    }

def vet_and_select_influencers(state: CampaignState) -> CampaignState:
    """Vets and selects influencers."""
    log_messages = []
    log_messages.append("\n\n" + "="*80)
    log_messages.append("🔍 VETTING AND SELECTING TOP INFLUENCERS")
    log_messages.append("="*80)

    influencers_found = state["influencers_found"]

    if not influencers_found:
        log_messages.append("❌ No influencers found to vet.")
        return {
            "campaign_details": state["campaign_details"],
            "influencers_found": [],
            "selected_influencers": [],
            "negotiation_status": state.get("negotiation_status", {}),
            "campaign_performance": state.get("campaign_performance", {}),
            "messages": [*state["messages"], AIMessage(content="No influencers found to select from.")],
            "negotiation_messages": state.get("negotiation_messages", {}),
            "log_messages": log_messages
        }

    log_messages.append(f"📊 Vetting {len(influencers_found)} influencers...")

    vetted_influencers = []
    for inf in influencers_found:
        log_messages.append(f"  Vetting @{inf.handle}...")
        # Correctly invoke the tool with the expected structure
        tool_input = {"influencer": inf}
        vetted = tool_executor.invoke(ToolInvocation(
            tool="vet_influencer",
            tool_input=tool_input
        ))
        vetted_influencers.append(vetted)
        log_messages.append(f"  - Quality Score: {vetted.content_quality_score}/10")
        log_messages.append(f"  - Authenticity Score: {vetted.authenticity_score}/10")
        log_messages.append(f"  - Engagement Rate: {vetted.engagement_rate}%")

    # Calculate total score for each influencer
    influencers_with_scores = [(inf, inf.content_quality_score + inf.authenticity_score + inf.engagement_rate * 10) 
                              for inf in vetted_influencers]

    # Sort by score and select top 3
    sorted_influencers = sorted(influencers_with_scores, key=lambda x: x[1], reverse=True)
    selected = [inf for inf, score in sorted_influencers[:3]]

    log_messages.append("\n\n🏆 TOP SELECTED INFLUENCERS:")
    for i, inf in enumerate(selected, 1):
        log_messages.append(f"  {i}. @{inf.handle} ({inf.name})")
        log_messages.append(f"     - Followers: {inf.followers}")
        log_messages.append(f"     - Niche: {inf.niche}")
        log_messages.append(f"     - Engagement Rate: {inf.engagement_rate}%")
        log_messages.append(f"     - Content Quality: {inf.content_quality_score}/10")
        log_messages.append(f"     - Audience: {inf.audience_demographics}")
        log_messages.append(f"     - Persona: {inf.persona}")

    return {
        "campaign_details": state["campaign_details"],
        "influencers_found": influencers_found,
        "selected_influencers": selected,
        "negotiation_status": state.get("negotiation_status", {}),
        "campaign_performance": state.get("campaign_performance", {}),
        "messages": [*state["messages"], AIMessage(content=f"Selected {len(selected)} influencers.")],
        "negotiation_messages": state.get("negotiation_messages", {}),
        "log_messages": log_messages
    }


async def manage_negotiations(state: CampaignState) -> CampaignState:
    import streamlit as st

    # Use a dedicated container for negotiations
    with st.expander("Negotiations", expanded=True):
        selected_influencers = state["selected_influencers"]
        campaign_details = state["campaign_details"]

        if not selected_influencers:
            st.warning("❌ No influencers selected for negotiations")
            return {
                "campaign_details": campaign_details,
                "influencers_found": state["influencers_found"],
                "selected_influencers": [],
                "negotiation_status": {},
                "campaign_performance": state.get("campaign_performance", {}),
                "messages": [*state["messages"], AIMessage(content="No influencers to negotiate with.")],
                "negotiation_messages": state.get("negotiation_messages", {})
            }

        st.markdown(f"📝 Starting negotiations with {len(selected_influencers)} influencers")

        # Initialize chat containers in session state
        if "chat_containers" not in st.session_state:
            st.session_state["chat_containers"] = {}
        
        for influencer in selected_influencers:
            if influencer.handle not in st.session_state["chat_containers"]:
                st.session_state["chat_containers"][influencer.handle] = st.container()

        negotiation_status = {}
        for i, influencer in enumerate(selected_influencers, 1):
            st.markdown(f"👤 NEGOTIATION {i}/{len(selected_influencers)}: @{influencer.handle}")
            chat_container = st.session_state["chat_containers"][influencer.handle]
            status = await negotiate_contract_with_ai(
                influencer=influencer,
                campaign_details=campaign_details,
                agent_llm=llm,
                state=state,
                chat_container=chat_container
            )
            negotiation_status[influencer.handle] = status

        st.markdown("\n\n📊 NEGOTIATION SUMMARY:")
        for handle, status in negotiation_status.items():
            st.markdown(f"  @{handle}: {status}")

        agreements = sum(1 for status in negotiation_status.values() if status == "Agreement reached")
        st.markdown(f"✅ Agreements reached: {agreements}/{len(negotiation_status)}")

    return {
        "campaign_details": campaign_details,
        "influencers_found": state["influencers_found"],
        "selected_influencers": selected_influencers,
        "negotiation_status": negotiation_status,
        "campaign_performance": state.get("campaign_performance", {}),
        "messages": [*state["messages"], AIMessage(content=f"Negotiation status: {negotiation_status}")],
        "negotiation_messages": state["negotiation_messages"]
    }

def monitor_and_report(state: CampaignState) -> CampaignState:
    """Monitors and reports."""
    log_messages = []
    log_messages.append("\n\n" + "="*80)
    log_messages.append("📈 CAMPAIGN MONITORING AND REPORTING")
    log_messages.append("="*80)

    campaign_details = state["campaign_details"]
    selected_influencers = state["selected_influencers"]
    negotiation_status = state.get("negotiation_status", {})

    if not negotiation_status:
        log_messages.append("❌ No contracts to monitor - no negotiations were conducted")
        return {
            "campaign_details": state.get("campaign_details"),
            "influencers_found": state.get("influencers_found"),
            "selected_influencers": state.get("selected_influencers"),
            "negotiation_status": state.get("negotiation_status"),
            "campaign_performance": state.get("campaign_performance"),
            "messages": [*state["messages"], AIMessage(content="No contracts to monitor.")],
            "negotiation_messages": state.get("negotiation_messages"),
            "log_messages": log_messages
        }

    # Check if all negotiations were successful
    successful_negotiations = [handle for handle, status in negotiation_status.items() 
                              if status == "Agreement reached"]

    if not all(status == "Agreement reached" for status in negotiation_status.values()):
        log_messages.append(f"⚠️ Not all negotiations were successful. Proceeding with {len(successful_negotiations)} agreements.")
        # Filter selected_influencers to only include those with successful negotiations
        selected_influencers = [inf for inf in selected_influencers if inf.handle in successful_negotiations]

    if not successful_negotiations:
        log_messages.append("❌ No successful negotiations to monitor")
        return {
            "campaign_details": state.get("campaign_details"),
            "influencers_found": state.get("influencers_found"),
            "selected_influencers": state.get("selected_influencers"),
            "negotiation_status": state.get("negotiation_status"),
            "campaign_performance": state.get("campaign_performance"),
            "messages": [*state["messages"], AIMessage(content="No successful negotiations to monitor.")],
            "negotiation_messages": state.get("negotiation_messages"),
            "log_messages": log_messages
        }

    log_messages.append(f"🚀 Monitoring campaign with {len(successful_negotiations)} influencers:")
    for inf in selected_influencers:
        if inf.handle in successful_negotiations:
            log_messages.append(f"  @{inf.handle} ({inf.name}) - {inf.followers} followers")

    log_messages.append("\n\n📊 Running performance simulation...")
    performance = tool_executor.invoke({
        "tool": "monitor_campaign_performance",
        "tool_input": {"campaign_details": campaign_details, "selected_influencers": selected_influencers}
    })

    log_messages.append("\n\n📝 Generating campaign report...")
    report = tool_executor.invoke({"tool":"generate_report", "tool_input":{"campaign_performance": performance}})

    log_messages.append("\n\n" + "="*80)
    log_messages.append("📑 FINAL CAMPAIGN REPORT")
    log_messages.append("="*80)
    log_messages.append(report)
    log_messages.append("="*80)

    return {
        "campaign_details": campaign_details,
        "influencers_found": state["influencers_found"],
        "selected_influencers": selected_influencers,
        "negotiation_status": state.get("negotiation_status", {}),
        "campaign_performance": performance,  # Update campaign_performance
        "messages": [*state["messages"], AIMessage(content=f"Report:\n\n{report}")],
        "negotiation_messages": state.get("negotiation_messages", {}),
        "log_messages": log_messages
    }

def route_based_on_negotiation(state: CampaignState) -> str:
    """Routes based on negotiation."""
    if not state.get("negotiation_status"):
        return "end"
    return "monitor" if all(status == "Agreement reached" for status in state["negotiation_status"].values()) else "end"

# Assuming this is part of your existing code
workflow = StateGraph(CampaignState)
workflow.add_node("initialize_campaign", initialize_campaign)
workflow.add_node("find_influencers", find_influencers)
workflow.add_node("vet_and_select", vet_and_select_influencers)
workflow.add_node("negotiate", manage_negotiations)  # No change needed here
workflow.add_node("monitor", monitor_and_report)
workflow.add_edge("initialize_campaign", "find_influencers")
workflow.add_edge("find_influencers", "vet_and_select")
workflow.add_edge("vet_and_select", "negotiate")
workflow.add_conditional_edges("negotiate", route_based_on_negotiation, {"monitor": "monitor", "end": END})
workflow.add_edge("monitor", END)
workflow.set_entry_point("initialize_campaign")
graph = workflow.compile()

# Custom CSS for a modern look
st.markdown(
    """
    <style>
        .reportview-container {
            background: #f0f2f6;
        }
       .main .block-container {
            max-width: 90%;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stTextInput > label {
            color: #313360;
            font-weight: bold;
        }
        .stButton > button {
            background-color: #313360;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #5a5c8f;
        }
        .css-10trblm {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .css-10trblm h2 {
            color: #313360;
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }
        .css-10trblm p {
            color: #555555;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚀 Influencer Marketing Campaign Agent")
st.subheader("Automate your influencer marketing with AI")

user_input = st.text_area("Enter Campaign Details:", 
                            "Campaign for Stimuler. Target: 18-35, language learners. Budget: $8000, 3 weeks. Offer: $2000-$4000.",
                            height=150)

if st.button("Start Campaign", use_container_width=True):
    if not user_input:
        st.error("Please enter campaign details to start.")
    else:
        # Initialize session state for logs
        if 'negotiation_logs' not in st.session_state:
            st.session_state['negotiation_logs'] = {}

        # Create a container for the workflow steps
        workflow_container = st.container()
        
        # Create a container for the negotiation logs
        negotiation_logs_container = st.container()

        # --- run_agent_streamlit ---
        async def run_agent_streamlit(user_input: str):
            import streamlit as st

            try:
                state = {"messages": [HumanMessage(content=user_input)]}

                with workflow_container:
                    st.subheader("📋 Workflow Progress")
                    with st.expander("Steps", expanded=True):
                        steps = ["initialize_campaign", "find_influencers", "vet_and_select", "negotiate", "monitor"]
                        for i, step in enumerate(steps, 1):
                            st.markdown(f"  {i}. {step}")
                        st.markdown("-"*80)

                    current_node = None
                    async for event in graph.astream_log(state, config={"recursion_limit": 100}):
                        evt_type = getattr(event, "event", None)
                        
                        if evt_type == "on_node_start":
                            node_name = getattr(event, "data", {}).get("node")
                            if node_name != current_node:
                                current_node = node_name
                                st.info(f"🔶 STARTING WORKFLOW NODE: {node_name}")
                        
                        elif evt_type == "on_node_end":
                            node_name = getattr(event, "data", {}).get("node")
                            st.success(f"✅ COMPLETED WORKFLOW NODE: {node_name}")
                            if "log_messages" in state:
                                with st.expander(f"Logs for {node_name}", expanded=False):
                                    for log_message in state["log_messages"]:
                                        st.markdown(log_message)
                                del state["log_messages"]
                        
                        elif evt_type == "on_chat_model_stream":
                            data = getattr(event, "data", None)
                            content = data.get("chunk", {}).get("content", "") if isinstance(data, dict) else str(data)
                            if content:
                                st.markdown(content)
                        
                        elif evt_type == "on_tool_start":
                            tool_name = getattr(event, "data", {}).get("name", "unknown")
                            st.info(f"🔧 STARTING TOOL: {tool_name}")
                        
                        elif evt_type == "on_tool_end":
                            st.success(f"✅ FINISHED TOOL EXECUTION")
                        
                        elif evt_type == "on_chat_model_end":
                            st.markdown("\n\n")

                        # Update state
                        if hasattr(event, "data") and "state" in event.data:
                            state = event.data["state"]

            except Exception as e:
                st.error(f"\n\n❌ ERROR: Agent execution failed: {e}")
                import traceback
                st.write(traceback.format_exc())

        async def main_streamlit():
            import streamlit as st
            import os
            import asyncio
            import time

            st.write("\n\n" + "="*80)
            st.write("🚀 STARTING INFLUENCER MARKETING AGENT")
            st.write("="*80)

            data_file = "influencer_data_stimuler.json"
            if not os.path.exists(data_file):
                st.warning(f"⚠️ WARNING: Influencer data file '{data_file}' not found!")
                uploaded_file = st.file_uploader("Upload an influencer data JSON file", type="json")
                if uploaded_file:
                    with open(data_file, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success("Data file uploaded successfully!")

            loaded_data = load_influencer_data()
            st.write(f"Loaded {len(loaded_data)} influencers for campaign")

            st.write(f"💬 USER INPUT: {user_input}")
            st.write("="*80)

            global workflow_container
            workflow_container = st.container()

            try:
                await run_agent_streamlit(user_input)
                st.write("\n\n" + "="*80)
                st.success("✅ CAMPAIGN PROCESS COMPLETED")
                st.write("="*80)

                st.write("Cleaning up resources...")
                pending = asyncio.all_tasks() - {asyncio.current_task()}
                if pending:
                    st.write(f"Waiting for {len(pending)} pending tasks...")
                    await asyncio.gather(*pending, return_exceptions=True)
                time.sleep(2)

            except Exception as e:
                st.error(f"\n\n❌ ERROR: Execution failed: {e}")
                import traceback
                st.write(traceback.format_exc())

            finally:
                st.write("Final cleanup...")
                time.sleep(1)

        # Run the main function
        asyncio.run(main_streamlit())