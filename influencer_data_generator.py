# influencer_data_generator.py
import os
import json
import random
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()

class Influencer(BaseModel):
    platform: str = Field(description="The social media platform (e.g., Instagram, TikTok, YouTube).")
    handle: str = Field(description="The influencer's username or handle.")
    name: str = Field(..., description="The influencer's name or display name.")
    followers: int = Field(description="The number of followers.")
    engagement_rate: float = Field(description="The engagement rate (e.g., likes, comments, shares).")
    audience_demographics: Dict[str, float] = Field(description="A dictionary of audience demographics (e.g., {'age_18-24': 0.3, 'gender_female': 0.6}).")
    content_quality_score: float = Field(description="A score (0-10) representing the quality of the influencer's content.")
    authenticity_score: float = Field(description="A score (0-10) representing the authenticity of the influencer's following and engagement.")
    past_collaborations: List[str] = Field(description="A list of brands the influencer has collaborated with in the past.")
    contact_info: str = Field(description="The influencer's contact information (email, etc.).")
    video_summaries: List[Dict[str, str]] = Field(description="A list of dictionaries, each containing a video title and a short summary.")
    niche: str = Field(description="The primary niche or category of the influencer (e.g., 'fitness', 'beauty', 'gaming').")
    persona: Dict[str, str] = Field(description="A dictionary containing details about the influencer's persona.")


# --- LLM Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"), convert_system_message_to_human=True)

# --- Stimuler Information ---
STIMULER_INFO = {
    "name": "Stimuler",
    "description": "An AI-powered English language learning app that helps users improve their speaking skills, particularly for IELTS and TOEFL exams.",
    "features": [
        "AI-powered speech analysis and feedback",
        "Real-time IELTS simulation with an AI interviewer",
        "Diverse speaking topics for practice",
        "Personalized tips and insights",
        "Global recognition as one of the top AI apps for language learning",
    ],
    "target_audience": "Individuals preparing for IELTS and TOEFL exams, and anyone looking to improve their English speaking skills."
}

def generate_influencer_data(num_influencers: int = 20) -> List[Influencer]:
    """Generates synthetic data for influencers, tailored for Stimuler, including persona."""

    influencers = []
    for _ in range(num_influencers):
        # --- Focus on Relevant Niches and Platforms ---
        platform = random.choice(["YouTube", "Instagram", "TikTok"])
        niches = ["education", "languagelearning", "studyabroad", "ielts", "toefl", "travel", "lifestyle", "productivity"]
        niche = random.choice(niches)
        handle = f"{niche}_{random.randint(100, 999)}"
        name = f"Generated {niche.capitalize()} Influencer {random.randint(1, 100)}"
        followers = random.randint(5000, 500000)
        engagement_rate = round(random.uniform(2.0, 12.0), 2)

        # --- Generate Audience Demographics (Targeted) ---
        demographics = {
            "age_18-24": round(random.uniform(0.3, 0.7), 2),
            "age_25-34": round(random.uniform(0.2, 0.5), 2),
            "age_35-44": round(random.uniform(0.0, 0.2), 2),
            "gender_female": round(random.uniform(0.4, 0.7), 2),
            "gender_male": round(random.uniform(0.3, 0.6), 2),
        }
        total = sum(demographics.values())
        demographics = {k: round(v / total, 2) for k, v in demographics.items()}

        # --- Generate Scores ---
        content_quality_score = round(random.uniform(7.0, 10.0), 1)
        authenticity_score = round(random.uniform(7.0, 10.0), 1)

        # --- Generate Past Collaborations (Relevant) ---
        num_collaborations = random.randint(0, 3)
        collaboration_options = ["LanguageApp", "StudyMaterials", "OnlineCourse", "TestPrep", "EdTech"]
        past_collaborations = [f"{random.choice(collaboration_options)}{random.randint(1,10)}" for _ in range(num_collaborations)]

        # --- Generate Contact Info ---
        contact_info = f"{handle}@example.com"

        # --- Generate Video Summaries (Tailored to Stimuler) ---
        video_summaries = generate_video_summaries(platform, niche, STIMULER_INFO)

        # --- Generate Influencer Persona ---
        persona = generate_influencer_persona(niche, platform)

        influencer = Influencer(
            platform=platform,
            handle=handle,
            name=name,
            followers=followers,
            engagement_rate=engagement_rate,
            audience_demographics=demographics,
            content_quality_score=content_quality_score,
            authenticity_score=authenticity_score,
            past_collaborations=past_collaborations,
            contact_info=contact_info,
            video_summaries=video_summaries,
            niche=niche,
            persona=persona
        )
        influencers.append(influencer)

    return influencers


def generate_video_summaries(platform: str, niche: str, stimuler_info: Dict) -> List[Dict[str, str]]:
    """Generates video summaries, incorporating Stimuler information."""
    
    num_videos = random.randint(10, 20)

    # Create a non-templated prompt string
    system_prompt = f"""You are a helpful assistant that generates realistic video titles and short summaries for social media influencers.
The summaries should be concise (1-2 sentences) and engaging. The titles should be catchy.
Generate {num_videos} video titles and summaries. The platform is {platform} and the influencer's niche is {niche}.
Consider that these influencers might be promoting an app like **{stimuler_info['name']}**, which is {stimuler_info['description']}.
Its key features include: {', '.join(stimuler_info['features'])}. The target audience is: {stimuler_info['target_audience']}.
Some video ideas could relate to how the influencer uses the app, tips for using it, or reviews of its features.
Return the output as a JSON list, where each element is a dictionary with keys "title" and "summary".
"""

    # Use direct messages instead of a template
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Generate video summaries in JSON format like [{'title': 'Title', 'summary': 'Summary'}, ...]"}
    ])

    try:
        # Extract JSON from response if needed
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        video_data = json.loads(content)
        
        if isinstance(video_data, list) and all(isinstance(item, dict) and "title" in item and "summary" in item for item in video_data):
            filtered_videos = [
                video for video in video_data
                if 5 <= len(video['title']) <= 80 and 10 <= len(video['summary']) <= 200
            ]
            return filtered_videos[:num_videos]
        else:
            print(f"Warning: LLM response was not in the expected JSON list format: {content}")
            return []
    except json.JSONDecodeError:
        print(f"Error: LLM response was not valid JSON: {response.content}")
        return []

def generate_influencer_persona(niche: str, platform: str) -> Dict[str, str]:
    """Generates a persona for the influencer using the LLM, with fallback logic."""

    system_prompt = f"""You are a creative assistant that generates detailed personas for social media influencers.
Describe the influencer's personality, communication style, values, and unique traits.
The influencer's niche is '{niche}' and their primary platform is '{platform}'.
Return the persona as a JSON object with these exact keys: personality, communication_style, values, quirks, speaking_style, and tone.
"""

    # Use direct messages instead of a template
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate influencer persona in valid JSON format."}
            ])
            
            content = response.content
            # Extract JSON if it's wrapped in code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            try:
                persona = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Attempt {attempt + 1}: JSON decoding failed: {content}. Error: {e}")
                continue  # Skip to the next attempt

            # Validate the persona
            required_keys = ["personality", "communication_style", "values", "quirks", "speaking_style", "tone"]
            if (isinstance(persona, dict) and
                all(key in persona for key in required_keys) and
                all(isinstance(persona[key], str) and persona[key] for key in required_keys)):
                return persona
            else:
                print(f"Attempt {attempt + 1}: Invalid persona format or missing fields: {content}")

        except Exception as e:
            print(f"Attempt {attempt + 1}: Unexpected error: {str(e)}")

    # Fallback persona if LLM fails after retries
    print(f"Warning: Failed to generate persona for niche '{niche}' on platform '{platform}'. Using fallback.")
    return {
        "personality": "friendly and approachable",
        "communication_style": "casual and relatable",
        "values": "education and personal growth",
        "quirks": "often shares fun facts",
        "speaking_style": "clear and enthusiastic",
        "tone": "motivational"
    }

def main():
    influencers = generate_influencer_data()

    # Save to a JSON file:
    with open("influencer_data_stimuler.json", "w") as f:
        json.dump([influencer.dict() for influencer in influencers], f, indent=2)
    print("Influencer data saved to influencer_data_stimuler.json")

if __name__ == "__main__":
    main()
