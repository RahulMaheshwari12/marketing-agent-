import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Model for reasoning and validation (highly objective, temperature=0)
llm_strict = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

# Model for copywriting and creative drafts (engaging marketing tone, temperature=0.7)
llm_creative = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.7)

def _parse_content(content) -> str:
    """Helper to ensure we extract a clean string from both text blocks and content array lists."""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


# =====================================================================
# 1. Supervisor Coordinator Agent
# =====================================================================

class SupervisorRouting(BaseModel):
    event_id: str = Field(
        description="The event slug (e.g., 'nextjs_bootcamp', 'test_hackathon') extracted from the user's request."
    )
    target_contents: list[Literal["email", "newsletter", "social"]] = Field(
        description="The list of content formats the user wants to generate. Can contain 'email', 'newsletter', and/or 'social'."
    )

async def run_supervisor(user_prompt: str, valid_event_ids: list[str]) -> SupervisorRouting:
    """Parses user input to extract the event_id and target content channels, matching against valid database IDs."""
    structured_llm = llm_strict.with_structured_output(SupervisorRouting)
    
    prompt = f"""
    You are the Lead Marketing Coordinator at HiDevs. Analyze the user's request and determine:
    1. Which event_id they are targeting. You MUST select the ID from the list of valid database IDs below. Find the closest match.
    2. Which content formats (email, newsletter, social) they want to generate.
    
    List of Valid Database Event IDs:
    ---
    {valid_event_ids}
    ---
    
    User Request:
    "{user_prompt}"
    """
    
    return await structured_llm.ainvoke(prompt)


# =====================================================================
# 2. Copywriter Agents (Email, Newsletter, Social)
# =====================================================================

async def run_email_writer(event_id: str, campaign_facts: str, layout: str, feedback: str = "") -> str:
    """Generates or refines promotional email copy, auto-adapting the tone to Students or Professionals."""
    prompt = f"""
    You are a professional Email Marketing Copywriter at HiDevs.
    
    TASK:
    1. Read the verified course facts below and analyze the target audience. Determine if it targets:
       - STUDENTS (beginner-friendly, low-price, career-starting, project-focused).
       - PROFESSIONALS (advanced topics, production-scaling, corporate, weekend/night format).
       - GENERAL/HYBRID (broad audience, beginner-to-advanced).
    2. Write an engaging, high-converting promotional email for '{event_id}' tailored specifically to that audience's motivations and tone of voice.
    
    Follow this required structural layout:
    ---
    {layout}
    ---
    
    Use these verified course facts:
    ---
    {campaign_facts}
    ---
    """
    if feedback:
        prompt += f"\nAn internal reviewer flagged issues in your previous draft. Rewrite the email to address this feedback:\nFEEDBACK: {feedback}"
        
    response = await llm_creative.ainvoke(prompt)
    return _parse_content(response.content)


async def run_newsletter_writer(
    event_id: str, 
    campaign_facts: str, 
    trainer_bio: str, 
    ai_house_highlights: str, 
    emerging_tech_trends: str, 
    layout: str, 
    feedback: str = ""
) -> str:
    """Generates an end-of-month review newsletter highlighting HiDevs event wrap-ups, trainer spotlight, and emerging industry tech trends in a unified community tone."""
    prompt = f"""
    You are a professional Newsletter Content Editor at HiDevs.
    
    TASK:
    1. Write an engaging, high-value end-of-month community newsletter segment for '{event_id}'.
    2. Maintain a unified community tone that appeals to BOTH professionals (high tech value, industry standards) and students/career transitioners (accessibility, learning, networking) simultaneously.
    3. Include three core sections:
       - AI House Wrap-up: Summarize what happened in our AI house this month (workshops, hackathons, student progress) using the provided Highlights.
       - Emerging Tech Trends: Detail the new technologies, releases, or trends emerging in this field.
       - Instructor Spotlight: Introduce the featured trainer who led our key event.
    
    Required Layout Structure:
    ---
    {layout}
    ---
    
    Target Campaign Facts:
    ---
    {campaign_facts}
    ---
    
    What Happened in our AI House (HiDevs) This Month:
    ---
    {ai_house_highlights}
    ---
    
    Emerging Technologies & Trends:
    ---
    {emerging_tech_trends}
    ---
    
    Instructor Spotlight Bio:
    ---
    {trainer_bio}
    ---
    """
    if feedback:
        prompt += f"\nAn internal reviewer flagged issues in your previous draft. Rewrite the newsletter to address this feedback:\nFEEDBACK: {feedback}"
        
    response = await llm_creative.ainvoke(prompt)
    return _parse_content(response.content)


async def run_social_writer(event_id: str, campaign_facts: str, layout: str, feedback: str = "") -> str:
    """Generates or refines social media posts (LinkedIn/Twitter), auto-adapting tone to Students or Professionals."""
    prompt = f"""
    You are a Social Media Manager at HiDevs.
    
    TASK:
    1. Read the verified course facts below and determine the target audience:
       - STUDENTS (use energetic tone, focus on hackathons/projects, career starters).
       - PROFESSIONALS (use authoritative/insightful tone, focus on scalability, tech stack, ROI).
       - GENERAL/HYBRID (balanced, engaging social hooks).
    2. Write an engaging social media post for '{event_id}' optimized for readability and click-throughs tailored to that audience.
    
    Follow this formatting style:
    ---
    {layout}
    ---
    
    Use these course details:
    ---
    {campaign_facts}
    ---
    """
    if feedback:
        prompt += f"\nAn internal reviewer flagged issues in your previous draft. Rewrite the post to address this feedback:\nFEEDBACK: {feedback}"
        
    response = await llm_creative.ainvoke(prompt)
    return _parse_content(response.content)


# =====================================================================
# 3. Verification Checker Agents (Fact-Checker, Style-Checker)
# =====================================================================

class CheckerResult(BaseModel):
    status: Literal["PASS", "FAIL"] = Field(
        description="Output 'PASS' if the content is 100% correct, or 'FAIL' if any issues are found."
    )
    feedback: str = Field(
        description="If FAIL, provide a clear, bulleted list of issues that the copywriter must correct. If PASS, return an empty string."
    )

async def run_fact_checker(content_type: str, draft: str, campaign_facts: str, event_metadata: dict) -> CheckerResult:
    """Compares the draft copy against raw database facts to check for price, links, dates, and syllabus errors."""
    structured_llm = llm_strict.with_structured_output(CheckerResult)
    
    prompt = f"""
    You are a meticulous Fact-Checker at HiDevs. Your job is to verify that the generated '{content_type}' copy has no factual mistakes.
    
    Compare the draft below against the source files. 
    Flag an error if:
    - The price or registration URL does not match the official metadata.
    - The syllabus topics or dates do not match the facts.
    - The draft contains placeholder text (e.g. '[Insert Link]').
    
    Official Event Metadata:
    ---
    {event_metadata}
    ---
    
    Official Syllabus/Brochure Facts:
    ---
    {campaign_facts}
    ---
    
    Generated '{content_type}' Draft:
    ---
    {draft}
    ---
    """
    return await structured_llm.ainvoke(prompt)


async def run_style_checker(content_type: str, draft: str, brand_guidelines: str, layout_template: str) -> CheckerResult:
    """Verifies that the draft adheres to the brand tone rules and follows the structural layout template."""
    structured_llm = llm_strict.with_structured_output(CheckerResult)
    
    prompt = f"""
    You are a strict Branding and Style Checker at HiDevs. Your job is to verify that the generated '{content_type}' copy matches corporate standards.
    
    Verify that:
    - The layout structure matches the required template.
    - The copy adheres to the brand tone guidelines.
    
    Brand Style & Tone Guidelines:
    ---
    {brand_guidelines}
    ---
    
    Required Formatting Layout:
    ---
    {layout_template}
    ---
    
    Generated '{content_type}' Draft:
    ---
    {draft}
    ---
    """
    return await structured_llm.ainvoke(prompt)


# =====================================================================
# 4. Final Content Director Reviewer Agent
# =====================================================================

async def run_final_reviewer(email_draft: str = "", newsletter_draft: str = "", social_draft: str = "") -> CheckerResult:
    """Performs a holistic review of all completed drafts to ensure consistent tone across the campaign."""
    structured_llm = llm_strict.with_structured_output(CheckerResult)
    
    prompt = f"""
    You are the Lead Content Director at HiDevs. Review the entire generated campaign bundle.
    Ensure that the messaging is cohesive and unified across all channels.
    
    Email Draft:
    ---
    {email_draft}
    ---
    
    Newsletter Draft:
    ---
    {newsletter_draft}
    ---
    
    Social Draft:
    ---
    {social_draft}
    ---
    """
    return await structured_llm.ainvoke(prompt)