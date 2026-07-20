import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Strict validation model (temperature=0)
llm_strict = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0)

# Creative copywriting model (temperature=0.7)
llm_creative = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.7)

def _parse_content(content) -> str:
    """Safe parser to extract flat strings from LangChain content block arrays."""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


#campaign coordinator & router
class SupervisorRouting(BaseModel):
    event_id: str = Field(
        description="The event slug (e.g., 'nextjs_bootcamp') matched against valid database IDs."
    )
    target_contents: list[Literal["email", "newsletter", "social"]] = Field(
        description="The target campaign channels to trigger."
    )

async def run_supervisor(user_prompt: str, valid_event_ids: list[str]) -> SupervisorRouting:
    """Orchestrates campaign layout requests by extracting the target event ID and content channels."""
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


#copywriting agents (email, newsletter, social)
async def run_email_writer(event_id: str, campaign_facts: str, layout: str, feedback: str = "") -> str:
    """Drafts promotional email copy tailored dynamically to student or professional audience profiles."""
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
    """Drafts monthly newsletter digests using highlights, emerging trends, and trainer spotlight sections."""
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
    """Drafts social media updates matching hooks and hashtags to target developer demographics."""
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


#verification checkers
class CheckerResult(BaseModel):
    status: Literal["PASS", "FAIL"] = Field(
        description="Verification outcome status."
    )
    feedback: str = Field(
        description="Bulleted feedback logs describing required edits (empty if PASS)."
    )

async def run_fact_checker(content_type: str, draft: str, campaign_facts: str, event_metadata: dict) -> CheckerResult:
    """Cross-checks generated draft details against metadata parameters and syllabus facts to prevent errors."""
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
    """Validates structural layout alignments and brand tone guidelines compliance."""
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


#holistic campaign review
async def run_final_reviewer(email_draft: str = "", newsletter_draft: str = "", social_draft: str = "", few_shot_examples: str = "") -> CheckerResult:
    """Holistic review of all content assets to guarantee overall messaging cohesion and match approved standards."""
    structured_llm = llm_strict.with_structured_output(CheckerResult)
    
    prompt = f"""
    You are the Lead Content Director at HiDevs. Review the entire generated campaign bundle.
    Ensure that the messaging is cohesive, unified, and matches the style of our previously approved high-converting marketing drafts.
    
    Approved HiDevs Copy Examples (for style and tone matching reference):
    ---
    {few_shot_examples}
    ---
    
    NOTE: If the Approved Copy Examples section above is empty, proceed with the holistic review by checking for natural tone, clarity, and overall messaging cohesion without reference-matching.
    
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