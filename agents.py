import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Model for reasoning and validation (highly objective)
llm_strict = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Model for copywriting and creative drafts (engaging marketing tone)
llm_creative = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

class SupervisorRouting(BaseModel):
    event_id: str = Field(description= "The event slug (e.g., 'nextjs_bootcamp', 'test_hackathon') extracted from the user's request.")
    target_content: list[Literal["Mail", "Newsletter", "social"]] = Field( description="The list of content formats the " \
    "user wants to generate. Can contain 'email', 'newsletter', and/or 'social'."
    )

async def run_supervisor(user_prompt: str) -> SupervisorRouting:
    """it parse users input to extract the event_id and target"""