import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
import firebase_admin
from firebase_admin import credentials, firestore
from sqlmodel import SQLModel, Field
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import Optional
from datetime import datetime

# Load environment variables
load_dotenv()

# =====================================================================
# 1. Asynchronous Qdrant (Vector DB) Client
# =====================================================================
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

# Initialize ASYNC Qdrant Client
async_qdrant_client = AsyncQdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key if qdrant_api_key else None
)


# =====================================================================
# 2. Asynchronous Firebase Firestore Client
# =====================================================================
firebase_cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "firebase_key.json")

# Initialize Firebase App
if not firebase_admin._apps:
    if os.path.exists(firebase_cred_path):
        cred = credentials.Certificate(firebase_cred_path)
        firebase_admin.initialize_app(cred)
    else:
        print(f"⚠️ Warning: Firebase credentials file not found at '{firebase_cred_path}'.")

# Initialize ASYNC Firestore Client
async_firestore_db = None
if firebase_admin._apps:
    # Initialize from the service account JSON key file
    async_firestore_db = firestore.AsyncClient.from_service_account_json(firebase_cred_path)


# =====================================================================
# 3. Asynchronous SQLModel / PostgreSQL (Campaign DB)
# =====================================================================
# We use PostgreSQL with asyncpg driver (Production Standard)
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/campaigns")

# Create Async Engine
async_engine = create_async_engine(DATABASE_URL, echo=False)

# Session factory for AsyncSession
async_session_maker = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

# Define the Campaign model (All draft fields are Optional for selective generation)
class Campaign(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: str = Field(index=True)  # Acts as the folder ID (e.g. bootcamp_july_2026)
    campaign_name: str
    newsletter_draft: Optional[str] = Field(default=None)
    email_draft_subject: Optional[str] = Field(default=None)
    email_draft_body: Optional[str] = Field(default=None)
    social_draft: Optional[str] = Field(default=None)
    status: str = Field(default="pending_review")  # pending_review, approved, rejected
    email_status: str = Field(default="not_started")  # not_started, success, failed
    social_status: str = Field(default="not_started")  # not_started, success, failed
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Helper function to create tables asynchronously
async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

# Dependency to get Async database sessions
async def get_async_db():
    async with async_session_maker() as session:
        yield session


# Quick self-test script
if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("Initializing async database tables...")
        await init_db()
        print("Async database tables created successfully!")
        
    asyncio.run(main())
