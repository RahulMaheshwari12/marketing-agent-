#importing necessary libraries
import os
import httpx
from google.cloud import firestore
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Import clients from database.py
from database import qdrant_client, async_firestore_db

#loading collection name from enviorment 
COLLECTION_NAME = os.getenv("Collection_name", "Hidevs_knowledge_base").strip()

#initializing the embedding model
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

#getting metadata from firestore for event_id 
async def get_event_metadata_base(event_id: str) -> dict:
    """Fetches the static metadata (URLs, pricing, dates, hashtags) from Firestore."""
    if not async_firestore_db:
        raise RuntimeError("Firebase Firestore client is not initialized.")
    doc_ref = async_firestore_db.collection("events").document(event_id)
    doc = await doc_ref.get()
    if not doc.exists:
        raise ValueError(f"Event metadata not found in Firebase for ID: {event_id}")
    return doc.to_dict()

#retrieving knowledge base from qdrant for event_id and category with semantic search
async def retrieve_knowledge_base_base(event_id: str, category: str, query: str, limit: int = 3) -> list[str]:
    """Semantic search inside Qdrant filtered strictly by event_id and data category."""
    qdrant_filter = Filter(
        must=[
            FieldCondition(key="metadata.event_id", match=MatchValue(value=event_id)),
            FieldCondition(key="metadata.category", match=MatchValue(value=category))
        ]
    )
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings_model
    )
    docs = await vector_store.asimilarity_search(query, k=limit, filter=qdrant_filter)
    return [doc.page_content for doc in docs]

#saving a copywriter's draft directly into the Firestore campaign document
async def save_campaign_draft_base(event_id: str, content_type: str, draft_text: str):
    """Saves or merges a copywriter's draft directly into the Firestore campaign document."""
    if not async_firestore_db:
        raise RuntimeError("Firebase Firestore client is not initialized.")
    doc_ref = async_firestore_db.collection("campaigns").document(event_id)
    await doc_ref.set({
        f"{content_type}_draft": draft_text,
        "last_updated": firestore.SERVER_TIMESTAMP
    }, merge=True)

#getting the current campaign package from firestore for a given event_id
async def get_campaign_package_base(event_id: str) -> dict:
    """Reads all drafts and current status of a campaign from Firestore."""
    if not async_firestore_db:
        raise RuntimeError("Firebase Firestore client is not initialized.")
    doc_ref = async_firestore_db.collection("campaigns").document(event_id)
    doc = await doc_ref.get()
    if not doc.exists:
        return {}
    return doc.to_dict()

#updating the status of a campaign in firestore for a given event_id
async def update_campaign_status_base(event_id: str, status: str):
    """Updates the state status (e.g. drafting, review, approved) of a campaign."""
    if not async_firestore_db:
        raise RuntimeError("Firebase Firestore client is not initialized.")
    doc_ref = async_firestore_db.collection("campaigns").document(event_id)
    await doc_ref.set({
        "status": status,
        "last_updated": firestore.SERVER_TIMESTAMP
    }, merge=True)

#getting the layout template for a given event_id and content_type, supporting campaign-specific dynamic custom templates
async def get_layout_template_base(event_id: str, content_type: str) -> str:
    """Retrieves formatting blueprints, supporting campaign-specific dynamic custom templates."""
    campaign_data = await get_campaign_package_base(event_id)
    custom_layout = campaign_data.get(f"{content_type}_custom_layout")
    if custom_layout:
        return custom_layout
    defaults = {
        "email": "Subject Line:\n[Catchy Hook]\n\nBody:\n[Problem & Value Proposition]\n\nCTA:\n[Registration Link]\n\nSign-off:\n[Signature]",
        "newsletter": "Header:\n[Catchy Title]\n\nFeatured Section:\n[Hook & Main Course Highlights]\n\nCurriculum Outline:\n[Details & Key Topics]\n\nInstructor Spotlight:\n[Trainer Biography]\n\nCTA:\n[Registration Link & Pricing]",
        "social": "[Hook Statement]\n\n[Key Highlights or Bullet Points]\n\n[Registration Link]\n\n[Hashtags]"
    }
    return defaults.get(content_type, "")

#getting the list of all registered event IDs from firestore
async def get_all_active_events_base() -> list[str]:
    """Retrieves a list of all event document IDs currently registered in Firestore."""
    if not async_firestore_db:
        raise RuntimeError("Firebase Firestore client is not initialized.")
    return [doc.id async for doc in async_firestore_db.collection("events").list_documents()]

#tool for retrieving event metadata from firestore for a given event_id
@tool
async def get_event_metadata_tool(event_id: str) -> dict:
    """Retrieves the official registration URL, pricing, dates, and recommended hashtags for a specific event."""
    return await get_event_metadata_base(event_id)

#tool for retrieving campaign knowledge base from qdrant for a given event_id and query
@tool
async def search_campaign_syllabus_tool(event_id: str, query: str) -> list[str]:
    """Search for curriculum highlights, syllabus topics, and academic facts about the bootcamp.
    Use this to retrieve exact course details for writing newsletters, emails, or posts.
    """
    return await retrieve_knowledge_base_base(event_id, "campaign", query)

#tool for retrieving professional knowledge base from qdrant for a given event_id and query
@tool
async def search_trainer_bios_tool(event_id: str, query: str) -> list[str]:
    """Search for the instructor biographies, professional experience, and profile details.
    Use this to write trainer highlights or spotlights in newsletters or emails.
    """
    return await retrieve_knowledge_base_base(event_id, "professional", query)

#tool for brand style knowledge base retrieval from qdrant for a given event_id and query
@tool
async def search_branding_style_tool(event_id: str, query: str) -> list[str]:
    """Search for the company's copywriting guidelines, tone of voice, brand style, or copy rules.
    Use this to verify if drafts follow corporate writing standards.
    """
    return await retrieve_knowledge_base_base(event_id, "brand_style", query)

#tool for layout template knowledge base retrieval from qdrant for a given event_id and query
@tool
async def search_layout_templates_tool(event_id: str, query: str) -> list[str]:
    """Search for the structural layouts, outlines, formats, or blueprints for email, newsletter, or social media posts."""
    return await retrieve_knowledge_base_base(event_id, "layout_template", query)

#tool for few-shot example knowledge base retrieval from qdrant for a given event_id and query
@tool
async def search_few_shot_examples_tool(event_id: str, query: str) -> list[str]:
    """Search for previously approved, high-converting copy examples (emails, posts) to use as references."""
    return await retrieve_knowledge_base_base(event_id, "few_shot_example", query)


#tool for saving a generated marketing draft to firestore for a given event_id and content_type
@tool
async def save_campaign_draft_tool(event_id: str, content_type: str, draft_text: str) -> str:
    """Saves a generated marketing draft (content_type must be 'email', 'newsletter', or 'social') to the database."""
    await save_campaign_draft_base(event_id, content_type, draft_text)
    return f"Successfully saved {content_type} draft."

#tool for retrieving the complete campaign drafts and status package from firestore for a given event_id
@tool
async def get_campaign_package_tool(event_id: str) -> dict:
    """Retrieves the complete campaign bundle (email, newsletter, and social drafts) currently saved in the database."""
    return await get_campaign_package_base(event_id)

#tool for updating the status of a campaign in firestore for a given event_id
@tool
async def update_campaign_status_tool(event_id: str, status: str) -> str:
    """Updates the status lifecycle of the campaign (e.g. 'drafting', 'checking', 'review', or 'approved')."""
    await update_campaign_status_base(event_id, status)
    return f"Campaign status updated to {status}."

#tool for retrieving the layout template for a given event_id and content_type
@tool
async def get_layout_template_tool(event_id: str, content_type: str) -> str:
    """Retrieves the layout template or outline structure that copywriters must follow for 'email', 'newsletter', or 'social' posts."""
    return await get_layout_template_base(event_id, content_type)

#tool for verifying if url of event registration or website is live and active for a given url
@tool
async def verify_url_status_tool(url: str) -> bool:
    """Tests if a generated registration URL or website link is live and active (returns True if 200 OK, False if broken)."""
    if not url:
        return False
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False

#tool for retrieving the list of all registered event IDs from firestore
@tool
async def get_all_active_events_tool() -> list[str]:
    """Retrieves a list of all active event IDs currently stored in the database."""
    return await get_all_active_events_base()