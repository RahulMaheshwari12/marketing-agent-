import os
import shutil
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# db and graph imports
from database import async_firestore_db
from graph import app

# ingestion and database deletion functions
from ingest import auto_ingest_event, delete_database_records, embeddings_model, COLLECTION_NAME, qdrant_client

# env config
load_dotenv()

#Initialize FastAPT server
api = FastAPI(
    title= "HiDevs marketing swarm API",
    description= "Backend service wrapper for HiDevs Multi-agent marketing campaign generation engine",
    version= "1.0.0"
)

#enable CORS for frontend and local connection 
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Directory to hold temporary uploads for ingestion process 
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_upload")
os.makedirs(UPLOAD_DIR, exist_ok= True)

@api.get("/")
def read_root():
    return {"status": "HiDevs Swarm API is running", "docs_url": "/docs"}

#API Pydantic Request model 
class GenerateRequest(BaseModel):
    users_prompt: str
    target_content: Optional[list[str]] = None #Allows selecting content type manually 

class ApproveRequest(BaseModel):
    event_id: str
    email_draft: Optional[str] = None
    newsletter_draft: Optional[str] = None
    social_draft: Optional[str] = None

class RejectRequest(BaseModel):
    event_id: str
    feedback: Optional[Dict[str, str]] = None  # Optional feedback dictionary (e.g. {"email": "make it longer"})

# campaign generation and review endpoints

@api.post("/api/campaigns/generate")
async def generate_campaign(payload: GenerateRequest):
    """trigger swarm campaign generation"""
    if not payload.users_prompt.strip():
        raise HTTPException(status_code=400, detail="User prompt cannot be empty.")

    initial_state = {
        "user_prompt": payload.users_prompt,
        "event_id": "",
        "target_contents": payload.target_content or [],
        "email_draft": "",
        "newsletter_draft": "",
        "social_draft": "",
        "revision_counts": {},
        "review_feedback": {},
        "status": ""
    }

    try:
        # Invoke the compiled state machine graph
        final_state = await app.ainvoke(initial_state)
        return {
            "event_id": final_state.get("event_id"),
            "target_contents": final_state.get("target_contents"),
            "status": final_state.get("status"),
            "revision_counts": final_state.get("revision_counts"),
            "review_feedback": final_state.get("review_feedback"),
            "email_draft": final_state.get("email_draft"),
            "newsletter_draft": final_state.get("newsletter_draft"),
            "social_draft": final_state.get("social_draft")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph Swarm Execution failed: {str(e)}")

@api.post("/api/campaigns/approve")
async def approve_campaign(payload: ApproveRequest):
    """approve campaign and save human manual edits"""
    if not payload.event_id:
        raise HTTPException(status_code=400, detail="event_id is required.")
        
    try:
        campaign_ref = async_firestore_db.collection("campaigns").document(payload.event_id)
        campaign = await campaign_ref.get()
        if not campaign.exists:
            raise HTTPException(status_code=404, detail=f"Campaign not found for ID: {payload.event_id}")
            
        # Prepare updates, supporting manual text edits submitted by the human reviewer
        update_data = {"status": "approved"}
        if payload.email_draft is not None:
            update_data["email_draft"] = payload.email_draft
        if payload.newsletter_draft is not None:
            update_data["newsletter_draft"] = payload.newsletter_draft
        if payload.social_draft is not None:
            update_data["social_draft"] = payload.social_draft

        await campaign_ref.set(update_data, merge=True)
        return {"status": "success", "message": f"Campaign {payload.event_id} has been approved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to approve campaign: {str(e)}")

@api.post("/api/campaigns/reject")
async def reject_campaign(payload: RejectRequest):
    """reject campaign and trigger loop revisions with feedback"""
    if not payload.event_id:
        raise HTTPException(status_code=400, detail="event_id is required.")
        
    try:
        campaign_ref = async_firestore_db.collection("campaigns").document(payload.event_id)
        campaign_doc = await campaign_ref.get()
        if not campaign_doc.exists:
            raise HTTPException(status_code=404, detail=f"Campaign not found for ID: {payload.event_id}")
            
        campaign_data = campaign_doc.to_dict()
        
        # If feedback comments are provided, run a revision loop in the graph
        if payload.feedback:
            target_contents = []
            if campaign_data.get("email_draft"):
                target_contents.append("email")
            if campaign_data.get("newsletter_draft"):
                target_contents.append("newsletter")
            if campaign_data.get("social_draft"):
                target_contents.append("social")

            initial_state = {
                "user_prompt": f"Revise campaign for {payload.event_id}",
                "event_id": payload.event_id,
                "target_contents": target_contents,
                "email_draft": campaign_data.get("email_draft", ""),
                "newsletter_draft": campaign_data.get("newsletter_draft", ""),
                "social_draft": campaign_data.get("social_draft", ""),
                "revision_counts": campaign_data.get("revision_counts") or {ch: 0 for ch in target_contents},
                "review_feedback": payload.feedback,
                "status": "drafting"
            }
            
            # Run graph execution loop
            final_state = await app.ainvoke(initial_state)
            
            return {
                "status": "success",
                "message": "Rejection processed. Automated rewrite loop completed.",
                "campaign_state": {
                    "event_id": final_state.get("event_id"),
                    "target_contents": final_state.get("target_contents"),
                    "status": final_state.get("status"),
                    "revision_counts": final_state.get("revision_counts"),
                    "review_feedback": final_state.get("review_feedback"),
                    "email_draft": final_state.get("email_draft"),
                    "newsletter_draft": final_state.get("newsletter_draft"),
                    "social_draft": final_state.get("social_draft")
                }
            }
        else:
            # If no feedback is provided, simply revert status back to "drafting"
            await campaign_ref.set({"status": "drafting"}, merge=True)
            return {"status": "success", "message": f"Campaign {payload.event_id} reverted back to drafting state."}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reject campaign: {str(e)}")

@api.get("/api/campaigns/{event_id}")
async def get_campaign(event_id: str):
    """get campaign drafts and status from firestore"""
    try:
        campaign_ref = async_firestore_db.collection("campaigns").document(event_id)
        campaign = await campaign_ref.get()
        if not campaign.exists:
            raise HTTPException(status_code=404, detail=f"Campaign not found for ID: {event_id}")
        return campaign.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read campaign data: {str(e)}")

# knowledge base management endpoints

@api.post("/api/knowledge/ingest")
async def ingest_knowledge_document(
    file: UploadFile = File(...),
    event_id: Optional[str] = Form(None),
    category: Optional[str] = Form(None)
):
    """upload and ingest new event document or bio"""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".txt", ".pdf"]:
        raise HTTPException(status_code=400, detail="Only .txt or .pdf files are supported.")
        
    temp_file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # Save file locally to process it
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Execute the auto-ingest pipeline
        await auto_ingest_event(temp_file_path, event_id=event_id, category=category)
        
        return {
            "status": "success",
            "message": f"File '{file.filename}' successfully ingested into knowledge base."
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        # Clean up temporary uploaded file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@api.get("/api/knowledge/events")
async def list_events():
    """get all registered event ids from firestore"""
    try:
        events_ref = async_firestore_db.collection("events")
        docs = await events_ref.stream()
        
        events_list = []
        async for doc in docs:
            events_list.append(doc.to_dict())
            
        return {"status": "success", "events": events_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list events: {str(e)}")

@api.get("/api/knowledge/search")
async def search_knowledge(
    event_id: str,
    query: str,
    category: Optional[str] = Query(None, description="Optional category to filter by"),
    limit: int = Query(5, description="Number of results to return")
):
    """semantic search in qdrant filtered by event id"""
    if not event_id or not query:
        raise HTTPException(status_code=400, detail="Both event_id and query parameters are required.")
        
    try:
        # embed query
        query_vector = await embeddings_model.aembed_query(query)
        
        # qdrant filter conditions
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = [FieldCondition(key="metadata.event_id", match=MatchValue(value=event_id))]
        if category:
            conditions.append(FieldCondition(key="metadata.category", match=MatchValue(value=category)))
            
        # search in qdrant
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=Filter(must=conditions),
            limit=limit
        )
        
        # format hits
        hits = []
        for hit in search_results:
            hits.append({
                "score": hit.score,
                "text": hit.payload.get("page_content", ""),
                "metadata": hit.payload.get("metadata", {})
            })
            
        return {"status": "success", "results": hits}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@api.delete("/api/knowledge/{event_id}")
async def delete_knowledge_records(
    event_id: str,
    category: Optional[str] = Query(None, description="Optional category to delete (e.g. brand_style)")
):
    """delete knowledge base files from qdrant and firestore"""
    try:
        await delete_database_records(event_id=event_id, category=category)
        return {
            "status": "success",
            "message": f"Successfully deleted knowledge base records for event_id: '{event_id}'."
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

