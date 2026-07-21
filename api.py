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
    channel: str # "email", "newsletter", or "social"

class RejectRequest(BaseModel):
    event_id: str
    channel: str # "email", "newsletter", or "social"
    feedback: Optional[str] = None # For AI rewrite loop
    manual_edit: Optional[str] = None # For human manual overwrite

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
    """approve specific campaign channel when everything is correct"""
    if not payload.event_id or not payload.channel:
        raise HTTPException(status_code=400, detail="event_id and channel are required.")
        
    try:
        campaign_ref = async_firestore_db.collection("campaigns").document(payload.event_id)
        campaign_doc = await campaign_ref.get()
        if not campaign_doc.exists:
            raise HTTPException(status_code=404, detail=f"Campaign not found for ID: {payload.event_id}")
            
        campaign_data = campaign_doc.to_dict()
        
        # Get or initialize channel statuses dictionary
        channel_statuses = campaign_data.get("channel_statuses") or {}
        target_contents = campaign_data.get("target_contents") or []
        
        # Initialize statuses if missing
        for ch in target_contents:
            if ch not in channel_statuses:
                channel_statuses[ch] = "review"
                
        if payload.channel not in target_contents:
            raise HTTPException(
                status_code=400, 
                detail=f"Channel '{payload.channel}' is not part of this campaign's target contents: {target_contents}"
            )
            
        # Update channel status to approved
        channel_statuses[payload.channel] = "approved"
        update_data = {"channel_statuses": channel_statuses}
        
        # If all active channels are now approved, mark the overall status as approved
        all_approved = True
        for ch in target_contents:
            if channel_statuses.get(ch) != "approved":
                all_approved = False
                break
                
        if all_approved:
            update_data["status"] = "approved"
        else:
            update_data["status"] = "review"  # keep in review if other channels are still pending
            
        await campaign_ref.set(update_data, merge=True)
        return {
            "status": "success", 
            "message": f"Channel '{payload.channel}' of campaign '{payload.event_id}' has been approved.",
            "campaign_status": update_data.get("status", "review"),
            "channel_statuses": channel_statuses
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to approve channel: {str(e)}")

@api.post("/api/campaigns/reject")
async def reject_campaign(payload: RejectRequest):
    """reject specific campaign channel and apply manual edit or run automated AI rewrite loop"""
    if not payload.event_id or not payload.channel:
        raise HTTPException(status_code=400, detail="event_id and channel are required.")
    if not payload.feedback and not payload.manual_edit:
        raise HTTPException(status_code=400, detail="Either feedback or manual_edit is required.")
        
    try:
        campaign_ref = async_firestore_db.collection("campaigns").document(payload.event_id)
        campaign_doc = await campaign_ref.get()
        if not campaign_doc.exists:
            raise HTTPException(status_code=404, detail=f"Campaign not found for ID: {payload.event_id}")
            
        campaign_data = campaign_doc.to_dict()
        target_contents = campaign_data.get("target_contents") or []
        
        if payload.channel not in target_contents:
            raise HTTPException(
                status_code=400, 
                detail=f"Channel '{payload.channel}' is not part of this campaign's target contents: {target_contents}"
            )
            
        # Get and update channel statuses
        channel_statuses = campaign_data.get("channel_statuses") or {}
        for ch in target_contents:
            if ch not in channel_statuses:
                channel_statuses[ch] = "review"
                
        # Guard: Check if the channel is already approved
        if channel_statuses.get(payload.channel) == "approved":
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot reject channel '{payload.channel}' because it is already approved."
            )
                
        # Handle manual human edit option
        if payload.manual_edit is not None:
            # Overwrite the draft copy directly with human edits (no AI loop)
            update_data = {
                f"{payload.channel}_draft": payload.manual_edit,
                "status": "review"
            }
            channel_statuses[payload.channel] = "review" # stays in review for confirmation
            update_data["channel_statuses"] = channel_statuses
            await campaign_ref.set(update_data, merge=True)
            
            # Fetch fresh state to return
            updated_doc = await campaign_ref.get()
            final_data = updated_doc.to_dict()
            
            return {
                "status": "success",
                "message": f"Manual edit saved successfully for channel '{payload.channel}'.",
                "campaign_state": {
                    "event_id": payload.event_id,
                    "target_contents": target_contents,
                    "status": "review",
                    "channel_statuses": channel_statuses,
                    "revision_counts": final_data.get("revision_counts") or {},
                    "email_draft": final_data.get("email_draft"),
                    "newsletter_draft": final_data.get("newsletter_draft"),
                    "social_draft": final_data.get("social_draft")
                }
            }
            
        # Handle AI rewrite loop option (feedback provided)
        else:
            channel_statuses[payload.channel] = "rejected"
            
            # Prepare initial state running rewrite loop ONLY for the rejected channel
            initial_state = {
                "user_prompt": f"Revise {payload.channel} draft for campaign {payload.event_id}",
                "event_id": payload.event_id,
                "target_contents": [payload.channel],
                "email_draft": campaign_data.get("email_draft", ""),
                "newsletter_draft": campaign_data.get("newsletter_draft", ""),
                "social_draft": campaign_data.get("social_draft", ""),
                "revision_counts": campaign_data.get("revision_counts") or {ch: 0 for ch in target_contents},
                "review_feedback": {payload.channel: payload.feedback},
                "status": "drafting",
                "is_rewrite": True
            }
            
            # Run graph execution loop
            final_state = await app.ainvoke(initial_state)
            
            # Get new revision counts and drafts
            new_revision_counts = final_state.get("revision_counts") or {}
            
            # Revert status of this channel back to 'review' for human review
            channel_statuses[payload.channel] = "review"
            
            update_data = {
                "email_draft": final_state.get("email_draft"),
                "newsletter_draft": final_state.get("newsletter_draft"),
                "social_draft": final_state.get("social_draft"),
                "revision_counts": new_revision_counts,
                "review_feedback": final_state.get("review_feedback"),
                "channel_statuses": channel_statuses,
                "status": "review"
            }
            
            await campaign_ref.set(update_data, merge=True)
            
            return {
                "status": "success",
                "message": f"Rejection processed for channel '{payload.channel}'. Rewrite loop completed.",
                "campaign_state": {
                    "event_id": payload.event_id,
                    "target_contents": target_contents,
                    "status": "review",
                    "channel_statuses": channel_statuses,
                    "revision_counts": new_revision_counts,
                    "email_draft": update_data["email_draft"],
                    "newsletter_draft": update_data["newsletter_draft"],
                    "social_draft": update_data["social_draft"]
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reject channel: {str(e)}")

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
        extracted_id, is_updated = await auto_ingest_event(temp_file_path, event_id=event_id, category=category)
        action = "updated" if is_updated else "created"
        
        return {
            "status": "success",
            "action": action,
            "message": f"File '{file.filename}' successfully {action} under event_id: '{extracted_id}'."
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
        docs = events_ref.stream()
        
        events_list = []
        async for doc in docs:
            events_list.append(doc.to_dict())
            
        return {"status": "success", "events": events_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list events: {str(e)}")

@api.get("/api/knowledge/events/search")
async def search_registered_events(
    query: str = Query(..., description="Query term to search for registered events by ID or name")
):
    """search for registered event ids or names using keywords and substring matching"""
    try:
        events_ref = async_firestore_db.collection("events")
        docs = events_ref.stream()
        
        query_lower = query.lower()
        results = []
        
        async for doc in docs:
            data = doc.to_dict()
            event_id = data.get("event_id", "").lower()
            event_name = data.get("event_name", "").lower()
            
            match_score = 0.0
            # Direct match or substring contains
            if query_lower in event_id or query_lower in event_name:
                match_score = 1.0
            else:
                # Word-level intersection checks (handles typos or partial matches like 'banglore' matching 'bangalore')
                query_words = query_lower.split()
                
                # Check character level matches for fuzzy similarity
                from difflib import SequenceMatcher
                for word in query_words:
                    # check if word is close to event_id segments or event_name segments
                    for seg in event_id.split("_"):
                        ratio = SequenceMatcher(None, word, seg).ratio()
                        if ratio > 0.7:
                            match_score = max(match_score, ratio)
                    for seg in event_name.split():
                        ratio = SequenceMatcher(None, word, seg.lower()).ratio()
                        if ratio > 0.7:
                            match_score = max(match_score, ratio)
                            
            if match_score > 0.2:
                results.append({
                    "event_id": data.get("event_id"),
                    "event_name": data.get("event_name"),
                    "relevance_score": round(match_score, 2),
                    "price": data.get("price"),
                    "dates": data.get("dates"),
                    "registration_url": data.get("registration_url")
                })
                
        # Sort results by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search events: {str(e)}")



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

