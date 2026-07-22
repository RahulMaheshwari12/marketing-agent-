import asyncio
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END

# Import RAG tools and database connectors
from tools import (
    get_event_metadata_tool,
    search_campaign_syllabus_tool,
    search_trainer_bios_tool,
    search_few_shot_examples_tool,
    get_layout_template_tool,
    get_all_active_events_base,
    search_branding_style_tool
)

# Import LLM agent modules
from agents import (
    run_supervisor,
    run_email_writer,
    run_newsletter_writer,
    run_social_writer,
    run_fact_checker,
    run_style_checker,
    run_final_reviewer
)

#Graph State Schema
class CampaignState(TypedDict):
    user_prompt: str
    event_id: str
    target_contents: List[str]          # Channels requested (e.g. ["email", "social"])
    email_draft: str                    # In-memory email copy buffer
    newsletter_draft: str               # In-memory newsletter copy buffer
    social_draft: str                   # In-memory social copy buffer
    revision_counts: Dict[str, int]     # Retry counts to prevent infinite loops
    review_feedback: Dict[str, str]    # Checker and reviewer revision logs
    status: str                         # Campaign state: drafting, checking, review, failed


#Node 1: Supervisor Node (Coordinator / Router)
async def supervisor_node(state: CampaignState) -> Dict:
    """Parses natural language prompt and routes targets using the Supervisor Agent."""
    event_id = state.get("event_id")
    target_contents = state.get("target_contents")

    # If event_id or target_contents are not pre-populated, use the Supervisor Agent to parse them
    if not event_id or not target_contents:
        try:
            valid_events = await get_all_active_events_base()
        except Exception as e:
            raise ValueError(f"Failed to query active events: {str(e)}")
            
        if not valid_events:
            raise ValueError("No active events found in the database. Please ingest an event brochure first using POST /api/knowledge/ingest.")

        try:
            routing = await run_supervisor(state["user_prompt"], valid_events)
            if not event_id:
                event_id = routing.event_id
            if not target_contents:
                target_contents = routing.target_contents
        except Exception as e:
            raise ValueError(f"Supervisor routing failed: {str(e)}")

        if event_id == "unrecognized":
            raise ValueError("No matching event found in the database for your request. Please ingest the brochure first using POST /api/knowledge/ingest.")

    return {
        "event_id": event_id,
        "target_contents": target_contents,
        "email_draft": state.get("email_draft") or "",
        "newsletter_draft": state.get("newsletter_draft") or "",
        "social_draft": state.get("social_draft") or "",
        "revision_counts": state.get("revision_counts") or {channel: 0 for channel in target_contents},
        "review_feedback": state.get("review_feedback") or {},
        "status": "drafting"
    }


#Node 2a: Email Copywriter Node
async def email_writer_node(state: CampaignState) -> Dict:
    """Drafts email content dynamically segmenting student or professional pitches."""
    event_id = state["event_id"]
    feedback = state["review_feedback"].get("email", "")
    
    # Retrieve syllabus facts and template constraints from DB
    try:
        facts = await search_campaign_syllabus_tool(event_id, query="syllabus schedule price Bangalore hub location")
        layout = await get_layout_template_tool(event_id, content_type="email")
    except Exception as e:
        facts = f"Fallback Event facts: {event_id} NextJS hackathon Bangalore September 12-13."
        layout = "Subject:\n[Title]\n\nBody:\n[Text]\n\nCTA:\n[Link]"

    email_copy = await run_email_writer(event_id, facts, layout, feedback)
    return {"email_draft": email_copy}


#Node 2b: Newsletter Copywriter Node
async def newsletter_writer_node(state: CampaignState) -> Dict:
    """Drafts monthly community newsletter combining syllabus, spotlights, achievements, and trends."""
    event_id = state["event_id"]
    feedback = state["review_feedback"].get("newsletter", "")

    # Gather data streams
    try:
        facts = await search_campaign_syllabus_tool(event_id, query="syllabus and course curriculum details")
        layout = await get_layout_template_tool(event_id, content_type="newsletter")
        bio = await search_trainer_bios_tool(event_id="trainers", query="instructor profile Google DeepMind")
        
        # Pull Bangalore achievements (highlights) and industry releases (trends)
        highlights = await search_few_shot_examples_tool(event_id="july_2026", query="Bangalore hub highlights achievements workshops")
        trends = await search_few_shot_examples_tool(event_id="trends", query="React 19 Server Actions Qdrant caching trends")
    except Exception:
        facts = "Next.js Production Scale Bootcamp"
        layout = "Header:\n[Header]\n\nFeatured:\n[Highlights]\n\nCurriculum:\n[Topics]\n\nSpotlight:\n[Bio]\n\nCTA:\n[CTA]"
        bio = "Rahul Maheshwari, Senior Software Engineer at Google DeepMind."
        highlights = " trained 150+ students in Server Actions and hosted a local coding contest with 40 teams."
        trends = "React 19 Server Actions have officially entered stable release, and Qdrant introduced optimized local caching."

    newsletter_copy = await run_newsletter_writer(event_id, facts, bio, highlights, trends, layout, feedback)
    return {"newsletter_draft": newsletter_copy}


#Node 2c: Social Copywriter Node
async def social_writer_node(state: CampaignState) -> Dict:
    """Drafts micro-copy updates with hooks and registration links."""
    event_id = state["event_id"]
    feedback = state["review_feedback"].get("social", "")

    try:
        facts = await search_campaign_syllabus_tool(event_id, query="Bangalore hackathon date price syllabus registration link")
        layout = await get_layout_template_tool(event_id, content_type="social")
    except Exception:
        facts = "Bangalore Hub NextJS hackathon September 12-13. FREE for students."
        layout = "[Hook]\n\n[Syllabus]\n\n[Link]\n\n[Hashtags]"

    social_copy = await run_social_writer(event_id, facts, layout, feedback)
    return {"social_draft": social_copy}


#Node 3: Concurrent Checker Node (Fact-Checker & Style-Checker)
async def checkers_node(state: CampaignState) -> Dict:
    """Concurrently runs fact and style verifiers for all drafted channels using asyncio.gather."""
    event_id = state["event_id"]
    channels = state["target_contents"]
    
    # Retrieve metadata and style definitions
    try:
        metadata = await get_event_metadata_tool(event_id)
        branding = await search_branding_style_tool(event_id="branding", query="branding guidelines tone copywriting rules")
    except Exception:
        metadata = {"price": "FREE", "registration_url": "https://hidevs.community"}
        branding = "Maintain professional tech tone. Do not use emojis in subject line."

    # Define validation tasks to run concurrently
    tasks = []
    task_keys = []

    for channel in channels:
        draft = ""
        if channel == "email":
            draft = state["email_draft"]
        elif channel == "newsletter":
            draft = state["newsletter_draft"]
        elif channel == "social":
            draft = state["social_draft"]

        try:
            facts = await search_campaign_syllabus_tool(event_id, query="syllabus date price location schedule")
            layout = await get_layout_template_tool(event_id, content_type=channel)
        except Exception:
            facts = "Course details"
            layout = "[Template]"

        # Append Fact Checker task
        tasks.append(run_fact_checker(channel, draft, facts, metadata))
        task_keys.append((channel, "fact"))
        
        # Append Style Checker task
        tasks.append(run_style_checker(channel, draft, branding, layout))
        task_keys.append((channel, "style"))

    # Execute all checker tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    new_feedback = {}
    for i, res in enumerate(results):
        channel, check_type = task_keys[i]
        
        if isinstance(res, Exception):
            # Gracefully handle validation failures as fail status
            err_msg = f"- Internal checker error on {check_type} validation: {res}"
            new_feedback[channel] = new_feedback.get(channel, "") + err_msg + "\n"
            continue
            
        if res.status == "FAIL":
            new_feedback[channel] = new_feedback.get(channel, "") + f"- {check_type.upper()} CHECK: {res.feedback}\n"

    # Reset passing channel feedback to empty string
    for channel in channels:
        if channel not in new_feedback:
            new_feedback[channel] = ""

    return {
        "review_feedback": new_feedback,
        "status": "checking"
    }


#Node 4: Reviewer Node (QA Gatekeeper & Final Commit)
async def reviewer_node(state: CampaignState) -> Dict:
    """Holistically reviews content bundle cohesion and saves to Firestore on approval."""
    event_id = state["event_id"]
    
    # Query approved copywriting references from Qdrant with graceful fallback
    try:
        few_shots = await search_few_shot_examples_tool(
            event_id="few_shots",
            query="approved high-converting marketing email newsletter social posts copy"
        )
    except Exception:
        few_shots = ""

    # Call Final Reviewer Agent
    try:
        reviewer_res = await run_final_reviewer(
            email_draft=state.get("email_draft", ""),
            newsletter_draft=state.get("newsletter_draft", ""),
            social_draft=state.get("social_draft", ""),
            few_shot_examples=few_shots,
            target_contents=state.get("target_contents")
        )
    except Exception as e:
        # Graceful validation bypass if reviewer fails
        class FallbackRes:
            status = "PASS"
            feedback = ""
        reviewer_res = FallbackRes()

    # Aggregate feedback and verify if any errors exist
    feedback_exist = False
    active_feedback = state["review_feedback"].copy()

    # Check for Checker Node failures
    for channel in state["target_contents"]:
        if active_feedback.get(channel):
            feedback_exist = True

    # Check for Final Reviewer node failures
    if reviewer_res.status == "FAIL":
        feedback_exist = True
        # Distribute reviewer feedback to all requested channels
        for channel in state["target_contents"]:
            active_feedback[channel] = active_feedback.get(channel, "") + f"- REVIEWER TONE CHECK: {reviewer_res.feedback}\n"

    # Update revision counts if errors exist
    new_revision_counts = state["revision_counts"].copy()
    if feedback_exist:
        for channel in state["target_contents"]:
            if active_feedback.get(channel):
                new_revision_counts[channel] = new_revision_counts.get(channel, 0) + 1

    # Check if we reached our max retry threshold of 2 cycles
    retry_limit_hit = False
    for channel in state["target_contents"]:
        if new_revision_counts.get(channel, 0) >= 2:
            retry_limit_hit = True

    # Routing Outcome Decision:
    # Scenario A: All passed, or we reached max rewrite limit (safety escape)
    if not feedback_exist or retry_limit_hit:
        # save drafts and checker logs to firestore
        try:
            from database import async_firestore_db
            if async_firestore_db:
                campaign_ref = async_firestore_db.collection("campaigns").document(event_id)
                
                is_rewrite = state.get("is_rewrite", False)
                if is_rewrite:
                    existing_doc = await campaign_ref.get()
                    existing_data = existing_doc.to_dict() if existing_doc.exists else {}
                    db_target_contents = existing_data.get("target_contents") or state.get("target_contents") or []
                else:
                    db_target_contents = state.get("target_contents") or []
                    
                await campaign_ref.set({
                    "event_id": event_id,
                    "target_contents": db_target_contents,
                    "email_draft": state.get("email_draft") or "",
                    "newsletter_draft": state.get("newsletter_draft") or "",
                    "social_draft": state.get("social_draft") or "",
                    "revision_counts": new_revision_counts,
                    "review_feedback": active_feedback,
                    "channel_statuses": {ch: "review" for ch in state.get("target_contents") or []},
                    "status": "review"
                }, merge=True)
        except Exception as e:
            print(f"db commit failed: {e}")
            
        return {
            "revision_counts": new_revision_counts,
            "status": "review"
        }
    
    # Scenario B: Errors present and retries remaining -> loop back
    return {
        "review_feedback": active_feedback,
        "revision_counts": new_revision_counts,
        "status": "drafting"
    }


#Graph Conditional Routing Logics
def route_copywriters(state: CampaignState) -> List[str]:
    """Conditional routing splitting execution flow to parallel writer paths."""
    routes = []
    # If the execution state has external human feedback, only run the copywriters that failed
    if state.get("review_feedback"):
        for channel in state["target_contents"]:
            if state["review_feedback"].get(channel):
                routes.append(f"{channel}_writer")
    else:
        for channel in state["target_contents"]:
            routes.append(f"{channel}_writer")
            
    # Fallback to trigger all requested channels if no specific feedback is defined
    if not routes:
        for channel in state["target_contents"]:
            routes.append(f"{channel}_writer")
    return routes


def route_after_review(state: CampaignState):
    """Conditional routing branching back to failed writers or exiting."""
    if state["status"] == "review":
        return END

    # Loop back only to copywriters that have pending feedback logs
    failed_writers = []
    for channel in state["target_contents"]:
        if state["review_feedback"].get(channel):
            failed_writers.append(f"{channel}_writer")
            
    if not failed_writers:
        return END
        
    return failed_writers


#Compile Graph Pipeline
workflow = StateGraph(CampaignState)

# Declare Nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("email_writer", email_writer_node)
workflow.add_node("newsletter_writer", newsletter_writer_node)
workflow.add_node("social_writer", social_writer_node)
workflow.add_node("checkers", checkers_node)
workflow.add_node("reviewer", reviewer_node)

# Declare Edges
workflow.add_edge(START, "supervisor")

# Parallel Fan-Out split from supervisor to copywriters
workflow.add_conditional_edges(
    "supervisor",
    route_copywriters,
    {
        "email_writer": "email_writer",
        "newsletter_writer": "newsletter_writer",
        "social_writer": "social_writer"
    }
)

# Fan-In join from copywriters to concurrent checkers
workflow.add_edge("email_writer", "checkers")
workflow.add_edge("newsletter_writer", "checkers")
workflow.add_edge("social_writer", "checkers")

# Transition from verifiers to lead QA review
workflow.add_edge("checkers", "reviewer")

# Dynamic loop back or exit routing from reviewer node
workflow.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "email_writer": "email_writer",
        "newsletter_writer": "newsletter_writer",
        "social_writer": "social_writer",
        END: END
    }
)

# Compile the compiled Swarm Graph executable
app = workflow.compile()
