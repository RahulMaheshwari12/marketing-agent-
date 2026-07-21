import os
import asyncio
import argparse
import hashlib
from google.cloud import firestore
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Import our database clients from database.py
from database import qdrant_client, async_firestore_db

def calculate_file_hash(file_path: str) -> str:
    """Calculates the MD5 hash of a file's raw bytes to detect duplicates."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


load_dotenv()  # Load environment variables from .env file

# Initialize Gemini Embeddings
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Load collection name from .env, with a fallback default
COLLECTION_NAME = os.getenv("Collection_name", "Hidevs_knowledge_base").strip()

# Initialize Gemini Chat model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

#Pydantic schema for AI document routing and classification
class DocumentRouting(BaseModel):
    category: Literal["campaign", "professional", "brand_style", "layout_template", "few_shot_example"] = Field(
        description=(
            "Classify the document type: "
            "'campaign' (event flyers/syllabus/curriculum), "
            "'professional' (trainer biography/profile), "
            "'brand_style' (branding guidelines/tone of voice/copy rules), "
            "'layout_template' (structure outlines for email/newsletter/social), or "
            "'few_shot_example' (previously approved campaign copy used as few-shot references, "
            "or emerging technology release notes, framework updates, and technical trends)."
        )
    )
    event_id_suggestion: str = Field(
        description=(
            "The targeted database folder name: "
            "If category is 'campaign', output a clean lowercase snake_case slug of the event name (e.g., 'nextjs_bootcamp'). "
            "If category is 'professional', output 'trainers'. "
            "If category is 'brand_style', output 'branding'. "
            "If category is 'layout_template', output 'templates'. "
            "If category is 'few_shot_example' and it contains technical trends/release notes/framework updates, output 'trends'. "
            "If category is 'few_shot_example' and it is an approved copy reference, output 'few_shots'."
        )
    )

#Pydantic schema for event static metadata extraction
class EventMetadata(BaseModel):
    event_name: str = Field(description="The name of the bootcamp or event")
    registration_url: str = Field(description="The official signup or registration URL found in the text. If not found, output an empty string.")
    price: str = Field(description="The price of the bootcamp/event, including any early bird details.")
    dates: str = Field(description="The start and end dates/times of the event.")
    suggested_hashtags: str = Field(description="3-5 relevant social media hashtags (e.g., #GenAI #LangGraph)")

def load_documents_from_file(file_path: str):
    """Load a text file or PDF file and return a list of documents."""
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at: {file_path}")
    
    # Determine the file extension and load accordingly
    ext = os.path.splitext(file_path)[-1].lower()

    # Load the file based on its extension
    if ext == ".txt":
        Loader = TextLoader(file_path, encoding="utf-8")
        return Loader.load()
    elif ext == ".pdf":
        Loader = PyPDFLoader(file_path)
        return Loader.load()
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .txt and .pdf are supported.")
    
#AI meta data Extractor 
async def extract_event_metadata(text_content: str) -> EventMetadata:
    """uses Gemini to extract event metadata from the text content of a file."""
    print(f"Extracting event metadata from text content...")

    #initialize structured output LLM with the EventMetadata schema
    structured_llm = llm.with_structured_output(EventMetadata)

    prompt = f"""
    You are an expert data extractor. Analyze the following event brochure text and extract the key details.
    
    Event brochure text:
    ---
    {text_content}
    ---
    """

    #Invoke the structured LLM with the prompt
    extracted_data = await structured_llm.ainvoke(prompt)
    return extracted_data 

#AI document classifier and router to determine the category and event_id for database storage
async def classify_and_route_document(text_content: str) -> DocumentRouting:
    """Uses Gemini to read the document and determine the database category and event ID."""
    print("AI is classifying the document type and determining database routing...")
    
    structured_llm = llm.with_structured_output(DocumentRouting)
    
    # We pass the first 2500 characters which is plenty of text to classify the type
    prompt = f"""
    You are an AI database administrator. Read the following document and classify what type of marketing data it is.
    
    Document text:
    ---
    {text_content[:2500]}
    ---
    """
    
    routing_decision = await structured_llm.ainvoke(prompt)
    return routing_decision
    
#database deletion utility to clean up Qdrant vectors and Firestore documents
async def delete_database_records(event_id: str, category: str = None):
    """Deletes records from Qdrant vectors and Firestore documents for a given event and optional category."""
    if not event_id:
        raise ValueError("Must provide event_id to execute database deletions.")
        
    print(f"Starting database deletion process for event_id: '{event_id}'...")
    
    # 1. Delete from Qdrant
    conditions = [FieldCondition(key="metadata.event_id", match=MatchValue(value=event_id))]
    if category:
        conditions.append(FieldCondition(key="metadata.category", match=MatchValue(value=category)))
        print(f"Removing vectors in Qdrant for event_id: '{event_id}' and category: '{category}'...")
    else:
        print(f"Removing ALL vectors in Qdrant for event_id: '{event_id}' across all categories...")
        
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(must=conditions)
    )
    
    # 2. Delete from Firestore (Wipe collections if no category is specified)
    if not category:
        if async_firestore_db:
            print(f"Deleting event document '{event_id}' from Firestore events collection...")
            await async_firestore_db.collection("events").document(event_id).delete()
            print(f"Deleting campaign document '{event_id}' from Firestore campaigns collection...")
            await async_firestore_db.collection("campaigns").document(event_id).delete()
    else:
        if category == "campaign" and async_firestore_db:
            print(f"Deleting event metadata '{event_id}' from Firestore events collection...")
            await async_firestore_db.collection("events").document(event_id).delete()
            
    print(f"SUCCESS: Database deletion completed successfully for event_id: '{event_id}'.")

#integrate static data into firebase 
async def ingest_static_metadata(event_id: str, metadata: EventMetadata):
    """Saves static event details to Firebase Firestore under the event_id document."""

    #checking if firebase client is initialized
    if not async_firestore_db:
        raise RuntimeError("Firebase Firestore client is not initialized. Please check your Firebase credentials.")
    
    print(f"Upload metadata for the event_id: {event_id} to Firebase....")
    doc_ref = async_firestore_db.collection("events").document(event_id)

    #storing the extracted metadata in firebase
    await doc_ref.set({
        "event_id": event_id,
        "event_name": metadata.event_name,
        "registration_url": metadata.registration_url,
        "price": metadata.price,
        "dates": metadata.dates,
        "suggested_hashtags": metadata.suggested_hashtags
    })
    print(f"Metadata for event_id: {event_id} successfully uploaded to Firebase Firestore.")


#AI document ingestion pipeline that loads a file, classifies it, extracts metadata, and uploads vectors to Qdrant
async def auto_ingest_event(file_path: str, event_id: str = None, category: str = None):
    """Loads file, classifies it if needed, extracts metadata, and uploads vectors to Qdrant."""
    print(f"Loading document from '{file_path}'...")
    
    # Load the document
    raw_docs = load_documents_from_file(file_path)
    full_text = "\n".join([doc.page_content for doc in raw_docs])
    
    # Classification Phase: Dynamic routing runs if parameters are not explicitly passed
    if not event_id or not category:
        routing = await classify_and_route_document(full_text)
        if not event_id:
            event_id = routing.event_id_suggestion
        if not category:
            category = routing.category
        print(f"AI Auto-Classification Result -> Event ID: '{event_id}' | Category: '{category}'")

    # Validate categories (including the 2 new ones)
    valid_categories = ["campaign", "professional", "brand_style", "layout_template", "few_shot_example"]
    if category not in valid_categories:
        raise ValueError(f"Invalid category '{category}'. Must be one of {valid_categories}")

    # Duplicate Verification Phase: Calculate hash and verify if file has changed
    file_hash = calculate_file_hash(file_path)
    if async_firestore_db:
        meta_ref = async_firestore_db.collection("ingestion_metadata").document(f"{event_id}_{category}")
        meta_doc = await meta_ref.get()
        if meta_doc.exists and meta_doc.to_dict().get("file_hash") == file_hash:
            print(f"INFO: The file '{file_path}' has already been ingested. Skipping to save API tokens.")
            return

    # Extraction Phase: Save metadata fields to Firestore for campaigns
    if category == "campaign":
        extracted_metadata = await extract_event_metadata(full_text)
        print(f"AI Extracted Metadata: {extracted_metadata}")
        await ingest_static_metadata(event_id, extracted_metadata)
    else:
        print(f"Skipping static metadata extraction for non-campaign category: '{category}'")

    # Chunking Phase: Split document pages for vectorization
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "•", "-", "*", " ", ""]
    )
    split_docs = splitter.split_documents(raw_docs)
    print(f"Split document into {len(split_docs)} chunks.")

    # Metadata Enrichment: Tag chunks with folder and category keys
    for doc in split_docs:
        doc.metadata["event_id"] = event_id
        doc.metadata["category"] = category

    # Vector Storage Phase: Embed and upsert chunks to Qdrant collection
    print(f"Uploading vectors to Qdrant collection '{COLLECTION_NAME}'...")
    from qdrant_client.models import VectorParams, Distance
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Creating Qdrant collection '{COLLECTION_NAME}'...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings_model
    )
    
    # Clean-on-Ingest Phase: Wipe out old vectors for this event ID and category to prevent duplicates
    print(f"Cleaning up old Qdrant vectors for '{event_id}' [{category}]...")
    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(key="metadata.event_id", match=MatchValue(value=event_id)),
                FieldCondition(key="metadata.category", match=MatchValue(value=category))
            ]
        )
    )
    
    await vector_store.aadd_documents(split_docs)
    print(f"SUCCESS: Uploaded {len(split_docs)} chunks to Qdrant.")
    
    # Save the file hash to Firestore for future de-duplication
    if async_firestore_db:
        await async_firestore_db.collection("ingestion_metadata").document(f"{event_id}_{category}").set({
            "file_hash": file_hash,
            "last_ingested": firestore.SERVER_TIMESTAMP
        })
        
    print("--- Auto-Ingestion Completed Successfully! ---")
    return event_id

# Configure CLI parser interface
if __name__ == "__main__":
    # Setting up argument parser for command-line execution
    parser = argparse.ArgumentParser(description="AI Auto-ingest documents into Qdrant and Firebase.")
    parser.add_argument("file_path", type=str, nargs="?", default=None, help="Path to the document file (.txt or .pdf).")
    
    # Optional flags (AI will determine them if left blank)
    parser.add_argument("--event_id", type=str, default=None, help="Force a specific event identifier/folder name.")
    parser.add_argument("--category", type=str, default=None, 
                        choices=["campaign", "professional", "brand_style", "layout_template", "few_shot_example"], 
                        help="Force a specific category folder.")
    parser.add_argument("--delete", action="store_true", help="Perform manual deletion of database records.")

    # Parsing command-line arguments
    args = parser.parse_args()

    # Execution block: launch pipeline or deletion utility
    async def main():
        try:
            if args.delete:
                if not args.event_id:
                    raise ValueError("Must provide --event_id to specify which records to delete.")
                await delete_database_records(event_id=args.event_id, category=args.category)
            else:
                if not args.file_path:
                    raise ValueError("Must provide a file_path to ingest documents, or use --delete to remove records.")
                await auto_ingest_event(
                    file_path=args.file_path,
                    event_id=args.event_id,
                    category=args.category
                )
        except FileNotFoundError as fnf_err:
            print(f"ERROR: File Error: {fnf_err}")
        except ValueError as val_err:
            print(f"ERROR: Input/Category Error: {val_err}")
        except Exception as e:
            print(f"ERROR: Unexpected System Error: {e}")

    asyncio.run(main())