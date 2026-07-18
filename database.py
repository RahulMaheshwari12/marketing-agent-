import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, AsyncQdrantClient
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables
load_dotenv()

# =====================================================================
# 1. Qdrant (Vector DB) Clients
# =====================================================================
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY", "")

# Initialize sync Qdrant Client (for LangChain compatibility)
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key if qdrant_api_key else None
)

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


# Quick self-test script to verify connections
if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("Checking database connections...")
        # Check Qdrant connection
        try:
            collections = await async_qdrant_client.get_collections()
            print(f"✓ Connected to Qdrant successfully! (Found {len(collections.collections)} collections)")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            
        # Check Firebase connection
        if async_firestore_db:
            print("✓ Connected to Firebase Firestore successfully!")
        else:
            print("❌ Firebase Firestore client not initialized.")
        
    asyncio.run(main())
