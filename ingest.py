import os
import asyncio
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Import our database clients from database.py
from database import async_qdrant_client, async_firestore_db

load_dotenv()  # Load environment variables from .env file

def load_documents_from_file(file_path: str):
    """Load a text file or PDF file and return a list of documents."""
    