import os
import json
import numpy as np # Used by sentence-transformers
from dotenv import load_dotenv

# --- Google Gen AI SDK for Gemini 2.5 Pro ---
from google import genai 
from google.genai import types

# --- Vector Database Imports ---
from chromadb import PersistentClient 
from sentence_transformers import SentenceTransformer

# --- Import your Spinvic_rag component functions ---
# NOTE: Ensure your project structure allows these relative imports to work.
# If they fail, you may need to adjust your PYTHONPATH or run from the root directory.
from rag_pipeline.data_loader import process_pdf_to_structured_jsonl
from rag_pipeline.text_chunker import chunk_processed_data
from rag_pipeline
.embedder import embed_and_store_chunks 

load_dotenv()

# --- 1. Configuration Setup ---

PDF_FILE_PATH = "input_doc.pdf"          # Your source document to be indexed
JSONL_OUTPUT_PATH = "temp_data.jsonl"    # Intermediate file
DB_DIRECTORY = "./chroma_db"             # Location of the persistent vector store
COLLECTION_NAME = "spinvic_rag_collection"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"     # Recommended embedding model for speed/quality

# Gemini Configuration
GEMINI_MODEL = "gemini-2.5-pro"          # The model you specified
MAX_RETRIEVED_CHUNKS = 3                 # Number of documents to fetch for context

# Ensure the API key is set in your .env file: GOOGLE_API_KEY="YOUR_API_KEY"
try:
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    if not API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    # Initialize the Gemini Client
    GEMINI_CLIENT = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    GEMINI_CLIENT = None


# --- 2. Indexing Phase: PDF -> Chunks -> Embeddings (Vector DB) ---

def build_vector_index():
    """
    Orchestrates the entire indexing process using your existing modules.
    """
    if not os.path.exists(PDF_FILE_PATH):
        raise FileNotFoundError(f"Source PDF not found at: {PDF_FILE_PATH}")

    # 1. Load and Process PDF (using data_loader.py)
    print(f"--- Stage 1: Processing PDF: {PDF_FILE_PATH} ---")
    process_pdf_to_structured_jsonl(PDF_FILE_PATH, JSONL_OUTPUT_PATH)
    print(f"PDF processed and structured data saved to {JSONL_OUTPUT_PATH}")

    # 2. Chunk Processed Data (using text_chunker.py)
    print(f"\n--- Stage 2: Chunking Data ---")
    chunked_documents = chunk_processed_data(JSONL_OUTPUT_PATH)
    print(f"Data chunked into {len(chunked_documents)} total chunks.")

    # 3. Embed and Store Chunks (using embedder.py)
    print(f"\n--- Stage 3: Embedding and Storing ---")
    # Note: embedder.py needs the fix mentioned in section 4.
    embed_and_store_chunks(
        chunked_documents,
        model_name=EMBEDDING_MODEL,
        collection_name=COLLECTION_NAME,
        db_path=DB_DIRECTORY
    )
    
    print("\n✅ Indexing Complete.")


# --- 3. Retrieval and Generation Phase: Query -> Answer ---

def run_query(query: str) -> str:
    """
    Performs retrieval using ChromaDB and generation using Gemini 2.5 Pro.
    """
    if GEMINI_CLIENT is None:
        return "ERROR: Gemini client not initialized. Check your GOOGLE_API_KEY."

    print(f"\n\n--- Running Query: '{query}' ---")
    
    # 1. Initialize the ChromaDB client and collection
    client = PersistentClient(path=DB_DIRECTORY)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # 2. Embed the Query
    # SentenceTransformer requires a fix in embedder.py to be imported correctly.
    embedding_model = SentenceTransformer(EMBEDDING_MODEL) 
    query_embedding = embedding_model.encode([query]).tolist()
    
    # 3. Retrieve relevant chunks (Retrieval)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=MAX_RETRIEVED_CHUNKS,
        include=['documents']
    )
    
    # Concatenate the retrieved documents/context
    relevant_chunks = results['documents'][0]
    context = "\n\n".join(relevant_chunks)
    
    print(f"[RETRIEVAL]: Found {len(relevant_chunks)} relevant chunks for context.")
    
    # 4. Generation (Prompting Gemini 2.5 Pro)
    
    system_instruction = (
        "You are an expert AI assistant for the Spinvic rag project. "
        "Your task is to answer the user's QUESTION based ONLY on the provided CONTEXT. "
        "Do not use external knowledge. If the CONTEXT does not contain the answer, state that you cannot answer based on the provided information."
    )
    
    prompt = f"CONTEXT:\n---\n{context}\n---\n\nQUESTION: {query}"
    
    # Prepare the contents for the API call
    contents = [types.Content(role='user', parts=[types.Part.from_text(text=prompt)])]
    
    # API Call to Gemini 2.5 Pro
    response = GEMINI_CLIENT.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2 # Use a low temperature for fact-based RAG
        )
    )
    
    return response.text.strip()


# --- 4. Main Execution Block ---

def main():
    """Main function to orchestrate the RAG pipeline."""
    
    # --- STEP 1: INDEXING (Run this the first time or when data changes) ---
    try:
        # build_vector_index() 
        print("NOTE: Indexing skipped. Uncomment 'build_vector_index()' to process PDF.")
    except Exception as e:
        print(f"Indexing failed: {e}")
        return

    # --- STEP 2: QUERYING (Run this after the index is built) ---
    
    # Example Query for the Spinvic rag
    query = "What are the core materials used in Spinvic rag?"
    
    try:
        answer = run_query(query)
        print("\n==================================================")
        print(f"🤖 Spinvic Rag AI Response for '{query}':")
        print("==================================================")
        print(answer)
        print("==================================================")
    except Exception as e:
        print(f"\n❌ ERROR during Query/Generation phase: {e}")


if __name__ == "__main__":
    main()