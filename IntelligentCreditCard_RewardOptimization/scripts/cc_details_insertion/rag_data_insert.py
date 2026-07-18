import os
import time
import yaml
import numpy as np
import os
from typing import List, Dict

# from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# from langchain_google_genai import ChatGoogleGenerativeAI  # Commenting out due to version conflict
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pypdf import PdfReader

import os
import re
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please check your .env file.")
# pinecone_api_key = os.environ.get('PINECONE_API_KEY')

project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or "intelligent-cc-optimization"
location = "us"
processor_id = os.environ.get("DOCUMENT_AI_PROCESSOR_ID") or "3ccf5d1678ae6859"
bucket_name = "my_storage_files"
# file_name = "CC_project/CC_documents/AE_platinum_travel.pdf"
db_host = os.environ.get("DB_HOST") or "34.10.133.69" 
db_name = os.environ.get("DB_NAME") or "postgres"
db_user = os.environ.get("DB_USER") or "postgres"
db_pass = os.environ.get("DB_PASS") or "rag123"
file_path_local = ".\\data\\raw_pdfs\\"  # Local path for testing
local_files = [
    os.path.join(file_path_local, file_name)
    for file_name in os.listdir(file_path_local)
    if os.path.isfile(os.path.join(file_path_local, file_name))
]

# local_files = ["HDFC_diners_black.pdf"]




def read_pdf_pages(pdf_path: str) -> List[Dict[str, object]]:
    """
    Reads a PDF file and returns extracted text grouped by source page.
    """
    try:
        reader = PdfReader(pdf_path)
        page_entries: List[Dict[str, object]] = []

        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                page_entries.append({
                    "page_number": index,
                    "text": text,
                })

        return page_entries

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []

# Example Usage: gcs_uri=f"gs://{bucket_name}/{file_name}"
# document_text = read_pdf_text_only(".\\data\\raw_pdfs\\HDFC_diners_black.pdf")
# document_text = read_pdf_text_only(gcs_uri=f"gs://{bucket_name}/{file_name}")
# print(document_text)





def split_words_with_overlap(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start = end - overlap

    return chunks


def enforce_vertex_input_limits(chunks: List[str], max_chars_per_chunk: int = 12000) -> List[str]:
    """
    Ensures each text chunk is within a safe size for Vertex embedding token limits.
    Uses conservative char-based splitting to avoid 20k-token request failures.
    """
    limited_chunks: List[str] = []

    for chunk in chunks:
        if len(chunk) <= max_chars_per_chunk:
            limited_chunks.append(chunk)
            continue

        # Fallback split by characters to handle pathological texts with very long "words".
        start = 0
        while start < len(chunk):
            end = start + max_chars_per_chunk
            limited_chunks.append(chunk[start:end])
            start = end

    return limited_chunks

# chunk_text = split_words_with_overlap(document_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)



def generate_vertex_embeddings(project_id, location, text_list):
    """Generates 768-dimensional dense vector embeddings using text-embedding-004."""
    # If the location passed is 'us', override it to 'us-central1' for Vertex AI compatibility
    vertex_location = "us-central1" if location == "us" else location
    vertex_location = "europe-west1" if location == "eu" else vertex_location
    
    # Initialize using the valid regional endpoint string
    vertexai.init(project=project_id, location=vertex_location)
    
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    
    print(f"Generating vectors for {len(text_list)} text blocks via Vertex AI ({vertex_location})...")
    if not text_list:
        return []

    # Vertex AI limits:
    # - up to 250 instances per request
    # - total request token budget (error indicates 20,000 for this model/endpoint)
    max_instances_per_request = 250
    max_tokens_per_request = 18000  # safety margin below 20,000

    def estimate_tokens(text: str) -> int:
        # Conservative approximation for English-heavy text.
        return max(1, len(text) // 4)

    all_vectors = []
    current_batch = []
    current_batch_tokens = 0

    for text in text_list:
        text_tokens = estimate_tokens(text)

        # Flush current batch if adding this text would exceed request limits.
        if current_batch and (
            len(current_batch) >= max_instances_per_request
            or current_batch_tokens + text_tokens > max_tokens_per_request
        ):
            batch_embeddings = model.get_embeddings(current_batch)
            all_vectors.extend([emb.values for emb in batch_embeddings])
            current_batch = []
            current_batch_tokens = 0

        current_batch.append(text)
        current_batch_tokens += text_tokens

    if current_batch:
        batch_embeddings = model.get_embeddings(current_batch)
        all_vectors.extend([emb.values for emb in batch_embeddings])

    return all_vectors


def store_in_postgres(db_config, metadata, chunks, vectors):
    """Safely stores chunks, custom card tracking attributes, and vector arrays."""
    conn = psycopg2.connect(
        host=db_config['host'],
        database=db_config['dbname'],
        user=db_config['user'],
        password=db_config['password'],
        port=db_config['port']
    )
    cursor = conn.cursor()
    
    insert_document_query = """
    INSERT INTO tbl_document_details (
        document_id, card_name, card_issuer_bank, document_type,
        effective_date, source_url, uploaded_at
    ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP);
    """
    
    insert_query = """
    INSERT INTO cc_reward_chunks (
        document_id, card_name, issuer, document_type, 
        page_number, effective_date, source_url, chunk_text, embedding
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    
    print(f"Inserting entries cleanly into PostgreSQL table...")
    try:
        cursor.execute(insert_document_query, (
            metadata['document_id'],
            metadata['card_name'],
            metadata['issuer'],
            metadata['document_type'],
            metadata['effective_date'],
            metadata['source_url'],
        ))

        for idx, chunk in enumerate(chunks):
            cursor.execute(insert_query, (
                metadata['document_id'],
                metadata['card_name'],
                metadata['issuer'],
                metadata['document_type'],
                chunk['page_number'],
                metadata['effective_date'],
                metadata['source_url'],
                chunk['text'],
                vectors[idx]  # psycopg2 handles list to array translations natively for pgvector
            ))
        conn.commit()
        print("🚀 Successfully saved database assets!")
    except Exception as e:
        conn.rollback()
        print(f"Error executing database transactions: {e}")
    finally:
        cursor.close()
        conn.close()


# --- RUN RETRIEVAL PIPELINE ---
# if __name__ == "__main__":

def generate_and_store_embeddings(file_name, bucket_name, project_id, location, db_host, db_name, db_user, db_pass):
    """
    this function orchestrates the entire pipeline: reading a PDF, splitting it into chunks, generating embeddings, and storing them in PostgreSQL.
    """
    CHUNK_SIZE_WORDS = 500
    CHUNK_OVERLAP_WORDS = 75

    # BUCKET_NAME = "my_storage_files"
    # FILE_NAME = "CC_project/CC_documents/AE_platinum_travel.pdf"

    # 2. Extract Document Metadata Context attributes cleanly from naming conventions
    # e.g., 'AE_platinum_travel.pdf' -> issuer: 'AE', card_name: 'platinum_travel'
    base_file = os.path.basename(file_name)
    name_split = base_file.replace(".pdf", "").split("_", 1)

    inferred_issuer = name_split[0].upper() if len(name_split) > 0 else "UNKNOWN"
    inferred_card = name_split[1].replace("_", " ").title() if len(name_split) > 1 else "Generic Card"

    doc_metadata = {
        "document_id": f"DOC_{int(datetime.utcnow().timestamp())}", # Unique anchor ID string
        "card_name": inferred_card,
        "issuer": inferred_issuer,  # issuer_bank in database
        "document_type": "Terms and Conditions",
        "effective_date": datetime.utcnow().date().isoformat(),
        "source_url": f"https://storage.googleapis.com/{bucket_name}/{file_name}".replace("//", "/")
    }

    # 3. Cloud SQL Connection Configurations
    db_credentials = {
        "host": db_host, # Your public IP or 127.0.0.1 if running through Cloud SQL Proxy
        "dbname": db_name,
        "user": db_user,
        "password": db_pass,
        "port": "5432"
    }

    # --- PIPELINE WORKFLOW EXECUTION ---
    # Step A: Parse PDF and create text chunks
    extracted_pages = read_pdf_pages(file_name)
    extracted_characters = sum(len(page["text"]) for page in extracted_pages)
    print(f"Extracted {extracted_characters} characters from PDF")

    if extracted_pages:
        # Step B: Split each page into chunks while preserving source page metadata
        structured_chunks = []
        for page in extracted_pages:
            page_chunks = split_words_with_overlap(page["text"], CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
            page_chunks = enforce_vertex_input_limits(page_chunks)

            structured_chunks.extend([
                {
                    "text": chunk,
                    "page_number": page["page_number"]
                }
                for chunk in page_chunks
            ])

        text_chunks = [chunk["text"] for chunk in structured_chunks]
        print(f"Created {len(text_chunks)} chunks")

        # Step D: Generate embeddings
        computed_vectors = generate_vertex_embeddings(project_id, location, text_chunks)
        print(f"Generated {len(computed_vectors)} embeddings of dimension {len(computed_vectors[0])}")
        
        # Step E: Save to PostgreSQL Database with enhanced metadata
        store_in_postgres(db_credentials, doc_metadata, structured_chunks, computed_vectors)
    else:
        print("Pipeline aborted: Document text parsing step returned empty sequences.")


# --- RUN RETRIEVAL PIPELINE ---
if __name__ == "__main__":
    if not local_files:
        print(f"No files found in: {file_path_local}")
    else:
        print(f"Found {len(local_files)} file(s) in {file_path_local}")
        for input_file in local_files:
            print(f"Processing file: {input_file}")
            generate_and_store_embeddings(input_file, bucket_name, project_id, location, db_host, db_name,
                                        db_user, db_pass)