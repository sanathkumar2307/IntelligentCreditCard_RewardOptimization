import psycopg2
import vertexai
from vertexai.language_models import TextEmbeddingModel
import os
from psycopg2.extras import RealDictCursor


project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or "intelligent-cc-optimization"
location = "us"
processor_id = os.environ.get("DOCUMENT_AI_PROCESSOR_ID") or "3ccf5d1678ae6859"
bucket_name = "my_storage_files"
file_path = "CC_project/CC_documents/"


db_host = os.environ.get("DB_HOST") or "34.10.133.69" 
db_name = os.environ.get("DB_NAME") or "postgres"
db_user = os.environ.get("DB_USER") or "postgres"
db_pass = os.environ.get("DB_PASS") or "rag123"


def generate_query_embedding(project_id, location, query_text):
    """Converts the user's question into a 768-dimension semantic vector."""
    # Handle the Vertex AI precise geographic region requirement
    vertex_location = "us-central1" if location == "us" else location
    vertex_location = "europe-west1" if location == "eu" else vertex_location
    
    vertexai.init(project=project_id, location=vertex_location)
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    
    print(f"Embedding user query: '{query_text}'...")
    embeddings = model.get_embeddings([query_text])
    return embeddings[0].values

def query_rag_database(query_vector, limit=5):
    """
    Queries Cloud SQL using Cosine Distance (<=>) to find the top matching chunks.
    Calculates exact Cosine Similarity from the Cosine Distance layer.
    """
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_pass,
        port="5432"
    )
    cursor = conn.cursor()
    
    # 1 - (embedding <=> %s) converts cosine distance directly into absolute similarity %
    search_sql = """
    SELECT 
        document_id,
        card_name,
        issuer,
        document_type,
        page_number,
        effective_date,
        source_url,
        chunk_text,
        1 - (embedding <=> %s::vector) AS cosine_similarity
    FROM cc_reward_chunks
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
    """
    
    results = []
    try:
        # Pass the query vector twice: once for the similarity calculation, once for sorting
        cursor.execute(search_sql, (query_vector, query_vector, limit))
        rows = cursor.fetchall()
        
        for row in rows:
            results.append({
                "document_id": row[0],
                "card_name": row[1],
                "issuer": row[2],
                "document_type": row[3],
                "page_number": row[4],
                "effective_date": str(row[5]),
                "source_url": row[6],
                "chunk_text": row[7],
                "similarity": round(row[8] * 100, 2)  # Convert to percentage format (e.g. 87.45%)
            })
    except Exception as e:
        print(f"Database retrieval failed: {e}")
    finally:
        cursor.close()
        conn.close()
        
    return results


def query_rag_database_with_query(sql_query, query_params=None):
    """
    Execute a read-only SQL query on remote PostgreSQL and return rows as dictionaries.

    Args:
        sql_query (str): SQL query text (must start with SELECT).
        query_params (tuple | list | None): Optional bind parameters for safe placeholders.

    Returns:
        list[dict]: Query results as a list of row dictionaries.
    """
    if not isinstance(sql_query, str) or not sql_query.strip():
        raise ValueError("sql_query must be a non-empty string")

    if not sql_query.lstrip().lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed")

    conn = None
    rows = []
    try:
        conn = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_pass,
            port="5432"
        )
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql_query, query_params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        print(f"Database retrieval failed: {e}")
        return []
    finally:
        if conn is not None:
            conn.close()


def get_matching_chunks(user_query, limit=3):
    """
    Main function to retrieve the most relevant context chunks for a given user query.
    Returns a list of structured metadata for each matched chunk.
    """
    # 1. Generate the query embedding
    query_vector = generate_query_embedding(project_id, location, user_query)
    
    # 2. Query the database for top matching chunks
    matched_chunks = query_rag_database(query_vector, limit=limit)
    
    return matched_chunks

# --- RUN RETRIEVAL PIPELINE ---
if __name__ == "__main__":
    # 1. Configs
    GCP_PROJECT = "intelligent-cc-optimization"
    GCP_LOCATION = "us" 
    
    db_credentials = {
        "host": db_host, # Change to 127.0.0.1 if tunneling via Cloud SQL Proxy
        "dbname": db_name,
        "user": db_user,
        "password": db_pass,
        "port": "5432"
    }
    
    # 2. Define the user question from user input or a static example
    user_query = input("Enter your question: ") or "What lounge access or travel insurance benefits do I get with the Platinum Travel card?"
    
    # 3. Generate query vector matching your text-embedding-004 structure
    # query_vector = generate_query_embedding(GCP_PROJECT, GCP_LOCATION, user_query)
    
    # # 4. Extract matched context blocks from Cloud SQL
    # matched_contexts = query_rag_database(query_vector, limit=3)

    matched_contexts = get_matching_chunks(user_query, limit=3)
    
    # 5. Output the structured metadata details cleanly
    print(f"\n🚀 Found {len(matched_contexts)} highly relevant context chunks:\n")
    for rank, context in enumerate(matched_contexts, 1):
        print(f"--- MATCH RANK #{rank} ({context['similarity']}% Match Score) ---")
        print(f"📍 Card Context: {context['issuer']} - {context['card_name']} ({context['document_type']})")
        print(f"📄 Page Reference: Page {context['page_number']} | Effective: {context['effective_date']}")
        print(f"🔗 Source Asset: {context['source_url']}")
        print(f"📝 Text Chunk:\n{context['chunk_text']}\n")