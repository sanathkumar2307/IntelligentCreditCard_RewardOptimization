import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from openai import OpenAI
from datetime import date

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please check your .env file.")

db_host = os.environ.get("DB_HOST") or "34.10.133.69" 
db_name = os.environ.get("DB_NAME") or "postgres"
db_user = os.environ.get("DB_USER") or "postgres"
db_pass = os.environ.get("DB_PASS") or "rag123"
model = os.environ.get("MODEL") or "gpt-4o-mini"


# Define Literals
SpendCategoryType = Literal["Dining", "Groceries", "Travel", "Gas & EV", "Entertainment", "Utilities", "Retail Shopping", "All Spend", "Exclusions"]
RewardType = Literal["Cashback", "Reward Points", "Air Miles", "Hotel Points", "Vouchers/Gift Cards", "Discounts", "None"]
RewardUnit = Literal["Percent (%)", "Multiplier (X)", "Points per Amount Spent", "Fixed Amount per Transaction", "Miles per Amount Spent", "Not Applicable"]
CapType = Literal["Monthly Billing Cycle", "Calendar Month", "Quarterly", "Annual", "Per Transaction", "Lifetime", "None"]
CapUnit = Literal["Rewards Earned", "Eligible Spend", "Not Applicable"]

class RewardRule(BaseModel):
    """Represents a single parsed credit card reward rule."""
    spend_category: SpendCategoryType = Field(description="The spend segment class.")
    reward_type: RewardType = Field(description="The nature of the reward asset accumulated.")
    
    # NUMERIC FIELD: Cannot be a string literal, typed as float
    reward_rate: Optional[float] = Field(None, description="The literal numeric rate or multiplier. E.g., 5 for 5%, 3 for 3x. Null if exclusion.")
    reward_unit: RewardUnit = Field(description="The scaling metric unit for the reward_rate.")
    
    cap_type: CapType = Field(description="The time/frequency cycle restriction constraint.")
    
    # NUMERIC FIELD: Cannot be a string literal, typed as float
    cap_value: Optional[float] = Field(None, description="The maximum limit threshold numeric value. Null if uncapped.")
    cap_unit: CapUnit = Field(description="Identifies if cap_value applies to the total Spend volume or total Rewards accrued.")
    
    exclusion_flag: bool = Field(description="True if this text block explicitly defines an ineligible category.")
    milestone_condition: Optional[str] = Field(None, description="Description of structural thresholds required (e.g. Spend $5000).")
    confidence_score: float = Field(description="LLM Extraction precision validation rating from 0.0 to 1.0.")

class RewardExtractionBatch(BaseModel):
    rules: List[RewardRule]



# ----------------------------------------------------------------------
# 3. Refactored Processing Pipeline
# ----------------------------------------------------------------------
def extract_and_migrate_rules():
    # Database Connection Configuration
    DB_PARAMS = {
        "dbname": db_name,
        "user": db_user,
        "password": db_pass,
        "host": db_host,
        "port": 5432
    }
    
    # Initialize Core Clients
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    
    print("🚀 Fetching raw text chunks from 'cc_reward_chunks'...")
    cursor.execute("""
        SELECT document_id, card_name, chunk_text 
        FROM cc_reward_chunks ;
    """)
    chunks = cursor.fetchall()
    print(f"📋 Found {len(chunks)} text chunks to process.\n")
    
    for doc_id, card_name, chunk_text in chunks:
        # doc_id is safely isolated as a string now
        print(f"Processing rules for Card: {card_name} (Doc ID: {doc_id})...")
        
        system_prompt = (
            "You are an expert financial analyst parsing credit card terms and conditions. "
            "Your job is to read the provided chunk of text and extract all credit card spending benefit rules "
            "and exclusions into structured objects matching the provided schema. "
            "You must map values strictly into the enumerated categorical strings allowed by the schema."
            "You have to give and rate your confidence score for each rule extracted based on information available for rule creation and "
            "after extraction how well the rule can be utilized. "
        )
        
        user_prompt = f"Card Name: {card_name}\n\nText Chunk:\n{chunk_text}"
        
        try:
            # Leverage OpenAI Structured Outputs Engine
            completion = openai_client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=RewardExtractionBatch,
            )
            
            extracted_data = completion.choices[0].message.parsed
            
            if not extracted_data or not extracted_data.rules:
                print(f"⚠️ No distinct rules detected inside text chunk for Doc ID: {doc_id}")
                continue
                
            # Pack extracted rules into tuples for database insertion
            insert_rows = []
            for rule in extracted_data.rules:
                insert_rows.append((
                    card_name,
                    rule.spend_category,
                    rule.reward_type,
                    rule.reward_rate,
                    rule.reward_unit,
                    rule.cap_type,
                    rule.cap_value,
                    rule.cap_unit,       # <-- Added new column variable
                    rule.exclusion_flag,
                    rule.milestone_condition,
                    doc_id,              # <-- Safely mapped as string
                    rule.confidence_score
                ))
            
            # Updated Insert Query to accept the new structural changes
            insert_query = """
                INSERT INTO tbl_reward_rules (
                    card_name, spend_category, reward_type, reward_rate, reward_unit, 
                    cap_type, cap_value, cap_unit, exclusion_flag, milestone_condition, 
                    source_document_id, confidence_score
                ) VALUES %s;
            """
            
            execute_values(cursor, insert_query, insert_rows)
            conn.commit()
            print(f"✅ Successfully wrote {len(insert_rows)} rules to 'tbl_reward_rules'.")
            
        except Exception as e:
            conn.rollback()
            print(f"❌ Failed to process Doc ID {doc_id} due to error: {e}")
            
    # Connection cleanup
    cursor.close()
    conn.close()
    print("\n🏁 Extraction process completed successfully.")


PartnerType = Literal["Airline", "Hotel", "Retail/Shopping", "Other", "Not Applicable"]

class TransferPartnerRule(BaseModel):
    """Represents a single parsed loyalty transfer partner contract profile."""
    partner_name: str = Field(description="The formal brand name of the reward partner (e.g., Emirates Skywards, Marriott Bonvoy).")
    partner_type: PartnerType = Field(description="The business category classification of the alliance partner.")
    
    transfer_ratio: str = Field(
        description="The mathematical exchange conversion ratio pattern string. Format it as 'X:Y' (e.g., '1:1', '4:5')."
    )
    
    minimum_transfer_points: Optional[int] = Field(
        None, description="The smallest initial milestone threshold of card points needed to convert (e.g., 1000). Null if omitted."
    )
    maximum_transfer_points: Optional[int] = Field(
        None, description="The ceiling cap limit value allowed to change per transaction. Null if uncapped or unknown."
    )
    
    # Defaults to today's date if the context document provides no alternate clarity timeline
    effective_date: Optional[date] = Field(
        None, description="The structural baseline date when this rule becomes active. Use YYYY-MM-DD if available, else null."
    )
    confidence_score: float = Field(description="LLM compliance validity evaluation rating value scaling from 0.0 to 1.0.")


class PartnerExtractionBatch(BaseModel):
    """Container allowing the LLM to return multiple partner profiles out of a single text block."""
    partners: List[TransferPartnerRule]


# ----------------------------------------------------------------------
# 2. Main Processing Core Pipeline
# ----------------------------------------------------------------------
def extract_and_migrate_partners():
    # Database Connection Configuration parameters
    DB_PARAMS = {
        "dbname": db_name,
        "user": db_user,
        "password": db_pass,
        "host": db_host,
        "port": 5432
    }
    
    # Initialize Core Clients
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    
    print("🚀 Fetching raw text chunks from 'cc_reward_chunks'...")
    cursor.execute("""
        SELECT document_id, card_name, chunk_text 
        FROM cc_reward_chunks;
    """)
    chunks = cursor.fetchall()
    print(f"📋 Found {len(chunks)} text chunks to evaluate.\n")
    
    for doc_id, card_name, chunk_text in chunks:
        print(f"Analyzing partner details for Card: {card_name} (Doc ID: {doc_id})...")
        
        system_prompt = (
            "You are an expert financial logistics analyst tracking credit card loyalty programs. "
            "Your job is to read the provided text chunk and extract all active points conversion transfer alliance partners, "
            "their ratios, and their structural transaction boundary controls into precise structured output objects."
            "You have to give and rate your confidence score for each partner details extracted based on information available about partners "
            "after extraction how well the partner details can be utilized. "
        )
        
        user_prompt = f"Card Name: {card_name}\n\nText Chunk:\n{chunk_text}"
        
        try:
            # Constrained decoding request call
            completion = openai_client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=PartnerExtractionBatch,
            )
            extracted_data = completion.choices[0].message.parsed
            
            if not extracted_data or not extracted_data.partners:
                print(f"ℹ️ No transfer partner profiles identified inside text chunk for Doc ID: {doc_id}")
                continue
                
            # Compile variables into tuples
            insert_rows = []
            today_fallback = date.today()
            
            for partner in extracted_data.partners:
                insert_rows.append((
                    card_name,
                    partner.partner_name,
                    partner.partner_type,
                    partner.transfer_ratio,
                    partner.minimum_transfer_points,
                    partner.maximum_transfer_points,
                    partner.effective_date if partner.effective_date else today_fallback,
                    doc_id,  # Safely mapped into our VARCHAR field
                    partner.confidence_score
                ))
            
            # Execute standard bulk transactions
            insert_query = """
                INSERT INTO tbl_transfer_partners (
                    card_name, partner_name, partner_type, transfer_ratio, 
                    minimum_transfer_points, maximum_transfer_points, effective_date, 
                    source_chunk_id, confidence_score
                ) VALUES %s;
            """
            
            execute_values(cursor, insert_query, insert_rows)
            conn.commit()
            print(f"✅ Successfully written {len(insert_rows)} alliance partner entries to 'tbl_transfer_partners'.")
            
        except Exception as e:
            conn.rollback()
            print(f"❌ Failed to process Doc ID {doc_id} due to error: {e}")
            
    # Connection memory cleanup handling
    cursor.close()
    conn.close()
    print("\n🏁 Extraction process completed successfully.")

if __name__ == "__main__":
#     extract_and_migrate_partners()

    extract_and_migrate_rules()
    extract_and_migrate_partners()