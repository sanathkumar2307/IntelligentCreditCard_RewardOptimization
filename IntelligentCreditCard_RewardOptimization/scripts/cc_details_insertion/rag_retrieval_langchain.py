import os
from pathlib import Path
from dotenv import load_dotenv
from rag_retrieval import get_matching_chunks,generate_query_embedding, query_rag_database_with_query
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from common local paths.
_HERE = Path(__file__).resolve()
_DOTENV_CANDIDATES = [
    _HERE.parents[1] / ".env",  # scripts/.env
    _HERE.parents[2] / ".env",  # repo root .env
]

for _env_path in _DOTENV_CANDIDATES:
    if _env_path.exists():
        load_dotenv(_env_path)
        break

# Also load default dotenv search path for compatibility.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please check your .env file.")

llm = ChatOpenAI(
    model="gpt-5.4-nano",
    api_key=OPENAI_API_KEY,
    temperature=0.0
)

def get_bank_card_list():
    query = """select DISTINCT card_name,issuer FROM
    cc_reward_chunks"""
    data = query_rag_database_with_query(query)
    # Extract card names and issuers into separate lists
    card_list = [row['card_name'] for row in data]
    bank_list = [row['issuer'] for row in data]

    return card_list, bank_list, data
    
def identify_bank_card(query: str) -> tuple:
    """
    Identifies which bank and credit card the user is asking about.
    Returns card name and bank name as separate variables.
    """

    card_list, bank_list, card_data = get_bank_card_list()
    print(f"Card List: {card_list}")
    print(f"Bank List: {bank_list}")
    system_prompt = (
        "You are a credit card and bank identifier. "
        "Your job is to identify from the user query which credit card and bank "
        "the user is asking about. You must pick from the provided lists. "
        "Return ONLY the card name and bank name separated by a comma. Do not explain."
        "please find the card name and bank name details available." 
        f"{card_data}"
    )

    user_prompt = f"""
    Available Cards:
    
    {', '.join(card_list)}

    Available Banks:
    {', '.join(bank_list)}

    User Query:
    "{query}"

    Return ONLY the card name and bank name in the format: "Card Name, Bank Name". Do not include anything else.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    response_text = response.content.strip()
    
    # Split the response into card_name and bank_name
    parts = [part.strip() for part in response_text.split(',')]
    card_name = parts[0] if len(parts) > 0 else ""
    bank_name = parts[1] if len(parts) > 1 else ""
    
    print(f"Identified Card: {card_name}, Bank: {bank_name}")
    return card_name, bank_name


def edit_query(query: str) -> str:
    prompt = f"""
    Expand this user query for retrieval.
    Add a few useful similar words.
    Return only one short expanded query.

    Query: {query}
    """
    return llm.invoke(prompt).content.strip()


def make_keywords(text, max_words=6):
    words = text.lower().replace("?", "").replace(",", "").replace(".", "").replace(')', '').replace('(', '').split()
    words = list(dict.fromkeys(words))   # remove duplicates, keep order
    # print(f"Keywords: {words[:max_words]}")
    return words

# query_text = "can you explain Fuel and Mileage policy?"
# edit_query(query)
# make_keywords(query)


def classify_intent(query: str, intent_list: list) -> str:
    """
    Classifies user intent into ONE label from intent_list.
    """

    system_prompt = (
        "You are an expert credit card query intent classification engine. "
        "if user didn't specified the booking channel then consider all channels else consider the channel specified by user. "
        "You must classify the user's query into exactly ONE intent "
        "from the provided list. "
        "Return ONLY the intent label. Do not explain."
    )

    user_prompt = f"""
    Intents:
    {', '.join(intent_list)}

    User Query:
    "{query}"

    Return only one intent from the list above.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    print(f"Classified intent: {response.content.strip()}")
    return response.content.strip()



def render_prompt(query: str, sources: str) -> str:
  PROMPT_TEMPLATE = """
    Imagine you're a financial assistant, your job is to give grounded answers using only the Context below:
    Make sure you don't answer from outside the given context, don't be robotic, and please answer in a friendly way.

    Context:
    {sources}

    Question:
    {query}
    """.strip()
  return PROMPT_TEMPLATE.format(query=query, sources=sources)



def build_context(results, max_chars: int = 1200) -> str:
    parts = []
    total_chars = 0

    for item in results:
        chunk = item["chunk_text"].strip()

        remaining = max_chars - total_chars
        if remaining <= 0:
            break

        if len(chunk) > remaining:
            chunk = chunk[:remaining]

        parts.append(chunk)
        total_chars += len(chunk)

    return "\n\n".join(parts)



# query = "can i reimburse fuel?"



# query = "I want compare my hdfc card with sbi card for fuel and mileage benefits"
# intent = classify_intent(query, query_type)

# print(intent)



def retrieve_docs(query_text, keywords=None, top_k=5, query_intent=None):
    # query_vector = generate_query_embedding(query_text)

    # if keywords:
    #     results = index.query(
    #         vector=query_vector,
    #         top_k=top_k,
    #         include_metadata=True,
    #         filter={"keywords": {"$in": keywords}}
    #     )
    # else:
    # card_list = get_bank_card_list()
    card_name, bank_name = identify_bank_card(query_text)
    if not card_name and not bank_name:
        results = get_matching_chunks(query_text, top_k=top_k, card_name=None, bank_name=None)
        print("No card or bank identified. Returning empty results.")
    else:
        results = get_matching_chunks(query_text, card_name=card_name, bank_name=bank_name)
        print(f"Retrieved {len(results)} results for card: {card_name}, bank: {bank_name}")



    # response = []
    # for match in results["matches"]:
    #     response.append({
    #         "id": match["id"],
    #         "score": float(match["score"]),
    #         "text": match["metadata"]["text"]
    #         # "links":
    #     })
# Home work - Put links from metadata in the response
    return results


query_type = ["Single transaction recommendation", "Monthly spend optimization", "Point transfer strategy", "Card comparison", "Missing information query"]
def _start(query):
  return {"query":query}

final_chain = (
    RunnableLambda(_start)
    .assign(edited_query=RunnableLambda(lambda d: edit_query(d["query"])))
    .assign(intent=RunnableLambda(lambda d: classify_intent(query=d["edited_query"], intent_list=query_type)))
    .assign(filter_keywords=RunnableLambda(lambda d: make_keywords(d["edited_query"], max_words=6)))
    .assign(results=RunnableLambda(lambda d: retrieve_docs(d["edited_query"], keywords=d["filter_keywords"] , query_intent=d["intent"], top_k=3)))
    .assign(sources = RunnableLambda(lambda d: build_context(d['results'], max_chars = 1200))) # {'sources':All sources combined}
    .assign(prompt = RunnableLambda(lambda d: render_prompt(d['edited_query'], d['sources']))) # {'prompt': 'updated prompt'}
    .assign(answer = RunnableLambda(lambda d: llm.invoke(d['prompt'])))
    .pick('answer') | StrOutputParser()
)

if __name__ == "__main__":
    
    query = input("Enter your query: ")
    final_output = final_chain.invoke(query)
    print(final_output)