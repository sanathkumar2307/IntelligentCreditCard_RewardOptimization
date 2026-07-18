import os
from dotenv import load_dotenv
from rag_retrieval import get_matching_chunks,generate_query_embedding
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please check your .env file.")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.0
)



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
        "You are an intent classification engine. "
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

    return response.content.strip()



def render_prompt(query: str, sources: str) -> str:
  PROMPT_TEMPLATE = """
Imagine you're a financial assitant, your job is to give grounded answers using only the Context below:
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

query_type = ["Single transaction recommendation", "Monthly spend optimization", "Point transfer strategy", "Card comparison", "Missing information query"]

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
    results = get_matching_chunks(query_text)



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