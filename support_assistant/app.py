import os
from typing import TypedDict
from unittest import result
from fastapi.responses import FileResponse
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from langgraph.graph import StateGraph, END
from fastapi import FastAPI
from pydantic import BaseModel

DB_PATH = "support_assistant/chroma_db"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="zepto_policies")

gemini_client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

class AssistantState(TypedDict):
    question: str
    category: str
    context: str
    answer: str
    sources: list[str]

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    category: str
    sources: list[str]

def route_question(state: AssistantState):
    question = state["question"].lower()

    policy_words = [
        "return",
        "refund",
        "delivery",
        "cancel",
        "membership",
        "gift card",
        "support",
        "damaged",
        "missing",
        "order",
        "pass",
        "tracking"
    ]

    category = "policy" if any(word in question for word in policy_words) else "general"

    return {"category": category}

def retrieve_policy(state):
    question = state["question"].lower()

    if any(word in question for word in ["delivery", "deliver", "minutes", "time"]):
        doc_ids = ["doc_01"]
    elif any(word in question for word in ["return", "refund", "damaged", "spoiled"]):
        doc_ids = ["doc_02", "doc_06"]
    elif any(word in question for word in ["membership", "pass", "subscription"]):
        doc_ids = ["doc_03"]
    elif any(word in question for word in ["tracking", "rider", "track", "location"]):
        doc_ids = ["doc_04"]
    elif any(word in question for word in ["cancel", "cancellation"]):
        doc_ids = ["doc_05"]
    elif any(word in question for word in ["gift card", "giftcard"]):
        doc_ids = ["doc_07"]
    elif any(word in question for word in ["support", "contact", "help", "chat"]):
        doc_ids = ["doc_08"]
    else:
        result = collection.query(
            query_embeddings=[embedding_model.encode(question).tolist()],
            n_results=3
        )
        documents = result["documents"][0]
        sources = result["ids"][0]
        return {
            "context": "\n\n".join(documents),
            "sources": sources
        }

    result = collection.get(
    ids=doc_ids,
    include=["documents"]
    )

    print("RETRIEVED DOCUMENTS:", result["documents"])

    documents = result["documents"]
    sources = result["ids"]

    return {
        "context": "\n\n".join(documents),
        "sources": sources
    }
def generate_policy_answer(state):
    context = state["context"]
    question = state["question"]
    print("QUESTION:", question)
    print("CONTEXT:", context)

    prompt = f"""
You are a Zepto customer support assistant.

Answer the user's question using ONLY the policy information provided below.

Policy information:
{context}

User question:
{question}

Give a direct and helpful answer.
Do not say that the information is unavailable if the answer is present in the policy information.
Do not add information that is not present in the policy information.
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return {"answer": response.text}

def generate_general_answer(state: AssistantState):
    prompt = f"""
You are a helpful Zepto support assistant.

Answer the user's general question clearly and briefly.

Do not invent specific Zepto policies.

User question:
{state["question"]}
"""

    response = gemini_client.models.generate_content(
        model="gemini-3.7-flash",
        contents=prompt
    )

    return {
        "answer": response.text,
        "sources": []
    }

def choose_path(state: AssistantState):
    if state["category"] == "policy":
        return "retrieve_policy"

    return "generate_general"

graph = StateGraph(AssistantState)

graph.add_node("route", route_question)
graph.add_node("retrieve_policy", retrieve_policy)
graph.add_node("generate_policy", generate_policy_answer)
graph.add_node("generate_general", generate_general_answer)

graph.set_entry_point("route")

graph.add_conditional_edges(
    "route",
    choose_path,
    {
        "retrieve_policy": "retrieve_policy",
        "generate_general": "generate_general"
    }
)

graph.add_edge("retrieve_policy", "generate_policy")
graph.add_edge("generate_policy", END)
graph.add_edge("generate_general", END)

assistant_graph = graph.compile()

app = FastAPI(
    title="Zepto Support Assistant",
    version="1.0"
)

@app.get("/")
def home():
    return FileResponse("support_assistant/index.html")

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = assistant_graph.invoke({
        "question": request.question,
        "category": "",
        "context": "",
        "answer": "",
        "sources": []
    })

    return AskResponse(
        answer=result["answer"],
        category=result["category"],
        sources=result["sources"]
    )