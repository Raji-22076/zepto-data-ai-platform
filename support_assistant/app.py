import os
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from langgraph.graph import StateGraph, END
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

DB_PATH = "support_assistant/chroma_db"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="zepto_policies")

gemini_client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

MODEL_NAME = "gemini-3.5-flash-lite"

policy_map = {
    "delivery": "doc_01",
    "return": "doc_02",
    "refund": "doc_02",
    "membership": "doc_03",
    "pass": "doc_03",
    "tracking": "doc_04",
    "cancel": "doc_05",
    "damaged": "doc_06",
    "spoiled": "doc_06",
    "missing": "doc_06",
    "gift card": "doc_07",
    "giftcard": "doc_07",
    "support": "doc_08"
}

fallback_answers = {
    "doc_01": "Zepto delivery typically takes 10–30 minutes. Standard delivery is free for orders above ₹149 and costs ₹25 for orders below ₹149. Priority delivery costs ₹15.",
    "doc_02": "Perishable items can be returned within 24 hours if they are damaged, spoiled, or incorrect. Non-perishable items can be returned within 7 days if unopened. Refunds usually take 3–5 business days, while wallet refunds are instant.",
    "doc_03": "Zepto has Basic, Zepto Pass, and Zepto Pass+ membership tiers. Zepto Pass costs ₹49 per month and Zepto Pass+ costs ₹99 per month. Membership can be cancelled according to the applicable membership terms.",
    "doc_04": "Live rider tracking is available for orders. If tracking shows no movement for more than 20 minutes past the original delivery estimate, contact support.",
    "doc_05": "Orders can usually be cancelled before they are packed, typically within the first 2 minutes. Once an order is packed, cancellation is generally not available.",
    "doc_06": "Damaged, spoiled, or missing items should be reported within 24 hours. Eligible items can receive a replacement or refund. For orders above ₹1000, a photo may be required.",
    "doc_07": "Zepto gift cards are available in ₹100, ₹250, ₹500, and ₹1000 denominations. They are valid for 1 year and cannot be combined with other gift cards.",
    "doc_08": "Zepto support is available 24/7 through in-app chat. The average response time is under 2 minutes. Email support is available with responses within 24 business hours."
}

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

    if any(word in question for word in policy_map):
        return {"category": "policy"}

    return {"category": "general"}

def retrieve_policy(state: AssistantState):
    question = state["question"].lower()

    selected_id = None

    for keyword, doc_id in policy_map.items():
        if keyword in question:
            selected_id = doc_id
            break

    if selected_id:
        result = collection.get(ids=[selected_id])
        documents = result["documents"]
        ids = result["ids"]
    else:
        query_embedding = embedding_model.encode([question]).tolist()

        result = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )

        documents = result["documents"][0]
        ids = result["ids"][0]

    return {
        "context": "\n\n".join(documents),
        "sources": ids
    }

def generate_policy_answer(state: AssistantState):
    try:
        prompt = f"""
You are a Zepto customer support assistant.

Answer the user's question using ONLY the policy document below.

Give a direct and concise answer.

Do not say that the information is unavailable when the document contains the answer.

Policy document:
{state["context"]}

User question:
{state["question"]}
"""

        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        answer = (response.text or "").strip()

        if not answer or "could not find this information" in answer.lower():
            answer = fallback_answers.get(
                state["sources"][0],
                state["context"]
            )

    except Exception:
        answer = fallback_answers.get(
            state["sources"][0],
            state["context"]
        )

    return {
        "answer": answer
    }

def generate_general_answer(state: AssistantState):
    try:
        response = gemini_client.models.generate_content(
            model=MODEL_NAME,
            contents=state["question"]
        )

        answer = (response.text or "").strip()

        if not answer:
            answer = "Please ask a more specific question."

    except Exception:
        answer = "The AI service is temporarily unavailable. Please try again later."

    return {
        "answer": answer,
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

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Zepto Support Assistant</title>
<style>
body {
    font-family: Arial, sans-serif;
    max-width: 800px;
    margin: 40px auto;
    padding: 20px;
}
h1 {
    text-align: center;
}
textarea {
    width: 100%;
    height: 100px;
    padding: 12px;
    box-sizing: border-box;
}
button {
    margin-top: 12px;
    padding: 12px 24px;
    cursor: pointer;
}
#result {
    margin-top: 25px;
    padding: 20px;
    border: 1px solid #ccc;
    white-space: pre-wrap;
}
</style>
</head>
<body>
<h1>Zepto Support Assistant</h1>
<textarea id="question" placeholder="Ask a question about delivery, returns, refunds, membership, tracking, cancellation, gift cards or support"></textarea>
<br>
<button onclick="askQuestion()">Ask</button>
<div id="result"></div>

<script>
async function askQuestion() {
    const question = document.getElementById("question").value;
    const result = document.getElementById("result");

    if (!question.trim()) {
        result.innerText = "Please enter a question.";
        return;
    }

    result.innerText = "Thinking...";

    const response = await fetch("/ask", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            question: question
        })
    });

    const data = await response.json();

    result.innerText =
        data.answer +
        "\\n\\nCategory: " +
        data.category +
        "\\nSources: " +
        data.sources.join(", ");
}
</script>
</body>
</html>
"""

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