## Module 3 - Support Assistant

The Support Assistant is a policy-aware GenAI application built using LangGraph, ChromaDB, Sentence Transformers, Gemini API, and FastAPI.

### Features

- Policy document ingestion
- Semantic document retrieval
- Policy-based question routing
- Grounded AI responses
- Source document identification
- FastAPI REST API
- Web-based support interface

### Run

python support_assistant/ingest.py

uvicorn support_assistant.app:app --port 8001

Open http://127.0.0.1:8001