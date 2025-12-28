# Real Estate GenAI (Hyderabad) — ML + RAG + Gemini

Prototype that predicts Hyderabad house prices and generates an investment explanation using RAG + Gemini.

## What it does
- **Stage 1 (ML):** Train and compare **Linear Regression** vs **XGBoost** using **RMSE/MAE**
- **Stage 2 (RAG):** Ingest legal PDFs + market/news text → chunk + embed → store in **ChromaDB** → retrieve Top-K chunks
- **Stage 3 (Reasoning):** Gemini produces structured narrative (drivers, risks, assumptions) grounded in retrieved chunks
- **Stage 5 (API):** FastAPI returns **JSON**

**Scope:** 3k+ rows + small doc corpus (focus on output quality over latency).

---

## Setup
- Install scikit-learn, xgboost, google-generativeai, chromadb, sentence-transformers, fastapi, uvicorn, python-dotenv, PyPDF2 if not present
- Configure your API key accordingly in .env
- Launch the API using "uvicorn src.ui_api:app --reload"

## Design decisions
Detailed design decisions regarding model selection, chunking strategy, prompt, hallucination safeguards and outputs are documented in the Technical Report
