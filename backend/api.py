from fastapi import FastAPI, HTTPException
import pandas as pd
import faiss
import numpy as np
from backend.faiss_db import search_faq, get_embedding
from backend.ai_response import improve_answer
from pydantic import BaseModel

app = FastAPI()

#Load dataset
try:
    df = pd.read_csv("data/FAQ_Bank.csv")
    print(f"Dataset loaded: {len(df)} rows")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Dataset file not found!")

#Load FAISS index
try:
    index = faiss.read_index("data/faiss_index.bin")
    print("FAISS index loaded successfully.")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="FAISS index file not found!")

#Define request model
class ChatRequest(BaseModel):
    query: str
    history: list = []

@app.get("/")
def home():
    """Root endpoint to check if API is running"""
    return {"message": "COVID-19 FAQ API is running!"}

@app.post("/ask")
def ask_question(request: ChatRequest):
    """Retrieve the best FAQ answer using similarity search and AI enhancement."""
    
    query = request.query.strip()
    history = request.history

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty!")

    #Get FAQs from FAISS
    results = search_faq(query)
    
    if not results:  # No FAQ found
        return {
            "question": query,
            "original_answer": "Information not available.",
            "enhanced_answer": "Sorry, this information is not available in the provided dataset."
        }

    best_faq = results[0]

    if "answer" not in best_faq or not best_faq["answer"]:
        return {
            "question": query,
            "original_answer": "Information not available.",
            "enhanced_answer": "Sorry, this information is not available in the provided dataset."
        }

    #Improve the answer using AI
    improved_answer = improve_answer(query, best_faq["answer"], history)

    #Store last 5 interactions for context
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": improved_answer})
    
    if len(history) > 10:  
        history = history[-10:]  # Keep only last 5 interactions 

    return {
        "question": best_faq["question"],
        "original_answer": best_faq["answer"],
        "enhanced_answer": improved_answer
    }