import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# File paths
DATA_PATH = "data/FAQ_Bank.csv"
FAISS_INDEX_PATH = "data/faiss_index.bin"
EMBEDDINGS_PATH = "data/faq_embeddings.npy"

# Load dataset
df = pd.read_csv(DATA_PATH).dropna(subset=["question", "answer"]).reset_index(drop=True)
df["question"] = df["question"].astype(str)
df["answer"] = df["answer"].astype(str)

# Load Embedding Model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    """Convert text into vector embedding."""
    return model.encode(text, convert_to_numpy=True)

# Load or Generate Embeddings
if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(FAISS_INDEX_PATH):
    print("Using stored embeddings & FAISS index.")
    embeddings = np.load(EMBEDDINGS_PATH)
    index = faiss.read_index(FAISS_INDEX_PATH)
else:
    print("Generating new embeddings...")
    embeddings = np.array([get_embedding(q) for q in tqdm(df["question"], desc="Generating Embeddings")])
    np.save(EMBEDDINGS_PATH, embeddings)

    # Store embeddings in FAISS
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"Embeddings saved to {EMBEDDINGS_PATH}")
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

print("Ready to use stored embeddings!")

def search_faq(query, top_k=3):
    """Find the most relevant FAQs from the dataset using Cosine Similarity."""
    query_embedding = get_embedding(query).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings)
    indices = np.argsort(similarities[0])[::-1][:top_k]

    results = []
    for idx in indices:
        if 0 <= idx < len(df):
            faq = df.iloc[idx]
            results.append({"question": faq["question"], "answer": faq["answer"]})

    return results