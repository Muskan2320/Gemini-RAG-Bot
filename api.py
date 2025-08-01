from fastapi import FastAPI, Query
import faiss
import numpy as np
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

index = faiss.read_index("embeddings/index.faiss")
texts = np.load("embeddings/texts.npy", allow_pickle=True)
metadatas = np.load("embeddings/metadatas.npy", allow_pickle=True)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()

@app.get("/query")
def query_rag(q: str = Query(...)):
    query_embedding = embedding_model.encode([q])
    D, I = index.search(np.array(query_embedding), 3)
    
    retrieved_chunks = [texts[i] for i in I[0]]
    sources = [metadatas[i]["source"] for i in I[0]]

    prompt = "Answer the question based on the context below:\n\n"
    for i, chunk in enumerate(retrieved_chunks):
        prompt += f"[{sources[i]}]\n{chunk}\n\n"

    prompt += f"Question: {q}\nAnswer:"
    response = llm.generate_content(prompt)

    return {
        "answer": response.text,
        "sources parsed": list(set(sources))
    }