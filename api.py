import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

index = faiss.read_index("embeddings/index.faiss")
texts = np.load("embeddings/texts.npy", allow_pickle=True)
metadatas = np.load("embeddings/metadatas.npy", allow_pickle=True)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = genai.GenerativeModel("gemini-1.5-flash")

def query_rag(q: str):
    query_embedding = embedding_model.encode([q])
    D, I = index.search(np.array(query_embedding), 3)

    retrieved_chunks = []
    sources = []
    for idx, dist in zip(I[0], D[0]):
        if dist < 1.0:  # threshold
            retrieved_chunks.append(texts[idx])
            sources.append(metadatas[idx]["source"])

    if not retrieved_chunks:
        return {
            "answer": "The provided text does not contain information to answer your question.",
            "sources": []
        }

    prompt = "Answer the question based on the context below:\n\n"
    for i, chunk in enumerate(retrieved_chunks):
        prompt += f"[{sources[i]}]\n{chunk}\n\n"
    prompt += f"Question: {q}\nAnswer:"

    response = llm.generate_content(prompt)

    return {
        "answer": response.text,
        "sources parsed": list(set(sources))
    }

# Optional: run this file if you wanna run FastAPI only or run directly
if __name__ == "__main__":
    from fastapi import FastAPI, Query
    import uvicorn

    app = FastAPI()

    @app.get("/query")
    def query_endpoint(q: str = Query(...)):
        return query_rag(q)

    uvicorn.run(app, host="0.0.0.0", port=8000)