import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

index = faiss.read_index("embeddings/index.faiss")
texts = np.load("embeddings/texts.npy", allow_pickle=True)
metadatas = np.load("embeddings/metadatas.npy", allow_pickle=True)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llm = genai.GenerativeModel("gemini-1.5-flash")

def query_rag(question: str):
    embedding = embedding_model.encode([question])
    embedding = np.asarray(embedding, dtype=np.float32)

    faiss.normalize_L2(embedding)
    distances, indices = index.search(np.array(embedding), 3)

    context_chunks = [
        (texts[i], metadatas[i]["source"])
        for i, d in zip(indices[0], distances[0])
        if d < 1.0
    ]

    context = "\n\n".join(f"[{src}]\n{chunk}" for chunk, src in context_chunks)
    prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {question}\nAnswer:"

    response = llm.generate_content(prompt)

    return {
        "answer": response.text,
        "sources parsed": list({src for _, src in context_chunks}) if context_chunks else ["No relevant sources found."]
    }

# Optional: run this file if you wanna run FastAPI only or run directly
if __name__ == "__main__":
    from fastapi import FastAPI, Query
    from pydantic import BaseModel
    import uvicorn

    class QueryIn(BaseModel):
        q: str

    app = FastAPI()

    @app.post("/query")
    def query_endpoint(body: QueryIn):
        return query_rag(body.q)

    uvicorn.run(app, host="0.0.0.0", port=8000)