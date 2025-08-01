import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.document_loader import extract_text_from_file

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)
texts = []
metadatas = []

for filename in os.listdir("docs"):
    path = f"docs/{filename}"
    text = extract_text_from_file(path)

    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    embeddings = model.encode(chunks)

    index.add(np.array(embeddings))
    texts.extend(chunks)

    metadatas.extend([{"source": filename}] * len(chunks))

faiss.write_index(index, "index.faiss")
np.save("texts.npy", texts)
np.save("metadatas.npy", metadatas)