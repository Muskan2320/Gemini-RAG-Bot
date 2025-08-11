import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter

from utils.document_loader import extract_text_from_file

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatIP(384)
texts = []
metadatas = []

splitter = CharacterTextSplitter(
    separator=" ",     
    chunk_size=500,   
    chunk_overlap=50,
    length_function=len
)

for filename in os.listdir("docs"):
    path = f"docs/{filename}"
    text = extract_text_from_file(path)

    if not text or not text.strip():
        continue

    chunks = splitter.split_text(text)
    embeddings = model.encode(chunks)
    
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)

    index.add(np.array(embeddings))
    texts.extend(chunks)

    metadatas.extend([{"source": filename}] * len(chunks))

faiss.write_index(index, "embeddings/index.faiss")
np.save("embeddings/texts.npy", texts)
np.save("embeddings/metadatas.npy", metadatas)