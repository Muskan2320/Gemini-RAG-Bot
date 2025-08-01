# Gemini RAG Chatbot 🧠💬

A simple RAG-based chatbot powered by Google's Gemini Pro. It allows you to query a local document collection and get answers with source references using embeddings, FAISS vector search, and a clean Streamlit UI.

---

## 🚀 Features

- Ask questions about your documents
- Uses `SentenceTransformers` for embeddings
- FAISS for fast similarity search
- Gemini Pro as the answering engine
- Clean and interactive Streamlit interface

---

## 🗂 Project Structure

```
gemini_rag_bot/
├── app.py                  # Streamlit UI
├── api.py                  # Gemini RAG logic
├── embeddings/
│   ├── index.faiss
│   ├── texts.npy
│   └── metadatas.npy
├── utils/
│   └── document_loader.py
├── docs/                   # Source PDFs, TXT, DOCX (optional)
├── index_documents.py      # Builds embeddings/index
├── .env                    # Local Gemini API key (for dev only)
├── requirements.txt
```

---

## 🧪 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/your-username/gemini-rag-bot.git
cd gemini-rag-bot
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your documents
Place `.pdf`, `.txt`, or `.docx` files inside the `docs/` folder.

### 5. Generate embeddings and FAISS index (if not downloaded)
```bash
python embeddings/index_documents.py
```

### 6. Set your Gemini API Key
Create a `.env` file in the root folder with:
```
GEMINI_API_KEY=your-api-key-here
```

### 7. Run the chatbot UI
```bash
streamlit run app.py
```

Go to `http://localhost:8501` to chat with your documents!

---

### 8. Run the API server (Optional)
```bash
uvicorn api:app --reload
```

Go to `http://localhost:8000/docs` to test the API using Swagger UI,  
or access `/query?q=your_question` directly.

## 🛠 Notes

- All embeddings are stored in the `embeddings/` folder
- Only top 3 most relevant document chunks are used in each answer

---