# File reader for pdfs, docx and text files

import fitz
import docx
from pathlib import Path

def extract_text_from_file(filepath):
    ext = Path(filepath).suffix.lower()

    if ext == ".pdf":
        doc = fitz.open(filepath)

        return "\n".join([page.get_text() for page in doc])
    elif ext == ".docx":
        doc = docx.Document(filepath)

        return "\n".join([para.text for para in doc.paragraphs])
    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
        
    return ""