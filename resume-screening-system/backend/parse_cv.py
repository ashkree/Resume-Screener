# Parses PDF and DOCX files to extract text for resume processing

from pdfminer.high_level import extract_text
from docx import Document
import os

def parse_cv(filepath):
    # Extract text based on file extension
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        try:
            text = extract_text(filepath)
        except Exception as e:
            text = f"Error extracting PDF text: {e}"
    elif ext == ".docx":
        try:
            doc = Document(filepath)
            paragraphs = [para.text for para in doc.paragraphs]
            text = "\n".join(paragraphs)
        except Exception as e:
            text = f"Error extracting DOCX text: {e}"
    else:
        text = "Unsupported file format for parsing. Please upload PDF or DOCX."

    return text
