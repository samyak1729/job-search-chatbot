import pdfplumber
from langchain_community.document_loaders import Docx2txtLoader
import re
import os
import unicodedata
import cohere
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text: str) -> str:
    """
    Clean Unicode artifacts and fix run-together words.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\u2013", "-").replace("\u2022", "-").replace("\u00a7", "")
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def semantic_chunking(segments: list, cohere_client: cohere.Client, similarity_threshold: float = 0.85) -> list:
    """
    Group segments into chunks based on semantic similarity using Cohere embeddings.
    """
    if not segments:
        return []

    try:
        response = cohere_client.embed(texts=segments, model="embed-english-v3.0", input_type="search_document")
        embeddings = np.array(response.embeddings)
        print(f"Debug: Embedded {len(segments)} segments")
    except Exception as e:
        raise ValueError(f"Error embedding segments: {str(e)}")

    similarity_matrix = cosine_similarity(embeddings)
    chunks = []
    used = set()
    for i in range(len(segments)):
        if i in used:
            continue
        cluster = [segments[i]]
        used.add(i)
        for j in range(i + 1, len(segments)):
            if j not in used and similarity_matrix[i][j] > similarity_threshold:
                cluster.append(segments[j])
                used.add(j)
        chunks.append(" ".join(cluster).strip())
    return chunks

def parse_resume(file_path: str, cohere_api_key: str) -> dict:
    """
    Parse resume and create semantic chunks using Cohere embeddings.
    """
    extracted_data = {
        "chunks": [],
        "raw_text": "",
        "filename": os.path.basename(file_path)
    }

    try:
        # Read file content
        if file_path.endswith(".pdf"):
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            text = docs[0].page_content
        else:
            raise ValueError("Unsupported file format: only PDF or Word allowed")

        # Clean text
        cleaned_text = clean_text(text)
        extracted_data["raw_text"] = cleaned_text
        print(f"Debug: Cleaned text preview:\n{cleaned_text[:200]}...")

        # Split into segments
        lines = cleaned_text.split("\n")
        segments = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 3:
                segments.append(line)
        print(f"Debug: Extracted {len(segments)} segments from {file_path}")
        print(f"Debug: First few segments: {segments[:5]}")

        if not segments:
            print(f"Warning: No segments extracted from {file_path}")
            return extracted_data

        # Semantic chunking
        cohere_client = cohere.Client(cohere_api_key)
        chunks = semantic_chunking(segments, cohere_client)
        extracted_data["chunks"] = chunks
        print(f"Debug: Created {len(chunks)} chunks from {file_path}")

        return extracted_data

    except Exception as e:
        raise ValueError(f"Error parsing {os.path.basename(file_path)}: {str(e)}")

def validate_file(file_path: str) -> bool:
    """Validate file exists and is PDF or Word."""
    if not os.path.exists(file_path):
        return False
    if not file_path.endswith((".pdf", ".docx")):
        return False
    return True
