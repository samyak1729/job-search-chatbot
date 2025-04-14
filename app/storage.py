import cohere
import pinecone
from pinecone import Pinecone, ServerlessSpec
import os
import hashlib

def initialize_pinecone(pinecone_api_key: str, index_name: str = "resume-chunks"):
    """
    Initialize Pinecone and create index if it doesn't exist.
    """
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created Pinecone index: {index_name}")
        else:
            print(f"Using existing Pinecone index: {index_name}")
        return pc.Index(index_name)
    except Exception as e:
        raise ValueError(f"Error initializing Pinecone: {str(e)}")

def store_chunks(chunks: list, filename: str, cohere_api_key: str, pinecone_index, namespace: str = "resumes"):
    """
    Embed chunks with Cohere and store in Pinecone.
    """
    try:
        cohere_client = cohere.Client(cohere_api_key)
        response = cohere_client.embed(
            texts=chunks,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        embeddings = response.embeddings
        print(f"Debug: Embedded {len(embeddings)} chunks for {filename}")

        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = hashlib.md5(f"{filename}_{i}_{chunk[:50]}".encode()).hexdigest()
            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "filename": filename,
                    "text": chunk,
                    "chunk_index": i
                }
            })

        pinecone_index.upsert(vectors=vectors, namespace=namespace)
        print(f"Debug: Stored {len(vectors)} vectors in Pinecone for {filename}")

    except Exception as e:
        raise ValueError(f"Error storing chunks: {str(e)}")
