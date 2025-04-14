import cohere
import requests
import json
import google.generativeai as genai

def query_pinecone_for_resume_details(
    resume_query: str,
    cohere_api_key: str,
    pinecone_index,
    namespace: str = "resumes",
    top_k: int = 5
) -> list:
    """
    Query Pinecone for relevant resume chunks.
    """
    try:
        cohere_client = cohere.Client(cohere_api_key)
        response = cohere_client.embed(
            texts=[resume_query],
            model="embed-english-v3.0",
            input_type="search_query"
        )
        query_embedding = response.embeddings[0]
        print(f"Debug: Embedded resume query: {resume_query}")

        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )

        chunks = []
        for match in results["matches"]:
            chunks.append({
                "text": match["metadata"]["text"],
                "score": match["score"]
            })
        print(f"Debug: Found {len(chunks)} relevant chunks")
        return chunks

    except Exception as e:
        raise ValueError(f"Error querying Pinecone: {str(e)}")

def generate_jsearch_query(chunks: list, gemini_api_key: str, role: str, model_name: str = "gemini-1.5-pro") -> dict:
    """
    Use Gemini to generate a personalized JSearch API query from resume chunks.
    """
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name)

        chunks_text = "\n".join([chunk["text"] for chunk in chunks])
        prompt = f"""
You are an AI job search assistant. Based on the resume chunks below, craft a job search query for the JSearch API for the role "{role}". Identify:

- **Role**: Use the provided role: "{role}".
- **Skills**: Select 2-3 key technical skills (e.g., ["Python", "Django"]).
- **Location**: Preferred job location (e.g., "Pune, India" or "remote").

Resume Chunks:
{chunks_text}

Output a JSON object with keys 'role', 'skills', 'location'. If location is unclear, use "Pune, India". Ensure skills are concise.
Example: {{"role": "{role}", "skills": ["Python", "Django"], "location": "Pune, India"}}
        """

        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        print(f"Debug: Raw Gemini output: {raw_text}")

        # Robust JSON parsing
        if raw_text.startswith("```json") or raw_text.endswith("```"):
            raw_text = raw_text.strip("```json\n").strip("```").strip()
        query_params = json.loads(raw_text)
        print(f"Debug: Generated JSearch query: {query_params}")
        return query_params

    except Exception as e:
        raise ValueError(f"Error generating JSearch query: {str(e)}")

def query_jsearch(params: dict, jsearch_api_key: str) -> list:
    """
    Query JSearch API with personalized parameters.
    """
    try:
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            "X-RapidAPI-Key": jsearch_api_key,
            "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
        }
        query = f"{params['role']} in {params['location']}".replace(" ", "+")
        query_params = {
            "query": query,
            "country": "in"  # India
        }
        print(f"Debug: JSearch query params: {query_params}")

        response = requests.get(url, headers=headers, params=query_params)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "OK":
            raise ValueError(f"JSearch API error: {data.get('message', 'Unknown error')}")

        jobs = data.get("data", [])
        print(f"Debug: Found {len(jobs)} jobs for query: {query}")

        formatted_jobs = []
        for job in jobs:
            formatted_jobs.append({
                "job_title": job.get("job_title", ""),
                "employer_name": job.get("employer_name", ""),
                "job_description": job.get("job_description", "")[:200] + "...",
                "job_apply_link": job.get("job_apply_link", ""),
                "location": job.get("job_city", "") + ", " + job.get("job_country", "")
            })
        return formatted_jobs

    except Exception as e:
        raise ValueError(f"Error querying JSearch: {str(e)}")

def match_jobs_with_llm(
    resume_query: str,
    cohere_api_key: str,
    jsearch_api_key: str,
    gemini_api_key: str,
    pinecone_index
) -> dict:
    """
    Match jobs using Gemini-generated JSearch query for multiple roles.
    """
    try:
        # Query Pinecone for relevant chunks
        chunks = query_pinecone_for_resume_details(resume_query, cohere_api_key, pinecone_index)

        # Generate queries for two roles
        results = {}
        for role in ["Python Developer", "Cybersecurity Analyst"]:
            query_params = generate_jsearch_query(chunks, gemini_api_key, role)
            jobs = query_jsearch(query_params, jsearch_api_key)
            results[role] = jobs

        return results

    except Exception as e:
        raise ValueError(f"Error matching jobs: {str(e)}")
