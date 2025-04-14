import os
import json
from dotenv import load_dotenv
from app.ingestion import parse_resume, validate_file
from app.storage import initialize_pinecone, store_chunks
from app.job_search import match_jobs_with_llm

def process_store_and_match_jobs(
    input_folder: str = "resumes",
    cohere_api_key: str = None,
    pinecone_api_key: str = None,
    jsearch_api_key: str = None,
    gemini_api_key: str = None,
    resume_query: str = "software development skills"
):
    if not all([cohere_api_key, pinecone_api_key, jsearch_api_key, gemini_api_key]):
        print("Error: All API keys required")
        return
    if not os.path.exists(input_folder):
        print(f"Error: Folder '{input_folder}' does not exist")
        return
    try:
        pinecone_index = initialize_pinecone(pinecone_api_key)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return
    parsed_results = []
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if not validate_file(file_path):
            print(f"Skipping {filename}: Invalid file")
            continue
        try:
            resume_data = parse_resume(file_path, cohere_api_key)
            parsed_results.append(resume_data)
            print(f"\nParsed {filename}:")
            print(json.dumps(resume_data, indent=2))
            if resume_data["chunks"]:
                store_chunks(
                    chunks=resume_data["chunks"],
                    filename=filename,
                    cohere_api_key=cohere_api_key,
                    pinecone_index=pinecone_index
                )
                print(f"Stored chunks for {filename} in Pinecone")
            else:
                print(f"No chunks to store for {filename}")
            job_matches = match_jobs_with_llm(
                resume_query=resume_query,
                cohere_api_key=cohere_api_key,
                jsearch_api_key=jsearch_api_key,
                gemini_api_key=gemini_api_key,
                pinecone_index=pinecone_index
            )
            for role, jobs in job_matches.items():
                print(f"\nJob Matches for '{role}' in Pune, India:")
                print(json.dumps(jobs, indent=2))
        except ValueError as e:
            print(f"Error processing {filename}: {str(e)}")
    if parsed_results:
        with open("parsed_resumes.json", "w") as f:
            json.dump(parsed_results, f, indent=2)
        print("\nSaved all results to parsed_resumes.json")

if __name__ == "__main__":
    # Load environment variables from .env
    load_dotenv()
    cohere_api_key = os.getenv("COHERE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    jsearch_api_key = os.getenv("JSEARCH_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    process_store_and_match_jobs(
        cohere_api_key=cohere_api_key,
        pinecone_api_key=pinecone_api_key,
        jsearch_api_key=jsearch_api_key,
        gemini_api_key=gemini_api_key
    )
