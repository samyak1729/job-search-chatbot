import streamlit as st
import os
import json
import tempfile
from dotenv import load_dotenv
from app.ingestion import parse_resume, validate_file
from app.storage import initialize_pinecone, store_chunks
from app.job_search import query_pinecone_for_resume_details, generate_jsearch_query, query_jsearch

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
JSEARCH_API_KEY = os.getenv("JSEARCH_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Pinecone
try:
    pinecone_index = initialize_pinecone(PINECONE_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {e}")
    st.stop()

# Streamlit UI
st.title("JobSearchRAG: Your AI Job Matchmaker")
st.write("Upload your resume, share your preferences, and chat to find jobs!")

# Sidebar for resume and info
with st.sidebar:
    st.header("Your Profile")
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    user_name = st.text_input("Your Name", "")
    user_location = st.text_input("Preferred Location", "Pune, India")
    user_role = st.text_input("Preferred Role", "Python Developer")

    # Process resume
    if uploaded_resume:
        with st.spinner("Processing resume..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_resume.read())
                    tmp_path = tmp.name
                if validate_file(tmp_path):
                    resume_data = parse_resume(tmp_path, COHERE_API_KEY)
                    st.session_state.resume_data = resume_data
                    if resume_data["chunks"]:
                        store_chunks(
                            chunks=resume_data["chunks"],
                            filename=uploaded_resume.name,
                            cohere_api_key=COHERE_API_KEY,
                            pinecone_index=pinecone_index
                        )
                        st.success(f"Resume processed: {len(resume_data['chunks'])} chunks stored!")
                    else:
                        st.warning("No chunks extracted from resume.")
                else:
                    st.error("Invalid resume file.")
                os.unlink(tmp_path)  # Clean up
            except Exception as e:
                st.error(f"Error processing resume: {e}")

# Chat interface
st.header("Chat with JobSearchRAG")
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Upload your resume or ask about jobs (e.g., 'Find Python Developer jobs in Pune')."}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "jobs" in message:
            for job in message["jobs"]:
                st.write(f"**{job['job_title']}** at {job['employer_name']}")
                st.write(f"{job['job_description']}")
                st.write(f"[Apply here]({job['job_apply_link']})")
                st.write(f"Location: {job['location']}")
                st.write("---")

# Chat input
if prompt := st.chat_input("Ask about jobs or preferences..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process query
    with st.spinner("Searching for jobs..."):
        try:
            # Use user info if provided
            role = user_role if user_role else "Python Developer"
            location = user_location if user_location else "Pune, India"
            resume_query = prompt if "job" in prompt.lower() else "software development skills"

            # Query Pinecone
            chunks = query_pinecone_for_resume_details(
                resume_query, COHERE_API_KEY, pinecone_index
            )
            if not chunks:
                response = "No relevant resume data found. Please upload a resume."
            else:
                # Generate JSearch query
                query_params = generate_jsearch_query(chunks, GEMINI_API_KEY, role)
                query_params["location"] = location  # Override with user input
                jobs = query_jsearch(query_params, JSEARCH_API_KEY)
                if jobs:
                    response = f"Found {len(jobs)} job matches for {role} in {location}!"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "jobs": jobs
                    })
                else:
                    response = "No jobs found. Try a different role or location."
        except Exception as e:
            response = f"Error: {str(e)}"

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)

# Job table
if st.session_state.messages and any("jobs" in msg for msg in st.session_state.messages):
    st.header("Your Job Matches")
    jobs = []
    for msg in st.session_state.messages:
        if "jobs" in msg:
            jobs.extend(msg["jobs"])
    if jobs:
        # Create table
        job_data = [
            {
                "Title": job["job_title"],
                "Employer": job["employer_name"],
                "Location": job["location"],
                "Description": job["job_description"],
                "Apply Link": f"[Apply]({job['job_apply_link']})"
            }
            for job in jobs
        ]
        st.table(job_data)
