import streamlit as st
import PyPDF2
import io
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set up Streamlit page
st.set_page_config(page_title="AI Resume Critiquer", page_icon="📃", layout="centered")
st.title("AI Resume Critiquer")
st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")

# Initialize Ollama with Mistral model
llm = Ollama(model="mistral")

# Create the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert resume reviewer with years of experience in HR and recruitment. 
     Analyze the resume and provide constructive feedback focusing on:
     1. Content clarity and impact
     2. Skills presentation
     3. Experience descriptions
     4. Specific improvements for the target job role"""),
    ("user", """Please analyze this resume and provide detailed feedback for {job_role} position:
     
     Resume content:
     {resume_text}
     
     Provide your analysis in a clear, structured format with specific recommendations.""")
])

# Create the chain
resume_chain = prompt_template | llm | StrOutputParser()

# File uploader
uploaded_file = st.file_uploader("Upload your Resume (PDF or txt)", type=["pdf", "txt"])
job_role = st.text_input("Enter the job role you're targeting (optional)", value="general job application")

analyze = st.button("Analyze Resume")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

if analyze and uploaded_file:
    try:
        with st.spinner("Analyzing your resume..."):
            file_content = extract_text_from_file(uploaded_file)

            if not file_content.strip():
                st.error("File does not have any content...")
                st.stop()

            # Generate feedback using LangChain and Mistral
            feedback = resume_chain.invoke({
                "resume_text": file_content,
                "job_role": job_role if job_role else "general job application"
            })

            st.markdown("### Analysis Results")
            st.markdown(feedback)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Make sure Ollama is running with the Mistral model installed.")
        st.info("To set up Ollama:\n1. Download from https://ollama.ai/\n2. Run `ollama pull mistral`")