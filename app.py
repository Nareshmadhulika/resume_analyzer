import streamlit as st
st.set_page_config(page_title="Resume & Job Description Analyzer", layout="wide")
import pickle
import docx 
import PyPDF2  
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = pickle.load(open('clf.pkl', 'rb'))  
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  
le = pickle.load(open('encoder.pkl', 'rb'))


@st.cache_resource
def load_sbert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

sbert_model = load_sbert_model()

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text


def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text])

    vectorized_text = vectorized_text.toarray()

    predicted_category = model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]

def compute_similarity(resume_text, jd_text):
    """
    Compute cosine similarity between resume and job description using SBERT embeddings
    """
    # Generate embeddings
    resume_emb = sbert_model.encode(resume_text)
    jd_emb = sbert_model.encode(jd_text)
    
    # Compute cosine similarity
    similarity_score = cosine_similarity([resume_emb], [jd_emb])[0][0]
    return similarity_score

# Streamlit app
def main():
    st.title("Resume & Job Description Analyzer")
    st.markdown("Upload a resume and job description to analyze category prediction and similarity matching.")
    
    # Create two columns for file uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Resume Upload")
        uploaded_resume = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"], key="resume")
    
    with col2:
        st.subheader("💼 Job Description Upload")
        uploaded_jd = st.file_uploader("Upload a Job Description", type=["pdf", "docx", "txt"], key="jd")
    
    # Process resume if uploaded
    resume_text = None
    if uploaded_resume is not None:
        try:
            resume_text = handle_file_upload(uploaded_resume)
            st.success("✅ Successfully extracted text from the uploaded resume.")
            
            # Show extracted text option
            if st.checkbox("Show extracted resume text", False):
                st.text_area("Extracted Resume Text", resume_text, height=200)
            
            # Category prediction section
            st.subheader("🎯 Resume Category Prediction")
            category = pred(resume_text)
            st.write(f"The predicted category of the uploaded resume is: **{category}**")
            
        except Exception as e:
            st.error(f"Error processing the resume file: {str(e)}")
    
    # Process job description if uploaded
    jd_text = None
    if uploaded_jd is not None:
        try:
            jd_text = handle_file_upload(uploaded_jd)
            st.success("✅ Successfully extracted text from the uploaded job description.")
            
            # Show extracted text option
            if st.checkbox("Show extracted job description text", False):
                st.text_area("Extracted Job Description Text", jd_text, height=200)
                
        except Exception as e:
            st.error(f"Error processing the job description file: {str(e)}")
    
    # Similarity analysis section
    if resume_text is not None and jd_text is not None:
        st.subheader("🔍 Similarity Analysis")
        st.markdown("Computing semantic similarity between resume and job description...")
        
        with st.spinner("Generating embeddings and computing similarity..."):
            similarity_score = compute_similarity(resume_text, jd_text)
            # Convert numpy float32 to native Python float for Streamlit compatibility
            similarity_score = float(similarity_score)
        
        # Display similarity score with visual elements
        st.markdown("### Similarity Score")
        
        # Create a progress bar and percentage display
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.progress(similarity_score)
        
        with col2:
            st.metric("Similarity", f"{similarity_score:.3f}")
        
        with col3:
            percentage = similarity_score * 100
            st.metric("Percentage", f"{percentage:.1f}%")
        
        # Add interpretation
        st.markdown("### Interpretation")
        if similarity_score >= 0.8:
            st.success("🎉 **Excellent Match!** The resume and job description show very high semantic similarity.")
        elif similarity_score >= 0.6:
            st.info("👍 **Good Match** The resume and job description show good semantic similarity.")
        elif similarity_score >= 0.3:
            st.warning("⚠️ **Moderate Match** The resume and job description show moderate semantic similarity.")
        else:
            st.error("❌ **Low Match** The resume and job description show low semantic similarity.")
    
    elif resume_text is not None or jd_text is not None:
        st.info("📝 Please upload both a resume and a job description to perform similarity analysis.")
    
    # --- Generate Suggestions Section ---
    st.markdown("---")
    st.subheader("💡 Generate Suggestions (Resume Suggestions Based on JD)")
    st.markdown("Get personalized resume improvement tips using a state-of-the-art language model.")

    if resume_text is not None and jd_text is not None:
        if 'groq_suggestions' not in st.session_state:
            st.session_state['groq_suggestions'] = ''
        if st.button("Generate Suggestions"):
            import requests
            import os
            # --- API Key Handling ---
            GROQ_API_KEY = os.getenv("GROQ_API_KEY")
            if not GROQ_API_KEY:
                st.error("Please set the GROQ_API_KEY environment variable with your Groq API key.")
                return
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            max_chars = 6000
            resume_input = resume_text[:max_chars]
            jd_input = jd_text[:max_chars]
            payload = {
                "model": "llama-3.1-8b-instant",  # or "mixtral-8x7b-32768"
                "messages": [
                    {"role": "system", "content": "You are a helpful career coach and resume expert."},
                    {"role": "user", "content": f"Here is a job description:\n{jd_input}"},
                    {"role": "user", "content": f"Here is a resume:\n{resume_input}"},
                    {"role": "user", "content": "Suggest improvements to this resume to better match the job description. Mention missing skills or any important changes."}
                ]
            }
            with st.spinner("Generating suggestions using llama 3..."):
                try:
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    response.raise_for_status()
                    suggestions = response.json()["choices"][0]["message"]["content"]
                    st.session_state['groq_suggestions'] = suggestions
                except Exception as e:
                    st.session_state['groq_suggestions'] = f"Error: {str(e)}"
        if st.session_state['groq_suggestions']:
            st.markdown("#### Suggestions to Improve Your Resume:")
            st.text_area("LLM Suggestions", st.session_state['groq_suggestions'], height=250)
    else:
        st.info("Please upload both a resume and a job description to generate suggestions.")
    
    # Add some helpful information
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **Resume Category Prediction:**
        - Uses TF-IDF vectorization and a pre-trained classifier
        - Predicts the professional category of the uploaded resume
        
        **Similarity Analysis:**
        - Uses SBERT (Sentence-BERT) embeddings for semantic understanding
        - Computes cosine similarity between resume and job description
        - Score ranges from 0 (no similarity) to 1 (perfect match)
        - Higher scores indicate better alignment between resume and job requirements
        """)


if __name__ == "__main__":
    main()
