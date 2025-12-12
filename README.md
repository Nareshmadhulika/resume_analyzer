# Resume & Job Description Analyzer

This project is a web application that allows users to upload a resume and a job description (JD), analyzes the resume to predict its professional category, computes the semantic similarity between the resume and the JD, and provides AI-powered suggestions to improve the resume for a better match with the job description.

## Features
- **Resume Category Prediction:** Uses TF-IDF vectorization and a pre-trained classifier to predict the professional category of the uploaded resume.
- **Semantic Similarity Analysis:** Uses SBERT (Sentence-BERT) embeddings to compute the semantic similarity between the resume and the job description.
- **Resume Suggestions:** Integrates with a language model API to provide personalized suggestions for improving the resume based on the job description.
- **Supports PDF, DOCX, and TXT files** for both resumes and job descriptions.

## Installation
1. **Clone the repository** (or download the project files):
   ```bash
   git clone <repo-url>
   cd Resume_Analyzer
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## API Key Setup
The resume suggestions feature uses the Groq API. To use this feature:

1. Sign up at [Groq](https://groq.com) and get your API key.
2. Set the environment variable `GROQ_API_KEY`:
   - On Windows (PowerShell):
     ```powershell
     $env:GROQ_API_KEY = "your-api-key-here"
     ```
   - Or create a `.env` file in the project root:
     ```
     GROQ_API_KEY=your-api-key-here
     ```
     And install `python-dotenv`:
     ```bash
     pip install python-dotenv
     ```
     Then add `from dotenv import load_dotenv; load_dotenv()` at the top of `app.py`.

## Usage
1. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
2. **Open your browser** and go to the local URL provided by Streamlit (usually http://localhost:8501).
3. **Upload a resume and a job description** in PDF, DOCX, or TXT format.
4. **View the predicted category, similarity score, and get suggestions** to improve your resume.

## Files
- `app.py`: Main Streamlit application.
- `test_sbert.py`: Script to test SBERT model and similarity computation.
- `clf.pkl`, `tfidf.pkl`, `encoder.pkl`: Pre-trained model and encoders (required for category prediction).
- `requirements.txt`: List of required Python packages.
- `UpdatedResumeDataSet.csv`: Dataset used for training (not required for app usage).

## Notes
- The app requires pre-trained model files (`clf.pkl`, `tfidf.pkl`, `encoder.pkl`) to be present in the project directory.
- The suggestions feature uses the Groq API. Ensure you have set the `GROQ_API_KEY` environment variable as described above.

## License
This project is for educational and demonstration purposes. 
