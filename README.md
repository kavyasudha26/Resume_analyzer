# AI-Powered Resume Screening System

A machine learning–driven system to parse, preprocess, and rank resumes against job descriptions using **Sentence-BERT embeddings** and **ML classifiers**. Includes an interactive **Streamlit web app** for visualization.

## Features

- Extracts text from PDF and DOCX resumes.
- Preprocesses text: removes stopwords, emails, phone numbers, URLs, punctuation.
- Computes semantic similarity using Sentence-BERT embeddings.
- Predicts candidate-job match probability using Logistic Regression.
- Ranks resumes based on combined similarity and ML probability.
- Streamlit interface for easy upload and real-time results.

## Tech Stack

- **Languages:** Python  
- **Libraries:** Streamlit, Sentence-Transformers, Scikit-learn, pdfplumber, docx2txt, Numpy, Pandas, NLTK  
- **Tools:** Git, Cloud (optional)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/AI-Resume-Comparator.git
cd AI-Resume-Comparator

Create a virtual environment:
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app
