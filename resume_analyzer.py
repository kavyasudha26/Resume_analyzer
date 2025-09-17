# app.py
# Run with: streamlit run app.py

import os
import re
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import streamlit as st

import pdfplumber
import docx2txt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# --------------------------------------------------
# ML Resume Comparator Class
# --------------------------------------------------
class MLResumeComparator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

        st.info("🤖 Loading Sentence-BERT model...")
        self.sentence_model = SentenceTransformer(model_name)
        st.success(f"✅ Model loaded: {model_name}")

        self.ml_classifier = None
        self.scaler = StandardScaler()
        self.classifier_trained = False

    def extract_text(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self._extract_pdf_text(file_path)
        elif ext == ".docx":
            return self._extract_docx_text(file_path)
        else:
            return ""

    def _extract_pdf_text(self, file_path: str) -> str:
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            self.logger.error(f"PDF extraction error: {e}")
            return ""
        return text.strip()

    def _extract_docx_text(self, file_path: str) -> str:
        try:
            return docx2txt.process(file_path).strip()
        except:
            return ""

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\S+@\S+', ' ', text)  # emails
        text = re.sub(r'\b\d{10,}\b', ' ', text)  # phone numbers
        text = re.sub(r'http\S+', ' ', text)  # URLs
        text = re.sub(r'[^a-zA-Z0-9\s\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        words = [w for w in text.split() if len(w) > 2]
        return " ".join(words)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        return np.array(self.sentence_model.encode(texts, convert_to_tensor=False))

    def calculate_similarity(self, job_emb, res_emb) -> float:
        return float(cosine_similarity([job_emb], [res_emb])[0][0])

    def train_ml_classifier(self):
        synthetic_pairs = [
            {"job": "Python developer with ML experience",
             "resume": "Python developer with 3 years ML experience", "label": 1},
            {"job": "Data scientist position requiring SQL and Python",
             "resume": "Data scientist proficient in SQL and Python", "label": 1},
            {"job": "Frontend developer with React",
             "resume": "Frontend developer skilled in React and JS", "label": 1},
            {"job": "Java backend developer",
             "resume": "Software engineer with Java", "label": 1},
            {"job": "Marketing manager",
             "resume": "Marketing professional with social media", "label": 1},
            {"job": "Python developer with ML experience",
             "resume": "Marketing manager with advertising", "label": 0},
            {"job": "Data scientist requiring SQL",
             "resume": "Graphic designer skilled in Adobe", "label": 0},
            {"job": "Frontend developer with React",
             "resume": "Accountant with financial analysis", "label": 0},
        ]

        jobs = [p["job"] for p in synthetic_pairs]
        resumes = [p["resume"] for p in synthetic_pairs]
        embeddings = self.get_embeddings(jobs + resumes)
        job_embs, res_embs = embeddings[:len(jobs)], embeddings[len(jobs):]

        X, y = [], []
        for i, pair in enumerate(synthetic_pairs):
            sim = self.calculate_similarity(job_embs[i], res_embs[i])
            features = [
                sim,
                np.mean(job_embs[i]), np.std(job_embs[i]),
                np.mean(res_embs[i]), np.std(res_embs[i]),
            ]
            X.append(features)
            y.append(pair["label"])

        X_scaled = self.scaler.fit_transform(X)
        self.ml_classifier = LogisticRegression(random_state=42)
        self.ml_classifier.fit(X_scaled, y)
        self.classifier_trained = True

    def predict_match_probability(self, job_emb, res_emb) -> float:
        if not self.classifier_trained:
            return 0.0
        sim = self.calculate_similarity(job_emb, res_emb)
        features = [
            sim,
            np.mean(job_emb), np.std(job_emb),
            np.mean(res_emb), np.std(res_emb),
        ]
        scaled = self.scaler.transform([features])
        return float(self.ml_classifier.predict_proba(scaled)[0][1])


# --------------------------------------------------
# Streamlit Interface
# --------------------------------------------------
def main():
    st.title("📄 ML-based Resume Comparator")
    st.write("Upload a **job description** and two resumes (PDF/DOCX). The system will compare them using semantic embeddings and ML.")

    # Job description input
    job_description = st.text_area("✍️ Enter Job Description")

    # File uploads
    resume_a_file = st.file_uploader("📁 Upload Resume A", type=["pdf", "docx"])
    resume_b_file = st.file_uploader("📁 Upload Resume B", type=["pdf", "docx"])

    if st.button("🚀 Compare Resumes"):
        if not job_description or not resume_a_file or not resume_b_file:
            st.error("Please provide a job description and upload two resumes.")
            return

        comparator = MLResumeComparator()
        comparator.train_ml_classifier()

        # Save uploaded files temporarily
        temp_a = Path("resume_a" + Path(resume_a_file.name).suffix)
        temp_b = Path("resume_b" + Path(resume_b_file.name).suffix)
        with open(temp_a, "wb") as f: f.write(resume_a_file.read())
        with open(temp_b, "wb") as f: f.write(resume_b_file.read())

        # Extract + preprocess
        job_proc = comparator.preprocess_text(job_description)
        res_a_text = comparator.extract_text(str(temp_a))
        res_b_text = comparator.extract_text(str(temp_b))
        res_a_proc = comparator.preprocess_text(res_a_text)
        res_b_proc = comparator.preprocess_text(res_b_text)

        embeddings = comparator.get_embeddings([job_proc, res_a_proc, res_b_proc])
        job_emb, res_a_emb, res_b_emb = embeddings

        sim_a = comparator.calculate_similarity(job_emb, res_a_emb)
        sim_b = comparator.calculate_similarity(job_emb, res_b_emb)

        prob_a = comparator.predict_match_probability(job_emb, res_a_emb)
        prob_b = comparator.predict_match_probability(job_emb, res_b_emb)

        score_a = 0.7 * sim_a + 0.3 * prob_a
        score_b = 0.7 * sim_b + 0.3 * prob_b

        st.subheader("📊 Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Resume A** ({resume_a_file.name})")
            st.write(f"Similarity: {sim_a:.4f}")
            st.write(f"ML Match Probability: {prob_a:.2%}")
            st.write(f"Combined Score: {score_a:.4f}")
        with col2:
            st.markdown(f"**Resume B** ({resume_b_file.name})")
            st.write(f"Similarity: {sim_b:.4f}")
            st.write(f"ML Match Probability: {prob_b:.2%}")
            st.write(f"Combined Score: {score_b:.4f}")

        if abs(score_a - score_b) < 0.05:
            st.info("🤝 It's a tie! Both resumes are equally good.")
        elif score_a > score_b:
            st.success("🥇 Winner: Resume A")
        else:
            st.success("🥇 Winner: Resume B")


if __name__ == "__main__":
    main()
