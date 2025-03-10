
# AI-Powered Resume Screening and Ranking System

This Python project implements a basic resume screening system that analyzes resumes against job descriptions to assess their suitability. The system:

*   Takes a job description as input from the user.
*   Accepts a resume in PDF format.
*   Extracts keywords from the job description (using basic stop word removal).
*   Extracts text from the resume using PyPDF2.
*   Matches keywords from the job description to the resume content.
*   Calculates a score based on the number of keyword matches.
*   Provides a basic analysis of the resume and its alignment with the job description.

## Tech Stack:

*   Python
*   PyPDF2 (for PDF text extraction)

## Installation:

1.  Make sure you have Python installed (version 3.6 or higher).
2.  Install the required libraries:
    ```bash
    pip install PyPDF2
    ```

## Usage:

1.  Clone this repository.
2.  Run the `resume_analyzer.py` script:
    ```bash
    python resume_analyzer.py
    ```
3.  Follow the prompts to enter the job description and the path to the resume file.

## Potential Future Enhancements:

*   Improved keyword extraction using NLTK (stemming, lemmatization, TF-IDF).
*   More sophisticated scoring algorithms.
*   GUI using Tkinter or Gradio.
*   Machine learning classification.

## License:

[Add a license here, e.g., MIT License]
