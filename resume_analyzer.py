import os  # Import the 'os' module for file system operations
import PyPDF2

def get_job_description():
    """Gets the job description from the user via command line input."""
    print("Please enter the job description:")
    job_description = input()  # Reads the user's input from the console
    return job_description

def get_resume_path():
    """Gets the resume file path from the user."""
    while True:  # Loop until a valid file path is entered
        print("\nPlease enter the path to your resume file (PDF):")
        resume_path = input()

        if os.path.exists(resume_path) and resume_path.lower().endswith(".pdf"):
            return resume_path  # Valid path, exit the loop
        else:
            print("Invalid file path. Please enter a valid path to a PDF file.")
        
stop_words = set([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "from", "in", "on", "at", "for", "of", "by", "with", "about", "against",
    "and", "or", "but", "not", "it", "its", "that", "this", "these", "those",
    "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "we", "us", "our", "ours", "they", "them", "their", "theirs"
])

def extract_keywords(text):
    """Extracts keywords from the given text by removing stop words and converting to lowercase."""
    words = text.lower().split()  # Convert to lowercase and split into words
    keywords = [word for word in words if word not in stop_words]  # Remove stop words
    return keywords

def extract_text_from_resume(resume_path):
    """Extracts text from a PDF resume using PyPDF2."""
    try:
        with open(resume_path, 'rb') as file:  # Open the PDF file in binary read mode
            reader = PyPDF2.PdfReader(file)  # Create a PdfReader object
            text = ""
            for page in reader.pages:  # Iterate through each page in the PDF
                text += page.extract_text()  # Extract the text from the page and append it to the 'text' variable
            return text
    except Exception as e:
        print(f"Error extracting text from resume: {e}")
        return ""  # Return an empty string if there's an error


def match_keywords(resume_text, keywords):
    """Counts the number of times each keyword appears in the resume text."""
    resume_text = resume_text.lower()  # Convert resume text to lowercase
    keyword_counts = {}  # Dictionary to store keyword counts

    for keyword in keywords:
        keyword_counts[keyword] = resume_text.count(keyword)  # Count occurrences of each keyword

    return keyword_counts     

def calculate_score(keyword_counts):
    """Calculates a score based on the keyword counts."""
    total_score = sum(keyword_counts.values())  # Sum of all keyword counts
    return total_score   



if __name__ == "__main__":
    job_description = get_job_description()
    print("\nYou entered the following job description:\n", job_description)

    resume_path = get_resume_path()  # Get the resume path from the user
    print("\nYou entered the following resume path:\n", resume_path)

    keywords = extract_keywords(job_description)  # Extract keywords from the job description
    print("\nExtracted keywords from job description:\n", keywords)

    resume_text = extract_text_from_resume(resume_path)  # Extract text from the resume
    
    keyword_counts = match_keywords(resume_text, keywords)  # Match keywords and get counts
    print("\nKeyword counts in resume:\n", keyword_counts)

    score = calculate_score(keyword_counts)  # Calculate the resume score
    
    print("\n--- Resume Analysis ---")  # Separator for clarity
    print("Resume Score:", score)

    print("\nKeyword Matches:")
    for keyword, count in keyword_counts.items():
        print(f"- {keyword}: {count}")

    # Basic Feedback
    if score > 10:
        print("\nThis resume seems well-aligned with the job description.")
    elif score > 5:
        print("\nThis resume has some relevant keywords but could be improved.")
    else:
        print("\nThis resume may not be a strong match for the job description. Consider adding more relevant keywords.")

    print("------------------------")  
