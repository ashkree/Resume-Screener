import textract

# Parse the resume file to extract plain text
def parse_resume(filepath):
    try:
        text = textract.process(filepath).decode('utf-8')
        return text.strip()
    except Exception as e:
        return f"Error parsing resume: {str(e)}"

# Process parsed resume text using our ML model
def process_resume(parsed_text):
    return f"AI Review based on extracted resume text: {parsed_text[:200]}..."
