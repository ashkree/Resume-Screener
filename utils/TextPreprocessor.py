import re


class TextPreprocessor:
    """
    A comprehensive text preprocessor for resume and job description data
    with extensive spacing, formatting, and normalization fixes.
    """
    
    def __init__(self):
        # Comprehensive spacing patterns
        self.spacing_patterns = [
            # Fix missing space between lowercase and uppercase letters
            (r'([a-z])([A-Z])', r'\1 \2'),
            
            # Fix missing space between uppercase and lowercase (reversed)
            (r'([A-Z]{2,})([a-z])', r'\1 \2'),
            
            # Fix missing space between characters and numbers
            (r'([a-zA-Z])([0-9])', r'\1 \2'),
            
            # Fix missing space between numbers and characters (reversed)
            (r'([0-9])([a-zA-Z])', r'\1 \2'),
            
            # Fix missing space after punctuation followed by letter
            (r'([.!?:;,])([A-Za-z])', r'\1 \2'),
            
            # Fix missing space before punctuation in some cases
            (r'([a-zA-Z])([.!?:;,]{2,})', r'\1 \2'),
            
            # Fix missing space around dashes and hyphens
            (r'([a-zA-Z])-([a-zA-Z])', r'\1 - \2'),
            (r'([0-9])-([0-9])', r'\1-\2'),  # Keep date ranges together
            
            # Fix missing space around slashes
            (r'([a-zA-Z])/([a-zA-Z])', r'\1 / \2'),
            
            # Fix missing space after common abbreviations
            (r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Corp|LLC|Ltd)\.([A-Z])', r'\1. \2'),
            
            # Fix missing space around parentheses
            (r'([a-zA-Z])\(', r'\1 ('),
            (r'\)([a-zA-Z])', r') \1'),
            
            # Fix missing space around brackets
            (r'([a-zA-Z])\[', r'\1 ['),
            (r'\]([a-zA-Z])', r'] \1'),
            
            # Fix missing space around mathematical operators
            (r'([a-zA-Z0-9])([+*=<>])([a-zA-Z0-9])', r'\1 \2 \3'),
            
            # Fix multiple consecutive spaces
            (r'\s+', ' '),
            
            # Fix space before punctuation (common OCR error)
            (r'\s+([.!?:;,])', r'\1'),
            
            # Fix missing space after currency symbols
            (r'([$£€¥])([0-9])', r'\1 \2'),
            
            # Fix missing space around ampersand
            (r'([a-zA-Z])&([a-zA-Z])', r'\1 & \2'),
        ]
        
        # Section header patterns (more comprehensive)
        self.section_patterns = [
            (r'Summary([A-Z])', r'Summary \1'),
            (r'Experience([A-Z])', r'Experience \1'),
            (r'Education([A-Z])', r'Education \1'),
            (r'Skills([A-Z])', r'Skills \1'),
            (r'Qualifications([A-Z])', r'Qualifications \1'),
            (r'Requirements([A-Z])', r'Requirements \1'),
            (r'Responsibilities([A-Z])', r'Responsibilities \1'),
            (r'Objective([A-Z])', r'Objective \1'),
            (r'Profile([A-Z])', r'Profile \1'),
            (r'Background([A-Z])', r'Background \1'),
            (r'Achievements([A-Z])', r'Achievements \1'),
            (r'Accomplishments([A-Z])', r'Accomplishments \1'),
            (r'Certifications([A-Z])', r'Certifications \1'),
            (r'Projects([A-Z])', r'Projects \1'),
            (r'References([A-Z])', r'References \1'),
        ]
        
        # Job-specific patterns
        self.job_patterns = [
            (r'Job([A-Z])', r'Job \1'),
            (r'Position([A-Z])', r'Position \1'),
            (r'Role([A-Z])', r'Role \1'),
            (r'Title([A-Z])', r'Title \1'),
            (r'Location([A-Z])', r'Location \1'),
            (r'Duration([A-Z])', r'Duration \1'),
            (r'Contract([A-Z])', r'Contract \1'),
            (r'Remote([A-Z])', r'Remote \1'),
            (r'Salary([A-Z])', r'Salary \1'),
            (r'Benefits([A-Z])', r'Benefits \1'),
        ]
        
        # Technical term patterns
        self.technical_patterns = [
            # Programming languages
            (r'([a-z])(JavaScript|Python|Java|PHP|Ruby|Swift|Kotlin)', r'\1 \2'),
            (r'(JavaScript|Python|Java|PHP|Ruby|Swift|Kotlin)([a-z])', r'\1 \2'),
            
            # Frameworks and libraries
            (r'([a-z])(React|Angular|Vue|Django|Flask|Spring|Laravel)', r'\1 \2'),
            (r'(React|Angular|Vue|Django|Flask|Spring|Laravel)([a-z])', r'\1 \2'),
            
            # Databases
            (r'([a-z])(MySQL|PostgreSQL|MongoDB|Redis|Oracle)', r'\1 \2'),
            (r'(MySQL|PostgreSQL|MongoDB|Redis|Oracle)([a-z])', r'\1 \2'),
            
            # Cloud platforms
            (r'([a-z])(AWS|Azure|Google Cloud|GCP|Docker|Kubernetes)', r'\1 \2'),
            (r'(AWS|Azure|Google Cloud|GCP|Docker|Kubernetes)([a-z])', r'\1 \2'),
        ]
        
        # Email and URL patterns
        self.contact_patterns = [
            # Fix email spacing
            (r'([a-zA-Z0-9])(@)', r'\1 \2'),
            (r'(@)([a-zA-Z0-9])', r'\1 \2'),
            
            # Fix URL/website spacing
            (r'(www\.)([a-zA-Z])', r'\1 \2'),
            (r'(\.com|\.org|\.net|\.edu|\.gov)([a-zA-Z])', r'\1 \2'),
            
            # Fix phone number patterns
            (r'(\d{3})(\d{3})(\d{4})', r'\1-\2-\3'),  # Format: 555-555-5555
        ]
        
        # Normalization patterns
        self.normalization_patterns = [
            # Standardize bullet points
            (r'[•·▪▫◦‣⁃]', '• '),
            
            # Standardize quotes
            (r'["""]', '"'),
            (r'[\'\'\']', "'"),
            
            # Standardize dashes
            (r'[–—―]', '-'),
            
            # Remove excessive newlines but preserve some structure
            (r'\n\s*\n\s*\n+', '\n\n'),
            (r'\n\s*\n', '\n'),
            
            # Fix common abbreviations
            (r'\bw/\b', 'with'),
            (r'\bw/o\b', 'without'),
            (r'\b&\b', 'and'),
            (r'\byrs?\b', 'years'),
        ]
    
    def clean_text(self, text):
        """
        Comprehensive text cleaning with all pattern fixes
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Comprehensively cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        cleaned = text
        
        # Apply section header fixes first
        for pattern, replacement in self.section_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Apply job-specific patterns
        for pattern, replacement in self.job_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Apply technical term patterns
        for pattern, replacement in self.technical_patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # Apply general spacing fixes
        for pattern, replacement in self.spacing_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Apply contact info patterns
        for pattern, replacement in self.contact_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # Final cleanup
        cleaned = cleaned.strip()
        
        return cleaned
    
    def process_dataset(self, df):
        """
        Process an entire dataset with comprehensive cleaning
        
        Args:
            df (pd.DataFrame): Dataset with resume_text and job_description_text columns
            
        Returns:
            pd.DataFrame: Dataset with comprehensively cleaned text
        """
        df_clean = df.copy()
        
        print("Processing resume texts with comprehensive cleaning...")
        df_clean['resume_text'] = df_clean['resume_text'].apply(self.clean_text)
        
        print("Processing job description texts with comprehensive cleaning...")
        df_clean['job_description_text'] = df_clean['job_description_text'].apply(self.clean_text)
        
        return df_clean
    

