import re
import unicodedata
import logging
from typing import Iterable


class TextPreprocessor:
    
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
            (r'Objective([A-Z])', r'Objective \1'),
            (r'Experience([A-Z])', r'Experience \1'),
            (r'Education([A-Z])', r'Education \1'),
            (r'Skills([A-Z])', r'Skills \1'),
            (r'Certifications([A-Z])', r'Certifications \1'),
            (r'Projects([A-Z])', r'Projects \1'),
            (r'Publications([A-Z])', r'Publications \1'),
            (r'References([A-Z])', r'References \1'),
            (r'Contact([A-Z])', r'Contact \1'),
            (r'Achievements([A-Z])', r'Achievements \1'),
            (r'Languages([A-Z])', r'Languages \1'),
            (r'Hobbies([A-Z])', r'Hobbies \1'),
            (r'Interests([A-Z])', r'Interests \1'),
            (r'Volunteer([A-Z])', r'Volunteer \1'),
            (r'Profile([A-Z])', r'Profile \1'),
            (r'Strengths([A-Z])', r'Strengths \1'),
            (r'Weaknesses([A-Z])', r'Weaknesses \1'),
            (r'Summary([A-Z])', r'Summary \1'),
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
        
        # Contact info patterns
        self.contact_patterns = [
            (r'\S+@\S+', '<EMAIL>'),
            (r'\b\d{10,}\b', '<PHONE>'),  # crude phone redaction
        ]
        
        # Normalization patterns
        self.normalization_patterns = [
            (r'[-—―]', '-'),
            (r'“|”', '"'),
            (r'‘|’', "'"),
        ]
        
        # Pre-compile frequently-used regex patterns for speed
        self.compiled_spacing = [(re.compile(p), r) for p, r in self.spacing_patterns]
        
        # Initialise logger
        self.logger = logging.getLogger(__name__)

    def clean_text(self, text):
        """
        Comprehensive text cleaning with all pattern fixes
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Unicode normalisation first
        cleaned = unicodedata.normalize("NFKC", text)
        
        # Apply section header fixes first
        for pattern, replacement in self.section_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Apply technical term patterns
        for pattern, replacement in self.technical_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Apply general spacing fixes (pre-compiled)
        for pattern, replacement in self.compiled_spacing:
            cleaned = pattern.sub(replacement, cleaned)
        
        # Apply contact info patterns
        for pattern, replacement in self.contact_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # Final cleanup - lower-case then strip
        cleaned = cleaned.lower().strip()
        
        return cleaned
    
    # ------------------------------------------------------------------
    # Optional pipeline steps
    # ------------------------------------------------------------------

    def remove_stop_words(self, text: str) -> str:

        if not self._stop_words:
            self.logger.warning('Stop-word list unavailable - skipping removal.')
            return text
        tokens = [tok for tok in text.split() if tok not in self._stop_words]
        return ' '.join(tokens)

    def lemmatize_text(self, text: str) -> str:

        if not self._lemmatizer:
            self.logger.warning('WordNet lemmatiser unavailable - skipping.')
            return text
        return ' '.join(self._lemmatizer.lemmatize(tok) for tok in text.split())

    # ------------------------------------------------------------------
    # Dataset-level processing with configurable pipeline
    # ------------------------------------------------------------------

    def process_dataset(
        self,
        df,
        remove_stop_words: bool = False,
        lemmatize: bool = False,
        text_cols: Iterable[str] = ('resume_text', 'job_description_text'),
    ):

        df_clean = df.copy()

        steps = [self.clean_text]
        if remove_stop_words:
            steps.append(self.remove_stop_words)
        if lemmatize:
            steps.append(self.lemmatize_text)

        # Compose sequentially for speed
        def _pipeline(txt: str) -> str:
            for fn in steps:
                txt = fn(txt)
            return txt

        for col in text_cols:
            if col in df_clean.columns:
                self.logger.info(f'Processing column: {col} …')
                df_clean[col] = df_clean[col].astype(str).map(_pipeline)
            else:
                self.logger.warning(f'Column "{col}" missing - skipped.')

        return df_clean