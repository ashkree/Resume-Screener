import re
import unicodedata
import logging
from typing import Iterable, Set, Optional

# Import for stop words and lemmatization
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Alternative lemmatizers
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


class TextPreprocessor:
    
    def __init__(self, enable_stopwords: bool = True, custom_stopwords: Optional[Set[str]] = None,
                 enable_lemmatizer: bool = True, lemmatizer_type: str = 'nltk'):
        
        # Initialize logger FIRST
        self.logger = logging.getLogger(__name__)
        
        # Initialize stop words
        self._stop_words = None
        if enable_stopwords:
            self._init_stop_words(custom_stopwords)
        
        # Initialize lemmatizer
        self._lemmatizer = None
        self._lemmatizer_type = lemmatizer_type
        if enable_lemmatizer:
            self._init_lemmatizer(lemmatizer_type)
        
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
            (r'"|"', '"'),
            (r"'|'", "'"),
        ]
        
        # Pre-compile frequently-used regex patterns for speed
        self.compiled_spacing = [(re.compile(p), r) for p, r in self.spacing_patterns]

    def _init_stop_words(self, custom_stopwords: Optional[Set[str]] = None):
        """Initialize stop words from NLTK or use custom set"""
        if custom_stopwords:
            self._stop_words = custom_stopwords
            self.logger.info(f"Using custom stop words list with {len(custom_stopwords)} words")
            return
        
        if NLTK_AVAILABLE:
            try:
                # Download stopwords if not already present
                nltk.download('stopwords', quiet=True)
                english_stopwords = set(stopwords.words('english'))
                
                # Add some common technical/resume stop words
                additional_stopwords = {
                    'experience', 'years', 'work', 'worked', 'working', 'job', 'role',
                    'position', 'company', 'team', 'project', 'projects', 'skill', 'skills',
                    'knowledge', 'ability', 'responsible', 'duties', 'tasks', 'requirements',
                    'including', 'various', 'multiple', 'several', 'many', 'different',
                    'related', 'relevant', 'strong', 'excellent', 'good', 'well', 'highly'
                }
                
                self._stop_words = english_stopwords.union(additional_stopwords)
                self.logger.info(f"Initialized NLTK stop words with {len(self._stop_words)} words")
                
            except Exception as e:
                self.logger.warning(f"Failed to load NLTK stopwords: {e}")
                self._init_basic_stopwords()
        else:
            self.logger.warning("NLTK not available, using basic stop words list")
            self._init_basic_stopwords()

    def _init_basic_stopwords(self):
        """Fallback basic stop words list if NLTK is not available"""
        basic_stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 
            'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
            'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        
        # Add resume-specific stop words
        resume_stopwords = {
            'experience', 'years', 'work', 'worked', 'working', 'job', 'role',
            'position', 'company', 'team', 'project', 'projects', 'skill', 'skills',
            'knowledge', 'ability', 'responsible', 'duties', 'tasks', 'requirements',
            'including', 'various', 'multiple', 'several', 'many', 'different',
            'related', 'relevant', 'strong', 'excellent', 'good', 'well', 'highly'
        }
        
        self._stop_words = basic_stopwords.union(resume_stopwords)
        self.logger.info(f"Using basic stop words list with {len(self._stop_words)} words")

    def _init_lemmatizer(self, lemmatizer_type: str = 'nltk'):
        """Initialize lemmatizer based on specified type"""
        if lemmatizer_type == 'nltk' and NLTK_AVAILABLE:
            try:
                # Download required NLTK data
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('punkt', quiet=True)
                
                self._lemmatizer = WordNetLemmatizer()
                self._lemmatizer_type = 'nltk'
                self.logger.info("Initialized NLTK WordNet lemmatizer")
                
            except Exception as e:
                self.logger.warning(f"Failed to load NLTK lemmatizer: {e}")
                self._init_basic_lemmatizer()
                
        elif lemmatizer_type == 'spacy' and SPACY_AVAILABLE:
            try:
                # Try to load spaCy model
                import spacy
                try:
                    self._lemmatizer = spacy.load("en_core_web_sm")
                    self._lemmatizer_type = 'spacy'
                    self.logger.info("Initialized spaCy lemmatizer")
                except OSError:
                    # Model not found, try to download
                    self.logger.info("spaCy model not found, trying to download...")
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                                 capture_output=True)
                    self._lemmatizer = spacy.load("en_core_web_sm")
                    self._lemmatizer_type = 'spacy'
                    self.logger.info("Downloaded and initialized spaCy lemmatizer")
                    
            except Exception as e:
                self.logger.warning(f"Failed to load spaCy lemmatizer: {e}")
                self._init_basic_lemmatizer()
                
        else:
            self.logger.warning(f"Lemmatizer type '{lemmatizer_type}' not available, using basic lemmatizer")
            self._init_basic_lemmatizer()

    def _init_basic_lemmatizer(self):
        """Fallback basic lemmatizer using simple rules"""
        # Simple rule-based lemmatizer as fallback
        self._basic_lemma_rules = {
            # Common plural to singular
            'ies': 'y',  # companies -> company
            'ies': 'y',  # studies -> study
            'ves': 'f',  # lives -> life
            'ves': 'fe', # wives -> wife
            'ses': 's',  # classes -> class
            'ches': 'ch', # matches -> match
            'shes': 'sh', # wishes -> wish
            'xes': 'x',  # boxes -> box
            'zes': 'z',  # prizes -> prize
            's': '',     # cats -> cat (general case)
            
            # Common verb forms
            'ing': '',   # working -> work
            'ed': '',    # worked -> work
            'er': '',    # worker -> work
            'est': '',   # biggest -> big
            'ly': '',    # quickly -> quick
        }
        
        self._lemmatizer_type = 'basic'
        self.logger.info("Using basic rule-based lemmatizer")

    def _get_wordnet_pos(self, word):
        """Map POS tag to first character used by WordNetLemmatizer"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def add_stopwords(self, words: Iterable[str]):
        """Add custom stop words to the existing set"""
        if self._stop_words is None:
            self._stop_words = set()
        
        new_words = set(word.lower() for word in words)
        self._stop_words.update(new_words)
        self.logger.info(f"Added {len(new_words)} custom stop words")

    def remove_stopwords(self, words: Iterable[str]):
        """Remove words from the stop words set"""
        if self._stop_words is None:
            return
        
        words_to_remove = set(word.lower() for word in words)
        self._stop_words -= words_to_remove
        self.logger.info(f"Removed {len(words_to_remove)} words from stop words list")

    def get_stopwords(self) -> Set[str]:
        """Get the current set of stop words"""
        return self._stop_words.copy() if self._stop_words else set()

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
        """Remove stop words from text"""
        if not self._stop_words:
            self.logger.warning('Stop-word list unavailable - skipping removal.')
            return text
        
        tokens = [tok for tok in text.split() if tok not in self._stop_words]
        return ' '.join(tokens)

    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text using the configured lemmatizer"""
        if not self._lemmatizer and self._lemmatizer_type != 'basic':
            self.logger.warning('Lemmatizer unavailable - skipping.')
            return text
        
        if not text.strip():
            return text
            
        if self._lemmatizer_type == 'nltk':
            # NLTK WordNet lemmatizer
            tokens = text.split()
            lemmatized = []
            for word in tokens:
                try:
                    pos = self._get_wordnet_pos(word)
                    lemma = self._lemmatizer.lemmatize(word, pos)
                    lemmatized.append(lemma)
                except Exception:
                    # Fallback to noun lemmatization
                    lemma = self._lemmatizer.lemmatize(word)
                    lemmatized.append(lemma)
            return ' '.join(lemmatized)
            
        elif self._lemmatizer_type == 'spacy':
            # spaCy lemmatizer
            doc = self._lemmatizer(text)
            return ' '.join([token.lemma_ for token in doc])
            
        elif self._lemmatizer_type == 'basic':
            # Basic rule-based lemmatizer
            tokens = text.split()
            lemmatized = []
            for word in tokens:
                lemma = word
                # Apply rules in order of specificity
                for suffix, replacement in sorted(self._basic_lemma_rules.items(), 
                                                key=lambda x: len(x[0]), reverse=True):
                    if word.endswith(suffix) and len(word) > len(suffix):
                        lemma = word[:-len(suffix)] + replacement
                        break
                lemmatized.append(lemma)
            return ' '.join(lemmatized)
            
        else:
            return text

    # ------------------------------------------------------------------
    # Dataset-level processing with configurable pipeline
    # ------------------------------------------------------------------

    def process_dataset(
        self,
        df,
        clean_text: bool = False,
        remove_stop_words: bool = False,
        lemmatize: bool = False,
        text_cols: Iterable[str] = ('resume_text', 'job_description_text'),
    ):
        """Process entire dataset with configurable text processing pipeline"""
        df_clean = df.copy()

        steps = []
        
        if clean_text:
            steps.append(self.clean_text)

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


# Example usage
if __name__ == "__main__":
    # Initialize with stop words and lemmatizer enabled
    preprocessor = TextPreprocessor(enable_stopwords=True, enable_lemmatizer=True, lemmatizer_type='nltk')
    
    # Test text
    test_text = "I have excellent experiences working with Python and JavaScript in various projects"
    
    print("Original:", test_text)
    print("Cleaned:", preprocessor.clean_text(test_text))
    print("Without stop words:", preprocessor.remove_stop_words(preprocessor.clean_text(test_text)))
    print("Lemmatized:", preprocessor.lemmatize_text(preprocessor.clean_text(test_text)))
    
    # Full pipeline
    cleaned = preprocessor.clean_text(test_text)
    no_stopwords = preprocessor.remove_stop_words(cleaned)
    lemmatized = preprocessor.lemmatize_text(no_stopwords)
    print("Full pipeline:", lemmatized)
    
    # Test different lemmatizers
    print("\n--- Testing Different Lemmatizers ---")
    
    # NLTK lemmatizer
    preprocessor_nltk = TextPreprocessor(enable_lemmatizer=True, lemmatizer_type='nltk')
    print("NLTK:", preprocessor_nltk.lemmatize_text("running cats wolves better"))
    
    # spaCy lemmatizer
    preprocessor_spacy = TextPreprocessor(enable_lemmatizer=True, lemmatizer_type='spacy')
    print("spaCy:", preprocessor_spacy.lemmatize_text("running cats wolves better"))
    
    # Basic lemmatizer
    preprocessor_basic = TextPreprocessor(enable_lemmatizer=True, lemmatizer_type='basic')
    print("Basic:", preprocessor_basic.lemmatize_text("running cats wolves better"))