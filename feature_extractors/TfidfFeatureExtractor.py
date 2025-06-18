from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from .BaseFeatureExtractor import BaseFeatureExtractor

class TfidfFeatureExtractor(BaseFeatureExtractor):
    """
    TF-IDF Feature Extractor
    """
    
    def __init__(self, 
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 3,
                 max_df: float = 0.85,
                 **kwargs):
        
        config = {
            'max_features': max_features,
            'ngram_range': ngram_range,
            'min_df': min_df,
            'max_df': max_df,
            **kwargs
        }
        
        super().__init__("TF-IDF", **config)
    
    def _create_vectorizer(self):
        return TfidfVectorizer(
            max_features=self.config.get('max_features', 5000),
            stop_words=self.config.get('stop_words', 'english'),
            ngram_range=self.config.get('ngram_range', (1, 2)),
            min_df=self.config.get('min_df', 3),
            max_df=self.config.get('max_df', 0.85),
            lowercase=self.config.get('lowercase', True),
            strip_accents=self.config.get('strip_accents', 'ascii'),
            token_pattern=self.config.get('token_pattern', r'\b[a-zA-Z][a-zA-Z0-9]*\b'),
            use_idf=self.config.get('use_idf', True),
            smooth_idf=self.config.get('smooth_idf', True),
            sublinear_tf=self.config.get('sublinear_tf', True)
        )
