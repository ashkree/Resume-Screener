import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Type, Any, Optional


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for feature extractors
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.vectorizer = None
        self.is_fitted = False
        self.fit_time = 0.0
        self.transform_time = 0.0
        self.feature_names = None
        self.config = kwargs
        
    @abstractmethod
    def _create_vectorizer(self):
        """Create the specific vectorizer instance"""
        pass
    
    def fit(self, texts: List[str]) -> 'BaseFeatureExtractor':
        """
        Fit the feature extractor on training texts
        
        Args:
            texts: List of text documents to fit on
            
        Returns:
            self: Returns the fitted extractor
        """
        if not texts:
            raise ValueError("Empty text list provided")
            
        print(f"Fitting {self.name} on {len(texts)} documents...")
        
        self.vectorizer = self._create_vectorizer()
        
        start_time = time.time()
        self.vectorizer.fit(texts)
        self.fit_time = time.time() - start_time
        
        self.is_fitted = True
        
        # Get feature names if available
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"✅ {self.name} fitted in {self.fit_time:.2f} seconds")
        return self
    
    def transform(self, texts: List[str]):
        """
        Transform texts to feature matrix
        
        Args:
            texts: List of text documents to transform
            
        Returns:
            Sparse matrix of features
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before transform")
            
        start_time = time.time()
        X = self.vectorizer.transform(texts)
        self.transform_time = time.time() - start_time
        
        return X
    
    def fit_transform(self, texts: List[str]):
        """
        Fit and transform in one step
        
        Args:
            texts: List of text documents
            
        Returns:
            Sparse matrix of features
        """
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self) -> Optional[np.ndarray]:
        """Get feature names if available"""
        return self.feature_names
    
    def get_config(self) -> Dict[str, Any]:
        """Get extractor configuration"""
        return {
            'name': self.name,
            'fitted': self.is_fitted,
            'fit_time': self.fit_time,
            'transform_time': self.transform_time,
            'config': self.config
        }
