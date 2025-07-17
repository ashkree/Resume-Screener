import pandas as pd
import numpy as np
import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import sparse
import re
import pickle
import os
from tqdm import tqdm
import time

class SkillNERPreprocessor:
    """Preprocess data with SkillNER extraction outside the pipeline"""
    
    def __init__(self, model="en_core_web_md", cache_dir="./skillner_cache"):
        self.model = model
        self.cache_dir = cache_dir
        self.nlp = None
        self.skill_extractor = None
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
    
    def _init_extractor(self):
        """Initialize SkillNER extractor"""
        if self.skill_extractor is None:
            print(f"Loading spaCy model: {self.model}")
            self.nlp = spacy.load(self.model)
            self.skill_extractor = SkillExtractor(self.nlp, SKILL_DB, PhraseMatcher(self.nlp.vocab))
            print("SkillNER extractor initialized")
    
    def _clean_text(self, text):
        """Clean text for SkillNER processing"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove problematic characters
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\/\@\#\$\%\&\*\+\=\<\>\'\"]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text or text.isspace():
            return "No content available"
        
        # Limit length
        if len(text) > 10000:
            text = text[:10000]
        
        return text
    
    def _extract_skills_safe(self, text):
        """Safely extract skills from text"""
        try:
            cleaned_text = self._clean_text(text)
            result = self.skill_extractor.annotate(cleaned_text)
            
            # Get detailed skill information
            full_matches = result.get("results", {}).get("full_matches", [])
            ngram_scored = result.get("results", {}).get("ngram_scored", [])
            
            return {
                'skill_count': len(full_matches) + len(ngram_scored),
                'full_matches': len(full_matches),
                'ngram_scored': len(ngram_scored),
                'skills': [skill.get('skill_name', skill.get('doc_node_value', '')) 
                          for skill in full_matches + ngram_scored]
            }
            
        except Exception as e:
            print(f"Skill extraction error: {e}")
            return {
                'skill_count': 0,
                'full_matches': 0,
                'ngram_scored': 0,
                'skills': []
            }
    
    def _get_cache_path(self, data_hash):
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"skillner_features_{data_hash}.pkl")
    
    def _hash_data(self, df):
        """Create hash of dataframe for caching"""
        import hashlib
        
        # Create hash from text columns
        text_data = df[['resume_text', 'job_description_text']].to_string()
        return hashlib.md5(text_data.encode()).hexdigest()[:16]
    
    def preprocess_data(self, df, force_recompute=False):
        """
        Preprocess dataframe to extract SkillNER features
        
        Args:
            df: DataFrame with 'resume_text' and 'job_description_text' columns
            force_recompute: If True, ignore cache and recompute
        
        Returns:
            DataFrame with additional SkillNER feature columns
        """
        
        # Check cache first
        data_hash = self._hash_data(df)
        cache_path = self._get_cache_path(data_hash)
        
        if not force_recompute and os.path.exists(cache_path):
            print(f"Loading SkillNER features from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                skill_features = pickle.load(f)
        else:
            print("Computing SkillNER features...")
            self._init_extractor()
            
            skill_features = []
            start_time = time.time()
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting skills"):
                # Extract skills from resume
                resume_skills = self._extract_skills_safe(row['resume_text'])
                
                # Extract skills from job description
                jd_skills = self._extract_skills_safe(row['job_description_text'])
                
                # Combine features
                features = {
                    'resume_skill_count': resume_skills['skill_count'],
                    'resume_full_matches': resume_skills['full_matches'],
                    'resume_ngram_scored': resume_skills['ngram_scored'],
                    'jd_skill_count': jd_skills['skill_count'],
                    'jd_full_matches': jd_skills['full_matches'],
                    'jd_ngram_scored': jd_skills['ngram_scored'],
                    'skill_overlap': len(set(resume_skills['skills']).intersection(set(jd_skills['skills']))),
                    'skill_match_ratio': len(set(resume_skills['skills']).intersection(set(jd_skills['skills']))) / max(len(resume_skills['skills']), 1)
                }
                
                skill_features.append(features)
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                pickle.dump(skill_features, f)
            
            elapsed = time.time() - start_time
            print(f"SkillNER preprocessing completed in {elapsed:.2f} seconds")
        
        # Add features to dataframe
        skill_df = pd.DataFrame(skill_features)
        result_df = pd.concat([df.reset_index(drop=True), skill_df], axis=1)
        
        return result_df
    
class PrecomputedSkillNERTransformer(BaseEstimator, TransformerMixin):
    """Transformer that uses precomputed SkillNER features"""
    
    def __init__(self, feature_columns=None):
        if feature_columns is None:
            self.feature_columns = [
                'resume_skill_count', 'jd_skill_count', 
                'skill_overlap', 'skill_match_ratio'
            ]
        else:
            self.feature_columns = feature_columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            # If X is numpy array, assume it's already the features we need
            return sparse.csr_matrix(X, dtype=np.float32)
        
        # Extract the specified feature columns
        features = X[self.feature_columns].values
        return sparse.csr_matrix(features, dtype=np.float32)