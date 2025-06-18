
import pandas as pd
from typing import List, Dict, Any

import time

from .BaseFeatureExtractor import BaseFeatureExtractor

class FeatureExtractionPipeline:
    """
    Pipeline for managing multiple feature extractors
    """
    
    def __init__(self, text_combiner_func=None):
        self.extractors: Dict[str, BaseFeatureExtractor] = {}
        self.results: Dict[str, Dict] = {}
        self.text_combiner = text_combiner_func or self._default_text_combiner
        
    def _default_text_combiner(self, resume_text: str, job_desc_text: str) -> str:
        """Default text combination strategy"""
        return f"{resume_text} [SEP] {job_desc_text}"
    
    def add_extractor(self, extractor: BaseFeatureExtractor) -> 'FeatureExtractionPipeline':
        """
        Add a feature extractor to the pipeline
        
        Args:
            extractor: Feature extractor instance
            
        Returns:
            self: Returns the pipeline for chaining
        """
        self.extractors[extractor.name] = extractor
        print(f"✅ Added {extractor.name} to pipeline")
        return self
    
    def prepare_texts(self, df: pd.DataFrame) -> List[str]:
        """
        Prepare combined texts from dataframe
        
        Args:
            df: DataFrame with 'resume_text' and 'job_description_text' columns
            
        Returns:
            List of combined texts
        """
        combined_texts = []
        for _, row in df.iterrows():
            combined = self.text_combiner(
                str(row['resume_text']), 
                str(row['job_description_text'])
            )
            combined_texts.append(combined)
        return combined_texts
    
    def fit_all(self, train_df: pd.DataFrame) -> 'FeatureExtractionPipeline':
        """
        Fit all extractors on training data
        
        Args:
            train_df: Training dataframe
            
        Returns:
            self: Returns the pipeline for chaining
        """
        if not self.extractors:
            raise ValueError("No extractors added to pipeline")
        
        print(f"🔧 Preparing training texts...")
        train_texts = self.prepare_texts(train_df)
        print(f"✅ Prepared {len(train_texts)} training texts")
        
        print(f"\n🚀 Fitting {len(self.extractors)} extractors...")
        for name, extractor in self.extractors.items():
            try:
                extractor.fit(train_texts)
                print(f"   ✓ {name} fitted successfully")
            except Exception as e:
                print(f"   ✗ {name} failed: {str(e)}")
        
        return self
    
    def transform_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Transform data using all fitted extractors
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Dictionary with results for each extractor
        """
        texts = self.prepare_texts(df)
        results = {}
        
        print(f"🔄 Transforming {len(texts)} texts with {len(self.extractors)} extractors...")
        
        for name, extractor in self.extractors.items():
            if not extractor.is_fitted:
                print(f"   ⚠️ Skipping {name} (not fitted)")
                continue
                
            try:
                start_time = time.time()
                X = extractor.transform(texts)
                transform_time = time.time() - start_time
                
                # Calculate statistics
                density = X.nnz / (X.shape[0] * X.shape[1]) if X.shape[0] > 0 and X.shape[1] > 0 else 0
                memory_mb = X.data.nbytes / 1024 / 1024 if hasattr(X, 'data') else 0
                
                results[name] = {
                    'features': X,
                    'shape': X.shape,
                    'density': density,
                    'memory_mb': memory_mb,
                    'transform_time': transform_time,
                    'feature_names': extractor.get_feature_names(),
                    'extractor': extractor
                }
                
                print(f"   ✓ {name}: {X.shape}, density={density:.4f}, {transform_time:.2f}s")
                
            except Exception as e:
                print(f"   ✗ {name} transform failed: {str(e)}")
        
        return results
    
    def extract_features(self, 
                        train_df: pd.DataFrame, 
                        val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete feature extraction pipeline
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            
        Returns:
            Complete results dictionary
        """
        print("="*60)
        print("FEATURE EXTRACTION PIPELINE")
        print("="*60)
        
        # Fit on training data
        self.fit_all(train_df)
        
        # Transform both datasets
        print(f"\n--- TRANSFORMING TRAINING DATA ---")
        train_results = self.transform_all(train_df)
        
        print(f"\n--- TRANSFORMING VALIDATION DATA ---")
        val_results = self.transform_all(val_df)
        
        # Combine results
        combined_results = {}
        for name in train_results:
            if name in val_results:
                combined_results[name] = {
                    'train': train_results[name],
                    'val': val_results[name],
                    'extractor': train_results[name]['extractor']
                }
        
        print(f"\n✅ Feature extraction completed for {len(combined_results)} extractors")
        
        self.results = combined_results
        return combined_results
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for all extractors
        
        Returns:
            DataFrame with comparison metrics
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for name, result in self.results.items():
            train_data = result['train']
            extractor = result['extractor']
            
            summary_data.append({
                'Extractor': name,
                'Train_Shape': str(train_data['shape']),
                'Features': train_data['shape'][1] if len(train_data['shape']) > 1 else 0,
                'Density': train_data['density'],
                'Memory_MB': train_data['memory_mb'],
                'Fit_Time_s': extractor.fit_time,
                'Transform_Time_s': train_data['transform_time']
            })
        
        return pd.DataFrame(summary_data)
