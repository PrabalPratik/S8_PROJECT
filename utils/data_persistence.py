"""
Data Persistence Module - Store JDs and Rankings

Saves generated job descriptions and candidate rankings to CSV
for persistence across sessions.
"""

import os
import csv
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


class DataPersistence:
    """
    Handles persistent storage for:
    - Generated job descriptions
    - Candidate rankings
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.jd_log_path = os.path.join(data_dir, "generated_jds.csv")
        self.rankings_log_path = os.path.join(data_dir, "ranking_history.csv")
        
        self._ensure_files_exist()
    
    def _ensure_files_exist(self) -> None:
        """Create log files with headers if they don't exist."""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # JD log
        if not os.path.exists(self.jd_log_path):
            with open(self.jd_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'role', 'experience_years', 
                    'mandatory_skills', 'optional_skills',
                    'jd_text', 'hri_score', 'drift_score'
                ])
        
        # Rankings log
        if not os.path.exists(self.rankings_log_path):
            with open(self.rankings_log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'role', 'mandatory_skills',
                    'candidate_name', 'global_score', 'semantic_match', 'risk_score'
                ])
    
    def save_jd(
        self,
        role: str,
        experience: int,
        mandatory_skills: str,
        optional_skills: str,
        jd_text: str,
        hri_score: Optional[float] = None,
        drift_score: Optional[float] = None
    ) -> bool:
        """
        Save a generated job description.
        
        Returns:
            True if saved successfully
        """
        try:
            with open(self.jd_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    role,
                    experience,
                    mandatory_skills,
                    optional_skills,
                    jd_text.replace('\n', '\\n'),  # Escape newlines
                    hri_score if hri_score else '',
                    drift_score if drift_score else ''
                ])
            return True
        except Exception as e:
            print(f"Failed to save JD: {e}")
            return False
    
    def save_rankings(
        self,
        role: str,
        mandatory_skills: str,
        rankings_df: pd.DataFrame
    ) -> bool:
        """
        Save candidate rankings.
        
        Args:
            role: Job role
            mandatory_skills: Required skills
            rankings_df: DataFrame with Candidate, Global Score, Semantic Match, Risk columns
            
        Returns:
            True if saved successfully
        """
        try:
            timestamp = datetime.now().isoformat()
            
            with open(self.rankings_log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for _, row in rankings_df.iterrows():
                    writer.writerow([
                        timestamp,
                        role,
                        mandatory_skills,
                        row.get('Candidate', ''),
                        row.get('Global Score', 0),
                        row.get('Semantic Match', 0),
                        row.get('Risk', 0)
                    ])
            return True
        except Exception as e:
            print(f"Failed to save rankings: {e}")
            return False
    
    def get_jd_history(self, limit: int = 20) -> pd.DataFrame:
        """Get recent generated JDs."""
        try:
            df = pd.read_csv(self.jd_log_path)
            return df.tail(limit)
        except Exception:
            return pd.DataFrame()
    
    def get_ranking_history(self, limit: int = 50) -> pd.DataFrame:
        """Get recent ranking history."""
        try:
            df = pd.read_csv(self.rankings_log_path)
            return df.tail(limit)
        except Exception:
            return pd.DataFrame()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics."""
        try:
            jd_df = pd.read_csv(self.jd_log_path)
            rankings_df = pd.read_csv(self.rankings_log_path)
            
            return {
                "total_jds_generated": len(jd_df),
                "unique_roles": jd_df['role'].nunique() if not jd_df.empty else 0,
                "total_rankings": len(rankings_df),
                "candidates_evaluated": rankings_df['candidate_name'].nunique() if not rankings_df.empty else 0,
                "avg_hri_score": jd_df['hri_score'].mean() if 'hri_score' in jd_df.columns and not jd_df.empty else None
            }
        except Exception:
            return {"error": "No data available"}


# Singleton instance
_persistence_instance = None

def get_data_persistence(data_dir: str = "data") -> DataPersistence:
    """Get or create the singleton DataPersistence instance."""
    global _persistence_instance
    if _persistence_instance is None:
        _persistence_instance = DataPersistence(data_dir)
    return _persistence_instance


if __name__ == "__main__":
    print("=" * 40)
    print("Data Persistence - Test")
    print("=" * 40)
    
    dp = DataPersistence("data")
    
    # Test saving JD
    dp.save_jd(
        role="Test Engineer",
        experience=5,
        mandatory_skills="Python, Testing",
        optional_skills="Selenium",
        jd_text="This is a test JD...",
        hri_score=0.15
    )
    
    # Get stats
    stats = dp.get_stats()
    print(f"\nStats: {stats}")
