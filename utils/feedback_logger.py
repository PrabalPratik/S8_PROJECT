"""
Feedback Logger - Human-Grounded Evaluation

Collects and stores human feedback on JD quality and ranking accuracy
to enable human-grounded evaluation of the AI system.

Security: Path validation and data sanitization to prevent injection.
"""

import os
import csv
import json
import html
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd


class FeedbackLogger:
    """
    Logs human feedback to CSV for analysis.
    
    Feedback types:
    - jd_quality: Thumbs up/down on generated JDs
    - ranking_accuracy: Thumbs up/down on candidate rankings
    - correction_helpful: Whether suggested corrections were useful
    """
    
    def __init__(self, log_path: str = "data/feedback_log.csv"):
        """
        Initialize the Feedback Logger.
        
        Args:
            log_path: Path to CSV file for storing feedback
        """
        self.log_path = self._validate_log_path(log_path)
        self._ensure_log_exists()
    
    def _validate_log_path(self, log_path: str) -> str:
        """
        Validate and normalize the log file path.
        
        Security: Prevents path traversal attacks.
        """
        # Normalize path
        normalized = os.path.normpath(log_path)
        
        # Prevent absolute paths or traversal
        if os.path.isabs(normalized) or '..' in normalized:
            print(f"Warning: Invalid log_path '{log_path}', using default")
            normalized = os.path.join('data', 'feedback_log.csv')
        
        return normalized
    
    def _ensure_log_exists(self) -> None:
        """Create log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_path):
            # Create directory if needed
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            
            # Create file with headers
            with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'feedback_type',
                    'rating',  # 1 = thumbs up, -1 = thumbs down
                    'context_role',
                    'context_skills',
                    'context_candidate',
                    'hri_score',
                    'drift_score',
                    'additional_notes'
                ])
    
    def log_feedback(
        self,
        feedback_type: str,
        rating: int,
        context: Optional[Dict[str, Any]] = None,
        notes: str = ""
    ) -> bool:
        """
        Log a feedback entry.
        
        Args:
            feedback_type: Type of feedback (jd_quality, ranking_accuracy, etc.)
            rating: 1 for thumbs up, -1 for thumbs down
            context: Optional context dict with role, skills, etc.
            notes: Optional additional notes
            
        Returns:
            True if logged successfully
        """
        if context is None:
            context = {}
        
        try:
            # Sanitize all string fields to prevent injection
            safe_role = self._sanitize_field(context.get('role', ''))
            safe_skills = self._sanitize_field(context.get('skills', ''))
            safe_candidate = self._sanitize_field(context.get('candidate', ''))
            safe_notes = self._sanitize_field(notes)
            
            with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    feedback_type,
                    rating,
                    safe_role,
                    safe_skills,
                    safe_candidate,
                    context.get('hri_score', ''),
                    context.get('drift_score', ''),
                    safe_notes
                ])
            return True
        except Exception as e:
            print(f"Feedback logging failed: {e}")
            return False
    
    def _sanitize_field(self, value: str) -> str:
        """
        Sanitize a field for safe CSV storage.
        
        Security: Prevents CSV formula injection and XSS if displayed.
        """
        if not value:
            return ""
        
        value = str(value)
        
        # Escape HTML entities
        value = html.escape(value)
        
        # Prevent CSV formula injection
        dangerous_prefixes = ('=', '+', '-', '@', '\t', '\r')
        if value.startswith(dangerous_prefixes):
            value = "'" + value
        
        return value
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of collected feedback.
        
        Returns:
            Dict with feedback counts and averages
        """
        try:
            df = pd.read_csv(self.log_path)
        except Exception:
            return {"error": "No feedback data available"}
        
        if df.empty:
            return {"total": 0, "message": "No feedback collected yet"}
        
        summary = {
            "total_feedback": len(df),
            "by_type": {},
            "overall_satisfaction": 0.0
        }
        
        # Aggregate by type
        for ftype in df['feedback_type'].unique():
            type_df = df[df['feedback_type'] == ftype]
            positive = len(type_df[type_df['rating'] == 1])
            negative = len(type_df[type_df['rating'] == -1])
            total = positive + negative
            
            summary["by_type"][ftype] = {
                "positive": positive,
                "negative": negative,
                "total": total,
                "satisfaction_rate": positive / total if total > 0 else 0.0
            }
        
        # Overall satisfaction
        total_positive = len(df[df['rating'] == 1])
        summary["overall_satisfaction"] = total_positive / len(df) if len(df) > 0 else 0.0
        
        return summary
    
    def get_recent_feedback(self, n: int = 10) -> List[Dict]:
        """
        Get n most recent feedback entries.
        
        Args:
            n: Number of entries to return
            
        Returns:
            List of feedback dicts
        """
        try:
            df = pd.read_csv(self.log_path)
            return df.tail(n).to_dict('records')
        except Exception:
            return []
    
    def format_summary(self, summary: Dict) -> str:
        """Generate human-readable summary."""
        output = "\n" + "=" * 40 + "\n"
        output += "📊 FEEDBACK SUMMARY\n"
        output += "=" * 40 + "\n\n"
        
        output += f"Total Feedback: {summary.get('total_feedback', 0)}\n"
        output += f"Overall Satisfaction: {summary.get('overall_satisfaction', 0):.1%}\n\n"
        
        for ftype, stats in summary.get("by_type", {}).items():
            output += f"📌 {ftype}:\n"
            output += f"   👍 {stats['positive']} | 👎 {stats['negative']}\n"
            output += f"   Satisfaction: {stats['satisfaction_rate']:.1%}\n\n"
        
        return output


# Singleton instance for easy access
_logger_instance = None

def get_feedback_logger(log_path: str = "data/feedback_log.csv") -> FeedbackLogger:
    """Get or create the singleton FeedbackLogger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = FeedbackLogger(log_path)
    return _logger_instance


# Example usage
if __name__ == "__main__":
    print("=" * 40)
    print("Feedback Logger - Test")
    print("=" * 40)
    
    logger = FeedbackLogger("data/test_feedback.csv")
    
    # Log some test feedback
    logger.log_feedback(
        "jd_quality",
        1,  # Thumbs up
        {"role": "Data Scientist", "skills": "Python, ML", "hri_score": 0.15}
    )
    
    logger.log_feedback(
        "ranking_accuracy",
        -1,  # Thumbs down
        {"role": "Backend Engineer", "candidate": "Alice Chen"}
    )
    
    # Get summary
    summary = logger.get_feedback_summary()
    print(logger.format_summary(summary))
