"""
Role Drift Detector - Novel Feature F4

Detects when a generated JD semantically drifts from the original role.
Prevents role contamination (e.g., ML Engineer → Data Engineer drift).
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional, List


class RoleDriftDetector:
    """
    Detects semantic drift between intended role and generated job description.
    
    Uses sentence embeddings to measure cosine similarity between:
    - Role description/title
    - Generated job description
    
    Low similarity indicates potential role drift/contamination.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Role Drift Detector.
        
        Args:
            model_name: HuggingFace model for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
        
        # Role-specific keywords for enhanced detection
        self.role_keywords = {
            "data scientist": ["machine learning", "statistics", "python", "modeling", "analysis", "data science"],
            "data engineer": ["pipeline", "etl", "spark", "airflow", "data warehouse", "infrastructure"],
            "ml engineer": ["deploy", "mlops", "model serving", "production", "kubernetes", "inference"],
            "software engineer": ["development", "coding", "architecture", "design patterns", "testing"],
            "frontend engineer": ["react", "javascript", "css", "ui", "ux", "web"],
            "backend engineer": ["api", "server", "database", "microservices", "rest"],
            "devops engineer": ["ci/cd", "docker", "kubernetes", "infrastructure", "automation"],
            "product manager": ["roadmap", "stakeholder", "requirements", "strategy", "prioritization"],
        }
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading Role Drift Detector model ({self.model_name})...")
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
        except ImportError:
            print("Warning: sentence-transformers not installed. Using fallback similarity.")
            self.model = None
        except Exception as e:
            print(f"Warning: Could not load drift detector model: {e}")
            self.model = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding for text."""
        if self.model is None:
            # Fallback: simple TF-based embedding
            words = text.lower().split()
            embedding = np.zeros(384)
            for i, word in enumerate(words[:384]):
                embedding[i % 384] += hash(word) % 1000 / 1000
            return embedding / (len(words) + 1)
        
        return self.model.encode(text, convert_to_numpy=True)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def detect_drift(
        self,
        role: str,
        generated_jd: str,
        mandatory_skills: Optional[str] = None
    ) -> Tuple[float, Dict]:
        """
        Detect semantic drift between role and generated JD.
        
        Args:
            role: Target role name/description
            generated_jd: Generated job description text
            mandatory_skills: Optional skills to incorporate into role context
            
        Returns:
            Tuple of (drift_score, details_dict)
            drift_score: 0.0 = perfect alignment, 1.0 = severe drift
        """
        if not role or not generated_jd:
            return 0.5, {"error": "Empty role or JD provided"}
        
        # Build role context
        role_context = f"{role}"
        if mandatory_skills:
            role_context += f" {mandatory_skills}"
        
        # Get role keywords if available
        role_lower = role.lower()
        keywords = []
        for known_role, kw_list in self.role_keywords.items():
            if known_role in role_lower or role_lower in known_role:
                keywords = kw_list
                role_context += " " + " ".join(kw_list)
                break
        
        # Get embeddings
        role_embedding = self._get_embedding(role_context)
        jd_embedding = self._get_embedding(generated_jd)
        
        # Calculate similarity
        similarity = self._cosine_similarity(role_embedding, jd_embedding)
        
        # Drift score is inverse of similarity
        drift_score = 1.0 - similarity
        drift_score = max(0.0, min(1.0, drift_score))
        
        # Check for keyword presence as additional signal
        jd_lower = generated_jd.lower()
        keyword_hits = sum(1 for kw in keywords if kw in jd_lower) if keywords else 0
        keyword_coverage = keyword_hits / len(keywords) if keywords else 1.0
        
        # Adjust drift score based on keyword coverage
        adjusted_drift = drift_score * (1.0 - keyword_coverage * 0.3)
        
        # Determine drift level and warning
        if adjusted_drift < 0.2:
            drift_level = "🟢 Aligned"
            warning = None
        elif adjusted_drift < 0.4:
            drift_level = "🟡 Minor Drift"
            warning = f"JD may slightly deviate from {role} focus"
        elif adjusted_drift < 0.6:
            drift_level = "🟠 Moderate Drift"
            warning = f"Review JD for alignment with {role} requirements"
        else:
            drift_level = "🔴 Significant Drift"
            warning = f"JD appears to drift from {role} - consider regeneration"
        
        # Detect potential contamination (which other role it might drift towards)
        contaminated_role = None
        if adjusted_drift > 0.3:
            contaminated_role = self._detect_contamination(generated_jd, role_lower)
        
        details = {
            "drift_score": round(adjusted_drift, 3),
            "drift_level": drift_level,
            "similarity": round(similarity, 3),
            "keyword_coverage": round(keyword_coverage, 2),
            "role_keywords": keywords,
            "contaminated_role": contaminated_role,
            "warning": warning
        }
        
        return adjusted_drift, details
    
    def _detect_contamination(self, jd: str, original_role: str) -> Optional[str]:
        """Detect if JD has drifted towards a different role."""
        jd_lower = jd.lower()
        
        max_score = 0
        detected_role = None
        
        for role, keywords in self.role_keywords.items():
            if role in original_role or original_role in role:
                continue
            
            hits = sum(1 for kw in keywords if kw in jd_lower)
            score = hits / len(keywords)
            
            if score > max_score and score > 0.3:
                max_score = score
                detected_role = role
        
        return detected_role
    
    def bulk_detect(
        self,
        role: str,
        jds: List[str],
        threshold: float = 0.4
    ) -> List[Dict]:
        """
        Detect drift for multiple JDs.
        
        Args:
            role: Target role
            jds: List of generated JDs
            threshold: Drift threshold for flagging
            
        Returns:
            List of results with drift scores
        """
        results = []
        for i, jd in enumerate(jds):
            drift_score, details = self.detect_drift(role, jd)
            results.append({
                "index": i,
                "drift_score": drift_score,
                "flagged": drift_score >= threshold,
                **details
            })
        return results


def format_drift_report(drift_score: float, details: Dict) -> str:
    """Generate a formatted drift analysis report."""
    report = "\n" + "=" * 50 + "\n"
    report += "🎯 Role Drift Analysis\n"
    report += "=" * 50 + "\n\n"
    
    # Drift bar visualization
    alignment = int((1 - drift_score) * 20)
    bar = "█" * alignment + "░" * (20 - alignment)
    
    report += f"📊 Alignment: {bar} {(1-drift_score)*100:.0f}%\n"
    report += f"🔄 Drift Score: {drift_score:.2f}\n"
    report += f"📌 Status: {details.get('drift_level', 'Unknown')}\n\n"
    
    if details.get('warning'):
        report += f"⚠️ Warning: {details['warning']}\n\n"
    
    if details.get('contaminated_role'):
        report += f"🔀 Detected drift towards: {details['contaminated_role'].title()}\n"
    
    if details.get('role_keywords'):
        report += f"\n📋 Role Keywords Checked:\n"
        for kw in details['role_keywords']:
            report += f"   • {kw}\n"
    
    return report


# Example usage
if __name__ == "__main__":
    print("=" * 50)
    print("Role Drift Detector - Test")
    print("=" * 50)
    
    detector = RoleDriftDetector()
    
    # Test 1: Good alignment
    role = "Data Scientist"
    good_jd = """
    We are looking for a Data Scientist to join our team. You will work on 
    machine learning models, statistical analysis, and data science projects.
    Experience with Python, pandas, and scikit-learn required.
    """
    
    drift, details = detector.detect_drift(role, good_jd, "Python, Machine Learning")
    print("\n📗 Test 1: Good Alignment")
    print(format_drift_report(drift, details))
    
    # Test 2: Role drift
    drifted_jd = """
    We need someone to build data pipelines and ETL processes. 
    Experience with Spark, Airflow, and data warehouse solutions required.
    You will focus on data infrastructure and pipeline automation.
    """
    
    drift, details = detector.detect_drift(role, drifted_jd, "Python, Machine Learning")
    print("\n📕 Test 2: Role Drift")
    print(format_drift_report(drift, details))
